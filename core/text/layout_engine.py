import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import skia
import uharfbuzz as hb

from core.text.text_processing import (
    STYLE_PATTERN,
    find_optimal_breaks_dp,
    find_optimal_breaks_contour_dp,
    parse_styled_segments,
    tokenize_styled_text,
    try_hyphenate_word,
)
from utils.exceptions import RenderingError
from utils.logging import log_message

# Epsilon to guard rounding when converting from HarfBuzz 26.6 fixed-point.
VISUAL_WIDTH_EPSILON = 0.0


def shape_line(
    text_line: str, hb_font: hb.Font, features: Dict[str, bool]
) -> Tuple[List[hb.GlyphInfo], List[hb.GlyphPosition], str]:
    """Shapes a line of text with HarfBuzz.

    Returns:
        Tuple of (glyph_infos, glyph_positions, direction) where direction
        is "ltr" or "rtl" as detected by HarfBuzz.

    Raises:
        RenderingError: If HarfBuzz shaping fails
    """
    hb_buffer = hb.Buffer()
    hb_buffer.add_str(text_line)
    hb_buffer.guess_segment_properties()
    direction = str(hb_buffer.direction)
    try:
        hb.shape(hb_font, hb_buffer, features)
        return hb_buffer.glyph_infos, hb_buffer.glyph_positions, direction
    except Exception as e:
        log_message(f"HarfBuzz shaping failed: {e}", always_print=True)
        raise RenderingError("HarfBuzz text shaping failed") from e


def calculate_line_width(positions: List[hb.GlyphPosition]) -> float:
    """Calculate visual width using advances and first/last x_offset."""
    if not positions:
        return 0.0
    HB_26_6_SCALE_FACTOR = 64.0

    total_advance_fixed = sum(pos.x_advance for pos in positions)
    first_offset_fixed = positions[0].x_offset
    last_offset_fixed = positions[-1].x_offset

    visual_width_fixed = total_advance_fixed + (last_offset_fixed - first_offset_fixed)
    visual_width = float(visual_width_fixed / HB_26_6_SCALE_FACTOR)
    return visual_width + VISUAL_WIDTH_EPSILON


def calculate_styled_line_width(
    line_with_markers: str,
    font_size: int,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features: Dict[str, bool],
) -> float:
    """Calculate the width of a line that may contain style markers.

    Uses the appropriate HarfBuzz faces per style segment, falling back to
    the 'regular' face if a style-specific face is missing.
    """
    if not line_with_markers:
        return 0.0

    segments = parse_styled_segments(line_with_markers)
    if not segments:
        return 0.0

    regular_face = loaded_hb_faces.get("regular")
    if regular_face is None:
        return 0.0

    total_advance_fixed_all = 0
    first_offset_fixed_global: Optional[int] = None
    last_offset_fixed_global: Optional[int] = None

    for segment_text, style_name in segments:
        hb_face_to_use = (
            loaded_hb_faces.get(style_name)
            if style_name in ("regular", "italic", "bold", "bold_italic")
            else None
        ) or regular_face

        hb_font_segment = hb.Font(hb_face_to_use)
        hb_font_segment.ptem = float(font_size)
        # Standard HarfBuzz scaling: font_size * 64 (for 26.6 fixed point coordinates)
        hb_scale = int(font_size * 64)
        hb_font_segment.scale = (hb_scale, hb_scale)

        _, positions, _ = shape_line(segment_text, hb_font_segment, features)
        if not positions:
            continue

        total_advance_fixed_all += sum(pos.x_advance for pos in positions)
        if first_offset_fixed_global is None:
            first_offset_fixed_global = positions[0].x_offset
        last_offset_fixed_global = positions[-1].x_offset

    if total_advance_fixed_all == 0 and first_offset_fixed_global is None:
        return 0.0

    HB_26_6_SCALE_FACTOR = 64.0
    offset_delta_fixed = 0
    if first_offset_fixed_global is not None and last_offset_fixed_global is not None:
        offset_delta_fixed = last_offset_fixed_global - first_offset_fixed_global

    visual_width_fixed_all = total_advance_fixed_all + offset_delta_fixed
    visual_width_all = float(visual_width_fixed_all / HB_26_6_SCALE_FACTOR)
    return visual_width_all + VISUAL_WIDTH_EPSILON


def check_fit(
    font_size: int,
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    line_spacing_mult: float,
    hyphenate_before_scaling: bool,
    hyphen_penalty: float,
    hyphenation_min_word_length: int,
    badness_exponent: float,
    word_width_cache: Optional[Dict[Tuple[str, int], float]] = None,
    verbose: bool = False,
    detach_trailing_ellipsis: bool = True,
    collision_mask: Optional[np.ndarray] = None,
    target_center: Optional[Tuple[float, float]] = None,
    mask_offset: Tuple[float, float] = (0.0, 0.0),
) -> Optional[Dict]:
    """Check if text fits within the given dimensions at the specified font size.

    Args:
        font_size: Font size to test
        text: Text to wrap and measure
        max_render_width: Maximum allowed width
        max_render_height: Maximum allowed height
        regular_hb_face: HarfBuzz face for shaping
        regular_typeface: Skia typeface for metrics
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        features_to_enable: HarfBuzz features to enable
        line_spacing_mult: Line spacing multiplier
        hyphenate_before_scaling: Whether to hyphenate before scaling
        hyphen_penalty: Penalty for hyphenated lines
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        word_width_cache: Optional cache for word widths
        verbose: Whether to print detailed logs
        detach_trailing_ellipsis: Whether to treat trailing ellipsis as separate token
        collision_mask: Optional binary mask of the safe area for contour wrapping
        target_center: Optional (x, y) coordinates of the text optical center for contour wrapping

    Returns:
        Dict containing fit data if successful, None if doesn't fit
    """
    try:
        hb_font = hb.Font(regular_hb_face)
        hb_font.ptem = float(font_size)

        # Standard HarfBuzz scaling: font_size * 64 (for 26.6 fixed point coordinates)
        hb_scale = int(font_size * 64)
        hb_font.scale = (hb_scale, hb_scale)

        skia_font_test = skia.Font(regular_typeface, font_size)
        try:
            metrics = skia_font_test.getMetrics()
            single_line_height = (
                -metrics.fAscent + metrics.fDescent + metrics.fLeading
            ) * line_spacing_mult
            if single_line_height <= 0:
                single_line_height = font_size * 1.2 * line_spacing_mult
        except Exception as e:
            if verbose:
                log_message(
                    f"Font metrics unavailable at size {font_size}: {e}",
                    verbose=verbose,
                )
            single_line_height = font_size * 1.2 * line_spacing_mult

        # Respect explicit newlines as hard line breaks (e.g., for vertical stacking)
        if "\n" in text:
            explicit_lines = text.split("\n")
            current_max_line_width = 0.0
            lines_data_at_size = []
            for line_text in explicit_lines:
                width = calculate_styled_line_width(
                    line_text, font_size, loaded_hb_faces, features_to_enable
                )
                lines_data_at_size.append(
                    {"text_with_markers": line_text, "width": width}
                )
                current_max_line_width = max(current_max_line_width, width)

            total_block_height = (-metrics.fAscent + metrics.fDescent) + (
                len(explicit_lines) - 1
            ) * single_line_height

            if (
                current_max_line_width <= max_render_width
                and total_block_height <= max_render_height
            ):
                return {
                    "lines": lines_data_at_size,
                    "metrics": metrics,
                    "max_line_width": current_max_line_width,
                    "line_height": single_line_height,
                }
            return None

        tokens: List[Tuple[str, bool]] = tokenize_styled_text(
            text, detach_trailing_ellipsis
        )
        augmented_tokens: List[str] = []

        if hyphenate_before_scaling:
            for token_text, is_styled in tokens:
                marker = ""
                core_text = token_text

                if is_styled:
                    styled_match = STYLE_PATTERN.match(token_text)
                    if not styled_match:
                        augmented_tokens.append(token_text)
                        continue
                    marker = styled_match.group(1)
                    core_text = styled_match.group(2)

                match = re.match(r"^(\W*)([\w\-]+)(\W*)$", core_text)
                if match:
                    core_word_length = len(match.group(2))
                else:
                    core_word_length = len(core_text)

                if core_word_length > hyphenation_min_word_length:
                    word_width = calculate_styled_line_width(
                        token_text, font_size, loaded_hb_faces, features_to_enable
                    )

                    if word_width > max_render_width:

                        def wrap_part(part: str) -> str:
                            return f"{marker}{part}{marker}" if marker else part

                        def width_test_func(part: str) -> bool:
                            wrapped = wrap_part(part)
                            w = calculate_styled_line_width(
                                wrapped, font_size, loaded_hb_faces, features_to_enable
                            )
                            return w <= max_render_width

                        split_parts = try_hyphenate_word(
                            core_text, hyphenation_min_word_length, width_test_func
                        )
                        if split_parts:
                            augmented_tokens.extend(wrap_part(p) for p in split_parts)
                        else:
                            augmented_tokens.append(token_text)
                    else:
                        augmented_tokens.append(token_text)
                else:
                    augmented_tokens.append(token_text)
        else:
            augmented_tokens = [t for t, _ in tokens]

        try:
            GLUE_TRAILING_PUNCT_RE = re.compile(r"^[,.;:!?…]+$")
            GLUE_CLOSERS_RE = re.compile(r"^[\)\]\}\u2019\u201D\'\"]+$")

            def _glue_trailing_punctuation(
                tokens_list: List[str], _detach: bool = True
            ) -> List[str]:
                glued: List[str] = []
                for tok in tokens_list:
                    match = STYLE_PATTERN.match(tok)
                    content = match.group(2) if match else tok

                    # Skip gluing for disconnected ellipsis to allow wrapping
                    if _detach and re.match(
                        r"^(\.{2,})[\)\]\}\u2019\u201D\'\"]*$", content
                    ):
                        glued.append(tok)
                        continue

                    if glued and (
                        GLUE_TRAILING_PUNCT_RE.match(content)
                        or GLUE_CLOSERS_RE.match(content)
                    ):
                        glued[-1] = glued[-1] + tok
                    else:
                        glued.append(tok)
                return glued

            augmented_tokens = _glue_trailing_punctuation(
                augmented_tokens, detach_trailing_ellipsis
            )
        except Exception:
            pass

        def word_width_func(word: str) -> float:
            if word_width_cache is not None:
                cached_key = (word, font_size)
                if cached_key in word_width_cache:
                    return word_width_cache[cached_key]

            width_val = calculate_styled_line_width(
                word, font_size, loaded_hb_faces, features_to_enable
            )

            if word_width_cache is not None:
                word_width_cache[(word, font_size)] = width_val

            return width_val

        space_width = calculate_styled_line_width(
            " ", font_size, loaded_hb_faces, features_to_enable
        )

        wrapped_lines_text = None

        if collision_mask is not None and target_center is not None:
            # TRUE CONTOUR WRAPPING MODE
            target_center_x, target_center_y = target_center
            mask_h, mask_w = collision_mask.shape

            best_contour_lines = None
            best_contour_centers = None
            best_contour_cost = float('inf')

            # Test all possible line counts from 1 to min(N, max_height/line_height)
            max_lines_by_height = max(1, int(max_render_height / single_line_height))
            max_k = min(len(augmented_tokens), max_lines_by_height)

            for K in range(1, max_k + 1):
                total_height = K * single_line_height
                start_y = target_center_y - total_height / 2.0

                # If the entire block exceeds the mask vertically, skip
                if start_y < 0 or start_y + total_height > mask_h:
                    continue

                anchor_k = int((target_center_y - start_y) / single_line_height)
                anchor_k = max(0, min(K - 1, anchor_k))

                max_widths_for_k = [0.0] * K
                line_centers_for_k = [float(target_center_x)] * K
                valid_k = True

                def compute_line_bounds(k_idx: int, prev_cx: int) -> Tuple[int, int, int]:
                    # Adjust start_y by offset
                    rel_start_y = start_y - mask_offset[1]
                    y_points = [
                        int(max(0, min(mask_h - 1, rel_start_y + (k_idx + 0.2) * single_line_height))),
                        int(max(0, min(mask_h - 1, rel_start_y + (k_idx + 0.5) * single_line_height))),
                        int(max(0, min(mask_h - 1, rel_start_y + (k_idx + 0.8) * single_line_height)))
                    ]

                    max_lw = 0
                    min_rw = mask_w - 1
                    curr_cx = int(prev_cx - mask_offset[0])

                    if curr_cx < 0 or curr_cx >= mask_w:
                        return -1, -1, -1

                    for py in y_points:
                        if collision_mask[py, curr_cx] == 0:
                            safe_pixels = np.where(collision_mask[py, :] == 255)[0]
                            if safe_pixels.size == 0:
                                return -1, -1, -1
                            curr_cx = int(safe_pixels[np.argmin(np.abs(safe_pixels - curr_cx))])

                        left_zeros = np.where(collision_mask[py, 0:curr_cx] == 0)[0]
                        lw = int(left_zeros.max()) if left_zeros.size > 0 else 0

                        right_zeros = np.where(collision_mask[py, curr_cx:] == 0)[0]
                        rw = int(right_zeros.min() + curr_cx) if right_zeros.size > 0 else mask_w - 1

                        max_lw = max(max_lw, lw)
                        min_rw = min(min_rw, rw)

                    if max_lw >= min_rw:
                        return -1, -1, -1

                    # Absolute max_lw and min_rw
                    abs_max_lw = max_lw + mask_offset[0]
                    abs_min_rw = min_rw + mask_offset[0]

                    if abs_min_rw <= abs_max_lw:
                        return -1, -1, -1

                    # Center is simply the midpoint of the available space
                    new_center = (abs_max_lw + abs_min_rw) / 2

                    return int(abs_max_lw) + 1, int(abs_min_rw) - 1, int(new_center)
                # Downward trace from anchor
                current_cx = int(target_center_x)
                if current_cx < 0 or current_cx >= mask_w:
                    current_cx = max(0, min(mask_w - 1, current_cx))

                for k in range(anchor_k, K):
                    lw, rw, new_cx = compute_line_bounds(k, current_cx)
                    if new_cx == -1:
                        valid_k = False
                        break
                    max_widths_for_k[k] = float(rw - lw)
                    line_centers_for_k[k] = float(new_cx)
                    current_cx = new_cx

                # Upward trace from anchor
                if valid_k and anchor_k > 0:
                    current_cx = int(line_centers_for_k[anchor_k])
                    for k in range(anchor_k - 1, -1, -1):
                        lw, rw, new_cx = compute_line_bounds(k, current_cx)
                        if new_cx == -1:
                            valid_k = False
                            break
                        max_widths_for_k[k] = float(rw - lw)
                        line_centers_for_k[k] = float(new_cx)
                        current_cx = new_cx

                if not valid_k:
                    continue

                result = find_optimal_breaks_contour_dp(
                    augmented_tokens,
                    max_widths_for_k,
                    word_width_func,
                    space_width,
                    badness_exponent,
                    hyphen_penalty,
                    detach_trailing_ellipsis,
                )

                if result:
                    lines_text, cost = result
                    if cost < best_contour_cost:
                        best_contour_cost = cost
                        best_contour_lines = lines_text
                        best_contour_centers = line_centers_for_k.copy()

            wrapped_lines_text = best_contour_lines
        else:
            # FALLBACK RECTANGULAR MODE
            wrapped_lines_text = find_optimal_breaks_dp(
                augmented_tokens,
                max_render_width,
                word_width_func,
                space_width,
                badness_exponent,
                hyphen_penalty,
                detach_trailing_ellipsis,
            )

        if not wrapped_lines_text:
            return None

        current_max_line_width = 0
        lines_data_at_size = []
        for i, line_text_with_markers in enumerate(wrapped_lines_text):
            width = calculate_styled_line_width(
                line_text_with_markers, font_size, loaded_hb_faces, features_to_enable
            )

            if collision_mask is not None and target_center is not None and 'best_contour_centers' in locals() and best_contour_centers is not None:
                center_x = best_contour_centers[i]
            else:
                center_x = target_center[0] if target_center else 0.0

            lines_data_at_size.append(
                {"text_with_markers": line_text_with_markers, "width": width, "center_x": float(center_x)}
            )
            current_max_line_width = max(current_max_line_width, width)
        for line in lines_data_at_size:
            line["target_width"] = current_max_line_width
        total_block_height = (-metrics.fAscent + metrics.fDescent) + (
            len(wrapped_lines_text) - 1
        ) * single_line_height

        if verbose:
            log_message(
                f"Size {font_size}: {current_max_line_width:.0f}x{total_block_height:.0f} "
                f"(max {max_render_width:.0f}x{max_render_height:.0f})",
                verbose=verbose,
            )

        fits_bounds = (
            current_max_line_width <= max_render_width
            and total_block_height <= max_render_height
        )

        if fits_bounds or (collision_mask is not None and target_center is not None):
            has_collision = False
            if collision_mask is not None and target_center is not None:
                has_collision = _check_collision(
                    lines_data_at_size,
                    target_center,
                    collision_mask,
                    single_line_height,
                    mask_offset,
                )

            if not has_collision:
                if verbose:
                    log_message(f"Size {font_size} fits", verbose=verbose)
                return {
                    "lines": lines_data_at_size,
                    "metrics": metrics,
                    "max_line_width": current_max_line_width,
                    "line_height": single_line_height,
                }

        return None

    except Exception as e:
        if verbose:
            log_message(f"Fit check failed at size {font_size}: {e}", verbose=verbose)
        return None


def _check_collision(
    lines_data: List[Dict],
    target_center: Tuple[float, float],
    collision_mask: np.ndarray,
    line_height: float,
    mask_offset: Tuple[float, float] = (0.0, 0.0),
) -> bool:
    """
    Check if any text pixel overlaps with background (0) in mask.

    Args:
        lines_data: List of dictionaries containing line width and text.
        target_center: (x, y) coordinates of the true optical center.
        collision_mask: Binary mask of the safe area (0=background, 255=bubble).
        line_height: Height of a single line of text.

    Returns:
        True if collision detected, False otherwise.
    """
    target_center_x, target_center_y = target_center
    mask_h, mask_w = collision_mask.shape

    total_text_height = len(lines_data) * line_height
    start_y = target_center_y - total_text_height / 2.0

    current_y = start_y
    for line in lines_data:
        line_w = line["width"]
        line_cx = line.get("center_x", target_center_x)
        line_x = line_cx - line_w / 2.0

        y1, y2 = int(current_y), int(current_y + line_height)
        x1, x2 = int(line_x), int(line_x + line_w)

        # Inset corners to allow text to naturally fit into curved bubbles without extreme shrinking.
        # Since the mask is already padded (config.padding_pixels), we don't need strict rectangular corners.
        inset_x = int(line_w * 0.1)
        inset_y = int(line_height * 0.2)

        points_to_check = [
            (x1 + inset_x, y1 + inset_y), (x2 - inset_x, y1 + inset_y),
            (x1 + inset_x, y2 - inset_y), (x2 - inset_x, y2 - inset_y),
            (x1 + int(line_w / 2), y1), (x1 + int(line_w / 2), y2),
            (x1, y1 + int(line_height / 2)), (x2, y1 + int(line_height / 2))
        ]

        for px, py in points_to_check:
            # Apply mask offset before checking collision mask
            px = int(px - mask_offset[0])
            py = int(py - mask_offset[1])

            if px < 0 or px >= mask_w or py < 0 or py >= mask_h:
                continue

            if collision_mask[py, px] == 0:
                return True

        current_y += line_height

    return False


def find_optimal_layout(
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    min_font_size: int = 8,
    max_font_size: int = 16,
    line_spacing_mult: float = 1.0,
    hyphenate_before_scaling: bool = True,
    hyphen_penalty: float = 1000.0,
    hyphenation_min_word_length: int = 8,
    badness_exponent: float = 3.0,
    verbose: bool = False,
    bubble_id: Optional[str] = None,
    collision_mask: Optional[np.ndarray] = None,
    target_center: Optional[Tuple[float, float]] = None,
    detach_trailing_ellipsis: bool = True,
    mask_offset: Tuple[float, float] = (0.0, 0.0),
) -> Dict:
    """Find the optimal font size and layout for text within given dimensions.

    Uses binary search to find the largest font size that fits.

    Args:
        text: Text to layout
        max_render_width: Maximum allowed width
        max_render_height: Maximum allowed height
        regular_hb_face: HarfBuzz face for the regular font
        regular_typeface: Skia typeface for the regular font
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        features_to_enable: HarfBuzz features to enable
        min_font_size: Minimum font size to try
        max_font_size: Maximum font size to try
        line_spacing_mult: Line spacing multiplier
        hyphenate_before_scaling: Whether to hyphenate before reducing font size
        hyphen_penalty: Penalty for hyphenated lines
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        verbose: Whether to print detailed logs
        bubble_id: Optional identifier for the bubble (for logging purposes)
        collision_mask: Optional binary mask of the bubble's safe area for collision detection
        target_center: Optional (x, y) coordinates of the text's target optical center

    Returns:
        Dictionary containing layout data (font_size, lines, metrics, etc.)

    Raises:
        RenderingError: If text doesn't fit at minimum font size or layout fails
    """
    # Preserve explicit newlines if present (e.g., vertical stacking),
    # otherwise collapse whitespace for normal paragraph layout
    if "\n" in text or "\r" in text:
        clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    else:
        clean_text = " ".join(text.split())
    if not clean_text:
        raise RenderingError("Empty text cannot be laid out")

    best_fit_size = -1
    best_fit_lines_data = None
    best_fit_metrics = None
    best_fit_max_line_width = float("inf")
    best_fit_line_height = 0.0

    word_width_cache: Dict[Tuple[str, int], float] = {}

    low = 1
    # Smart font size: dynamically scale up to the container's shortest dimension
    high = int(min(max_render_width, max_render_height))

    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            break

        log_message(f"Testing size {mid}", verbose=verbose)

        succeeded_at_current_size = False

        fit_data = check_fit(
            mid,
            clean_text,
            max_render_width,
            max_render_height,
            regular_hb_face,
            regular_typeface,
            loaded_hb_faces,
            features_to_enable,
            line_spacing_mult,
            hyphenate_before_scaling,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            word_width_cache,
            verbose,
            detach_trailing_ellipsis,
            collision_mask=collision_mask,
            target_center=target_center,
            mask_offset=mask_offset,
        )

        if fit_data is not None:
            has_collision = False
            if collision_mask is not None and target_center is not None:
                has_collision = _check_collision(
                    fit_data["lines"],
                    target_center,
                    collision_mask,
                    fit_data["line_height"],
                    mask_offset,
                )

            if not has_collision:
                # REJECT layouts that degenerate into vertical columns if it's supposed to be horizontal text
                # i.e., if max_line_width is very small compared to line_height, it means 1 char per line.
                # Only apply this penalty if there are multiple characters and multiple lines.
                is_degenerate_vertical = False
                if len(fit_data["lines"]) > 1 and len(clean_text) > 1:
                    if len(clean_text) <= 3:
                        # For 2 or 3 characters, any wrapping makes it look vertical. Force 1 line.
                        is_degenerate_vertical = True
                    elif fit_data["max_line_width"] < fit_data["line_height"] * 2.0:
                        # For longer text, require at least ~2 characters per line to look horizontal
                        is_degenerate_vertical = True
                
                if is_degenerate_vertical:
                    has_collision = True # Treat as collision to force smaller font
                    
            if not has_collision:
                best_fit_size = mid
                best_fit_lines_data = fit_data["lines"]
                best_fit_metrics = fit_data["metrics"]
                best_fit_max_line_width = fit_data["max_line_width"]
                best_fit_line_height = fit_data["line_height"]
                succeeded_at_current_size = True

        if succeeded_at_current_size:
            low = mid + 1
        else:
            high = mid - 1

    if best_fit_size == -1:
        log_message(
            f"Text too large for bubble even at size 1: '{clean_text[:30]}'. Forcing overflow layout.",
            always_print=True,
        )
        # Force a layout at size 4 with large bounds so Skia can still render it without DP breaking
        forced_fit = check_fit(
            4,
            clean_text,
            99999.0,
            99999.0,
            regular_hb_face,
            regular_typeface,
            loaded_hb_faces,
            features_to_enable,
            line_spacing_mult,
            hyphenate_before_scaling,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            word_width_cache,
            verbose,
            detach_trailing_ellipsis,
            collision_mask=None,
            target_center=target_center,
            mask_offset=mask_offset,
        )

        if forced_fit is not None:
            return {
                "font_size": 4,
                "lines": forced_fit["lines"],
                "metrics": forced_fit["metrics"],
                "max_line_width": forced_fit["max_line_width"],
                "line_height": forced_fit["line_height"],
            }
        else:
            raise RenderingError(
                "Text too large for bubble and fallback failed"
            )

    return {
        "font_size": best_fit_size,
        "lines": best_fit_lines_data,
        "metrics": best_fit_metrics,
        "max_line_width": best_fit_max_line_width,
        "line_height": best_fit_line_height,
    }
