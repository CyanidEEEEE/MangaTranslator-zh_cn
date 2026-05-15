from typing import Optional, Tuple

import numpy as np
import skia
from PIL import Image

from core.config import RenderingConfig
from core.image.image_utils import calculate_centroid_expansion_box
from core.text.drawing_engine import (
    draw_layout,
    load_font_resources,
    pil_to_skia_surface,
    skia_surface_to_pil,
)
from core.text.font_manager import (
    find_font_variants,
    get_font_features,
    sanitize_text_for_font,
)
from core.text.layout_engine import find_optimal_layout
from core.text.text_processing import is_rtl_script, parse_styled_segments
from utils.exceptions import FontError, ImageProcessingError, RenderingError
from utils.logging import log_message

GRAYSCALE_MIDPOINT = 128  # Threshold for determining text color
FALLBACK_PADDING_RATIO = 0.15  # 8% padding ratio when safe area calculation fails
AUTO_VERTICAL_MIN_ASPECT_RATIO = 1.6
AUTO_VERTICAL_MAX_CHARS = 12
AUTO_VERTICAL_MAX_WORDS = 1
AUTO_VERTICAL_MAX_HORIZONTAL_FILL = 0.45
AUTO_VERTICAL_MIN_FILL_GAIN = 0.20


def _plain_text_for_layout_policy(text: str) -> str:
    return "".join(segment_text for segment_text, _ in parse_styled_segments(text))


def _should_try_auto_vertical_text(
    text: str,
    max_render_width: float,
    max_render_height: float,
) -> bool:
    if max_render_width <= 0 or max_render_height <= 0:
        return False
    if max_render_height / max_render_width < AUTO_VERTICAL_MIN_ASPECT_RATIO:
        return False

    plain_text = _plain_text_for_layout_policy(text).strip()
    if not plain_text or is_rtl_script(plain_text):
        return False

    non_space_chars = sum(1 for ch in plain_text if not ch.isspace())
    word_count = len(plain_text.split())
    return (
        non_space_chars <= AUTO_VERTICAL_MAX_CHARS
        and word_count <= AUTO_VERTICAL_MAX_WORDS
    )


def _auto_vertical_layout_is_better(
    horizontal_layout: dict,
    vertical_layout: dict,
    max_render_height: float,
) -> bool:
    if max_render_height <= 0:
        return False

    horizontal_block_height = horizontal_layout.get("block_height") or 0.0
    vertical_block_height = vertical_layout.get("block_height") or 0.0
    if horizontal_block_height <= 0 or vertical_block_height <= 0:
        return False

    horizontal_fill = horizontal_block_height / max_render_height
    vertical_fill = vertical_block_height / max_render_height
    return (
        vertical_layout["font_size"] >= horizontal_layout["font_size"]
        and horizontal_fill <= AUTO_VERTICAL_MAX_HORIZONTAL_FILL
        and vertical_fill >= horizontal_fill + AUTO_VERTICAL_MIN_FILL_GAIN
    )


def render_text_skia(
    pil_image: Image.Image,
    text: str,
    bbox: Tuple[int, int, int, int],
    font_dir: str,
    cleaned_mask: Optional[np.ndarray] = None,
    extended_collision_mask: Optional[np.ndarray] = None,
    bubble_color_bgr: Optional[Tuple[int, int, int]] = (255, 255, 255),
    config: Optional[RenderingConfig] = None,
    raise_on_safe_error: bool = False,
    verbose: bool = False,
    bubble_id: Optional[str] = "1",
    rotation_deg: float = 0.0,
    vertical_stack: bool = False,
    text_color_rgb: Optional[Tuple[int, int, int]] = None,
    text_background_color: Optional[Tuple[int, int, int]] = None,
    stroke_color_rgb: Optional[Tuple[int, int, int]] = None,
    stroke_width_ratio: float = 0.0,
    layout_only: bool = False,
    text_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    """
    Fits and renders text within a bounding box using Skia and HarfBuzz.

    This is the high-level orchestrator that coordinates:
    1. Font loading (font_manager)
    2. Safe area calculation (image_utils)
    3. Layout optimization (layout_engine)
    4. Text rendering (drawing_engine)

    Uses the 5-step Distance Transform Insetting Method for safe area calculation:
    1. Establish Safe Zone: Create safe_area_mask using cv2.distanceTransform()
    2. Find Unbiased Anchor: Calculate centroid of the safe_area_mask
    3. Measure Available Space: Ray cast from centroid to find distances to edges
    4. Calculate Symmetrical Dimensions: Use min distances for width/height
    5. Construct Final Box: Create centered rectangle within safe zone

    This ensures text is perfectly centered and never touches bubble boundaries.

    Args:
        pil_image: PIL Image object to draw onto.
        text: Text to render.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        font_dir: Directory containing font files.
        cleaned_mask: Binary mask of the cleaned bubble (0/255). Used for safe area calculation.
        bubble_color_bgr: Background color of the bubble (BGR tuple). Used to determine text color.
        config: RenderingConfig object containing all rendering parameters. If None, uses defaults.
        verbose: Whether to print detailed logs.

    Returns:
        Modified PIL Image object with rendered text.

    Raises:
        RenderingError: If rendering fails due to invalid inputs, font issues, or layout problems
        FontError: If font loading fails
    """
    x1, y1, x2, y2 = bbox
    bubble_width = x2 - x1
    bubble_height = y2 - y1

    if bubble_width <= 0 or bubble_height <= 0:
        log_message(f"Invalid bbox dimensions: {bbox}", always_print=True)
        raise RenderingError(f"Invalid bounding box dimensions: {bbox}")

    # em dash can break wrapping
    normalized_text = text.replace("—", "-")

    clean_text = " ".join(normalized_text.split())
    if not clean_text:
        return pil_image

    layout_text = clean_text

    # Initialize config with defaults if not provided
    if config is None:
        config = RenderingConfig()

    target_center_x = None
    target_center_y = None
    safe_area_result = None
    safe_area_mask_for_collision = None
    safe_area_fallback_logged = False
    if cleaned_mask is not None:
        try:
            safe_area_result = calculate_centroid_expansion_box(
                cleaned_mask, padding_pixels=config.padding_pixels, verbose=verbose
            )
        except ImageProcessingError:
            # Safe area calculation failed, will use fallback below
            safe_area_result = None
            if raise_on_safe_error:
                raise
            log_message(
                "Safe area calculation failed, falling back to padded bbox method",
                verbose=verbose,
            )
            safe_area_fallback_logged = True

    if safe_area_result is not None:
        guaranteed_box, centroid, _ = safe_area_result

        if extended_collision_mask is not None:
            try:
                extended_safe_area_result = calculate_centroid_expansion_box(
                    extended_collision_mask, padding_pixels=config.padding_pixels, verbose=verbose
                )
                safe_area_mask_for_collision = extended_safe_area_result[2]
                box_x, box_y, box_w, box_h = extended_safe_area_result[0]
                target_center_x, target_center_y = box_x + box_w / 2.0, box_y + box_h / 2.0
                log_message("Using extended collision mask for conjoined bubble safe area", verbose=verbose)
            except ImageProcessingError:
                safe_area_mask_for_collision = safe_area_result[2]
                box_x, box_y, box_w, box_h = guaranteed_box
                target_center_x, target_center_y = box_x + box_w / 2.0, box_y + box_h / 2.0
                log_message("Extended safe area calculation failed, falling back to primary safe area", verbose=verbose)
        else:
            safe_area_mask_for_collision = safe_area_result[2]
            box_x, box_y, box_w, box_h = guaranteed_box
            target_center_x, target_center_y = box_x + box_w / 2.0, box_y + box_h / 2.0

        if text_bbox is not None and len(text_bbox) == 4:
            tx1, ty1, tx2, ty2 = [float(v) for v in text_bbox]
            target_center_x = tx1 + (tx2 - tx1) / 2.0
            target_center_y = ty1 + (ty2 - ty1) / 2.0
            log_message("Using original text_bbox centroid for visual centering", verbose=verbose)

        # When collision mask is available, use the mask's actual extent for sizing
        # instead of the guaranteed inscribed rectangle, which is overly conservative
        # for irregular shapes (starburst, spiky bubbles). The collision mask will
        # still enforce the true shape constraint during contour wrapping in layout.
        sizing_w, sizing_h = box_w, box_h
        expansion_ratio = 1.0
        if safe_area_mask_for_collision is not None:
            mask_ys, mask_xs = np.where(safe_area_mask_for_collision > 0)
            if len(mask_ys) > 0:
                mask_extent_w = float(mask_xs.max() - mask_xs.min())
                mask_extent_h = float(mask_ys.max() - mask_ys.min())
                # Use the larger of guaranteed box and mask extent for sizing
                # This prevents starburst bubbles from being overly constrained
                sizing_w = max(box_w, mask_extent_w)
                sizing_h = max(box_h, mask_extent_h)
                if sizing_w > box_w or sizing_h > box_h:
                    log_message(
                        f"Expanded sizing from guaranteed box {box_w:.0f}x{box_h:.0f} "
                        f"to mask extent {sizing_w:.0f}x{sizing_h:.0f}",
                        verbose=verbose,
                    )
                    # When the mask extent is significantly larger than the guaranteed
                    # box (e.g., starburst bubbles where centroid was moved to pole of
                    # inaccessibility), the box center is unreliable. Recenter to the
                    # mask's geometric center for better text placement.
                    expansion_ratio = max(
                        sizing_w / max(1, box_w), sizing_h / max(1, box_h)
                    )
                    if text_bbox is None and expansion_ratio > 1.3:
                        mask_center_x = float(mask_xs.min() + mask_xs.max()) / 2.0
                        mask_center_y = float(mask_ys.min() + mask_ys.max()) / 2.0
                        log_message(
                            f"Recentering target from ({target_center_x:.0f}, {target_center_y:.0f}) "
                            f"to mask center ({mask_center_x:.0f}, {mask_center_y:.0f})",
                            verbose=verbose,
                        )
                        target_center_x = mask_center_x
                        target_center_y = mask_center_y

        # Compute fill ratio to determine if this is a rectangle or an ellipse
        if expansion_ratio > 1.3:
            # For highly irregular shapes (starburst, spiky), the fill_ratio over the
            # full extent is naturally low and shouldn't penalize width. The collision
            # mask's contour wrapping already prevents text from exceeding bubble bounds,
            # so use a generous flat width_factor as the sizing is just a search bound.
            fill_ratio = 0.85
            width_factor = 0.85
        else:
            # Normal bubbles: compute fill ratio over the mask region
            if safe_area_mask_for_collision is not None and len(mask_ys) > 0:
                fill_cx = int(round((mask_xs.min() + mask_xs.max()) / 2.0))
                fill_cy = int(round((mask_ys.min() + mask_ys.max()) / 2.0))
                fill_hw = int(round(sizing_w / 2.0))
                fill_hh = int(round(sizing_h / 2.0))
                crop_y0 = max(0, fill_cy - fill_hh)
                crop_y1_end = min(safe_area_mask_for_collision.shape[0], fill_cy + fill_hh)
                crop_x0 = max(0, fill_cx - fill_hw)
                crop_x1_end = min(safe_area_mask_for_collision.shape[1], fill_cx + fill_hw)
                mask_crop = safe_area_mask_for_collision[crop_y0:crop_y1_end, crop_x0:crop_x1_end]
            else:
                mask_crop = safe_area_mask_for_collision[box_y:box_y+box_h, box_x:box_x+box_w]
            if mask_crop.size > 0:
                fill_ratio = np.count_nonzero(mask_crop) / mask_crop.size
            else:
                fill_ratio = 1.0

            # Map fill_ratio: 1.0 (rectangle) -> 0.95, 0.785 (ellipse) -> 0.75
            width_factor = 0.75 + (fill_ratio - 0.785) * (0.95 - 0.75) / (1.0 - 0.785)
            width_factor = max(0.65, min(0.95, width_factor))

        # Give wide bubbles a bit more width allowance
        bubble_aspect = sizing_w / max(1, sizing_h)
        if bubble_aspect > 1.2:
            width_factor = min(0.98, width_factor * 1.15)

        max_render_width = float(sizing_w) * width_factor
        max_render_height = float(sizing_h) * 0.9
        log_message(f"Using centroid-based safe area. Fill ratio: {fill_ratio:.2f}, width factor: {width_factor:.2f}", verbose=verbose)
    else:
        # 修复方案：让回退逻辑也使用用户设置的绝对像素值
        if not safe_area_fallback_logged:
            log_message(
                f"Safe area calculation failed, falling back to padded bbox (padding: {config.padding_pixels}px)",
                verbose=verbose,
            )
            
        # 增加保护：Padding 不能超过气泡宽度的 40%，防止减成负数
        effective_padding_w = min(config.padding_pixels, bubble_width * 0.4)
        effective_padding_h = min(config.padding_pixels, bubble_height * 0.4)
        
        max_render_width = bubble_width - (2 * effective_padding_w)
        max_render_height = bubble_height - (2 * effective_padding_h)

        if max_render_width <= 0 or max_render_height <= 0:
            # 最后的保底逻辑
            max_render_width = max(1.0, float(bubble_width) * 0.9)
            max_render_height = max(1.0, float(bubble_height) * 0.9)

        target_center_x = x1 + bubble_width / 2.0
        target_center_y = y1 + bubble_height / 2.0

    try:
        font_variants = find_font_variants(font_dir, verbose=verbose)
        regular_font_path = font_variants.get("regular")
    except FontError as e:
        raise RenderingError(f"Font loading failed: {e}") from e

    layout_text = sanitize_text_for_font(
        layout_text, str(regular_font_path), verbose=verbose
    )
    if not layout_text.strip():
        log_message(
            "All text characters unsupported by font, skipping render",
            always_print=True,
        )
        return pil_image

    try:
        _, regular_typeface, regular_hb_face = load_font_resources(
            str(regular_font_path)
        )
    except FontError as e:
        raise RenderingError(f"Font resource loading failed: {e}") from e

    available_features = get_font_features(str(regular_font_path))
    features_to_enable = {
        "kern": "kern" in available_features["GPOS"],
        "liga": config.use_ligatures and "liga" in available_features["GSUB"],
        "calt": "calt" in available_features["GSUB"],
    }
    log_message(
        f"Font features: {[k for k, v in features_to_enable.items() if v]}",
        verbose=verbose,
    )

    # Pre-load all required font variants for layout engine
    preload_hb_faces = {"regular": regular_hb_face}
    for style_key in ["italic", "bold", "bold_italic"]:
        style_path = font_variants.get(style_key)
        if style_path:
            _, _typeface, _hb_face = load_font_resources(str(style_path))
            if _hb_face:
                preload_hb_faces[style_key] = _hb_face

    detach_trailing_punctuation = getattr(
        config,
        "detach_trailing_punctuation",
        getattr(config, "detach_trailing_ellipsis", True),
    )

    def _find_layout(use_vertical_stack: bool) -> dict:
        return find_optimal_layout(
            layout_text,
            max_render_width,
            max_render_height,
            regular_hb_face,
            regular_typeface,
            preload_hb_faces,
            features_to_enable,
            config.min_font_size,
            config.max_font_size,
            config.line_spacing_mult,
            False if use_vertical_stack else config.hyphenate_before_scaling,
            config.hyphen_penalty,
            config.hyphenation_min_word_length,
            config.badness_exponent,
            verbose,
            bubble_id,
            (
                safe_area_mask_for_collision
                if safe_area_mask_for_collision is not None
                else cleaned_mask
            ),
            (target_center_x, target_center_y),
            detach_trailing_punctuation,
            mask_offset=(0.0, 0.0),
            vertical_stack=use_vertical_stack,
        )

    try:
        layout_data = _find_layout(vertical_stack)
    except RenderingError as e:
        if not (
            getattr(config, "auto_vertical_text", False)
            and not vertical_stack
            and _should_try_auto_vertical_text(
                layout_text, max_render_width, max_render_height
            )
        ):
            raise RenderingError(f"Layout optimization failed: {e}") from e

        try:
            layout_data = _find_layout(True)
            log_message(
                "Using auto vertical text layout after horizontal layout failed",
                verbose=verbose,
            )
        except RenderingError:
            raise RenderingError(f"Layout optimization failed: {e}") from e
    else:
        if (
            getattr(config, "auto_vertical_text", False)
            and not vertical_stack
            and _should_try_auto_vertical_text(
                layout_text, max_render_width, max_render_height
            )
        ):
            try:
                vertical_layout_data = _find_layout(True)
                if _auto_vertical_layout_is_better(
                    layout_data, vertical_layout_data, max_render_height
                ):
                    layout_data = vertical_layout_data
                    log_message("Using auto vertical text layout", verbose=verbose)
            except RenderingError:
                pass

    if layout_only:
        log_message(f"Rendered at size {layout_data['font_size']}", verbose=verbose)
        result = Image.new("RGBA", (1, 1))
        result.info["font_size"] = layout_data["font_size"]
        return result

    required_styles = {"regular"} | {
        style for _, style in parse_styled_segments(clean_text)
    }
    log_message(f"Required styles: {sorted(required_styles)}", verbose=verbose)

    loaded_typefaces = {"regular": regular_typeface}
    loaded_hb_faces = {"regular": regular_hb_face}

    for style in ["italic", "bold", "bold_italic"]:
        if style in required_styles:
            font_path = font_variants.get(style)
            if font_path:
                log_message(f"Loading {style}: {font_path.name}", verbose=verbose)
                _, typeface, hb_face = load_font_resources(str(font_path))
                if typeface and hb_face:
                    loaded_typefaces[style] = typeface
                    loaded_hb_faces[style] = hb_face
                else:
                    log_message(
                        f"Failed to load {style} variant, using regular",
                        verbose=verbose,
                    )
            else:
                log_message(
                    f"Style '{style}' not found, using regular",
                    verbose=verbose,
                )

    # Determine text color contrast based on sampled background brightness
    text_color = skia.ColorBLACK
    if text_color_rgb is not None:
        text_color = skia.Color(text_color_rgb[0], text_color_rgb[1], text_color_rgb[2])
    elif bubble_color_bgr is not None:
        try:
            # bubble_color_bgr may be a grayscale proxy; treat as BGR
            bg_brightness = (
                bubble_color_bgr[0] + bubble_color_bgr[1] + bubble_color_bgr[2]
            ) / 3.0
            # If background is dark, use white text; if light, use black
            text_color = (
                skia.ColorWHITE
                if bg_brightness < GRAYSCALE_MIDPOINT
                else skia.ColorBLACK
            )
        except Exception:
            text_color = skia.ColorBLACK

    skia_bg_color = None
    if text_background_color is not None:
        skia_bg_color = skia.Color(
            text_background_color[0], text_background_color[1], text_background_color[2]
        )

    skia_outline_color = None
    if stroke_color_rgb is not None:
        skia_outline_color = skia.Color(
            stroke_color_rgb[0], stroke_color_rgb[1], stroke_color_rgb[2]
        )

    final_outline_width = config.outline_width
    if stroke_width_ratio > 0.0:
        final_outline_width = max(final_outline_width, stroke_width_ratio * layout_data["font_size"])

    # Apply supersampling if enabled
    if config.supersampling_factor > 1:
        log_message(
            f"Using supersampling factor {config.supersampling_factor}", verbose=verbose
        )

        # Crop the bbox region from the original image
        # Ensure bbox coordinates are within image bounds
        img_width, img_height = pil_image.size
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(img_width, x2)
        crop_y2 = min(img_height, y2)

        cropped_region = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1

        # Upscale the cropped region
        factor = config.supersampling_factor
        scaled_width = int(crop_width * factor)
        scaled_height = int(crop_height * factor)
        upscaled_region = cropped_region.resize(
            (scaled_width, scaled_height), Image.Resampling.LANCZOS
        )

        # Scale coordinates relative to bbox origin
        scaled_target_center_x = (target_center_x - crop_x1) * factor
        scaled_target_center_y = (target_center_y - crop_y1) * factor

        # Scale font size in layout_data for rendering
        scaled_layout_data = layout_data.copy()
        scaled_layout_data["lines"] = [
            line_data.copy() for line_data in layout_data["lines"]
        ]
        scaled_layout_data["font_size"] = layout_data["font_size"] * factor
        scaled_layout_data["line_height"] = layout_data["line_height"] * factor
        scaled_layout_data["max_line_width"] = layout_data["max_line_width"] * factor
        if layout_data.get("block_height") is not None:
            scaled_layout_data["block_height"] = layout_data["block_height"] * factor

        # Scale line widths and vertical glyph measurements
        for line_data in scaled_layout_data["lines"]:
            for key in (
                "width",
                "height",
                "advance_height",
                "baseline_offset_y",
                "left",
                "right",
                "top",
                "bottom",
                "origin_y",
            ):
                if key in line_data:
                    line_data[key] = line_data[key] * factor
            if "center_x" in line_data:
                line_data["center_x"] = (line_data["center_x"] - crop_x1) * factor
            if "target_width" in line_data:
                line_data["target_width"] = line_data["target_width"] * factor

        # Scale metrics - create a simple object with scaled attributes
        original_metrics = layout_data["metrics"]

        class ScaledMetrics:
            def __init__(self, original, scale_factor):
                self.fAscent = original.fAscent * scale_factor
                self.fDescent = original.fDescent * scale_factor
                # Preserve other attributes if they exist
                if hasattr(original, "fLeading"):
                    self.fLeading = original.fLeading * scale_factor
                if hasattr(original, "fXMin"):
                    self.fXMin = original.fXMin * scale_factor
                if hasattr(original, "fXMax"):
                    self.fXMax = original.fXMax * scale_factor
                if hasattr(original, "fYMin"):
                    self.fYMin = original.fYMin * scale_factor
                if hasattr(original, "fYMax"):
                    self.fYMax = original.fYMax * scale_factor

        scaled_metrics = ScaledMetrics(original_metrics, factor)
        scaled_layout_data["metrics"] = scaled_metrics

        # Create Skia surface from upscaled region
        try:
            scaled_surface = pil_to_skia_surface(upscaled_region)
        except RenderingError as e:
            raise RenderingError(f"Scaled surface preparation failed: {e}") from e

        # Render text at high resolution
        success = draw_layout(
            scaled_surface,
            scaled_layout_data,
            (
                0.0
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else scaled_target_center_x
            ),
            (
                0.0
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else scaled_target_center_y
            ),
            loaded_typefaces,
            loaded_hb_faces,
            regular_typeface,
            regular_hb_face,
            features_to_enable,
            text_color,
            config.use_subpixel_rendering,
            config.font_hinting,
            final_outline_width * factor,  # Scale outline width too
            verbose,
            pre_translate_x=(
                float(scaled_target_center_x)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_translate_y=(
                float(scaled_target_center_y)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_rotate_deg=(
                float(rotation_deg)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            text_background_color=skia_bg_color,
            outline_color=skia_outline_color,
        )

        if not success:
            log_message("Drawing failed", always_print=True)
            raise RenderingError("Text drawing failed")

        # Convert back to PIL and downscale
        try:
            scaled_pil_result = skia_surface_to_pil(scaled_surface)
        except RenderingError as e:
            raise RenderingError(f"Scaled conversion failed: {e}") from e

        # Downscale using LANCZOS for high quality
        downscaled_result = scaled_pil_result.resize(
            (crop_width, crop_height), Image.Resampling.LANCZOS
        )

        # Paste the result back onto the original image
        final_pil_image = pil_image.copy()
        final_pil_image.paste(downscaled_result, (crop_x1, crop_y1))

        log_message(
            f"Rendered at size {layout_data['font_size']} with {factor}x supersampling",
            verbose=verbose,
        )
        final_pil_image.info["font_size"] = layout_data["font_size"]
        return final_pil_image
    else:
        # Normal rendering path (no supersampling)
        try:
            surface = pil_to_skia_surface(pil_image)
        except RenderingError as e:
            raise RenderingError(f"Surface preparation failed: {e}") from e

        # Delegate rotation/translate to drawing_engine so Skia state is consistent
        success = draw_layout(
            surface,
            layout_data,
            0.0 if (rotation_deg and abs(rotation_deg) > 0.01) else target_center_x,
            0.0 if (rotation_deg and abs(rotation_deg) > 0.01) else target_center_y,
            loaded_typefaces,
            loaded_hb_faces,
            regular_typeface,
            regular_hb_face,
            features_to_enable,
            text_color,
            config.use_subpixel_rendering,
            config.font_hinting,
            final_outline_width,
            verbose,
            pre_translate_x=(
                float(target_center_x)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_translate_y=(
                float(target_center_y)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_rotate_deg=(
                float(rotation_deg)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            text_background_color=skia_bg_color,
            outline_color=skia_outline_color,
        )

        if not success:
            log_message("Drawing failed", always_print=True)
            raise RenderingError("Text drawing failed")

        try:
            final_pil_image = skia_surface_to_pil(surface)
        except RenderingError as e:
            raise RenderingError(f"Final conversion failed: {e}") from e

        log_message(f"Rendered at size {layout_data['font_size']}", verbose=verbose)
        final_pil_image.info["font_size"] = layout_data["font_size"]
        return final_pil_image
