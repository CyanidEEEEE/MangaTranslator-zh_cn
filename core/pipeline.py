import json
import re
import asyncio
import base64
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.caching import get_cache
from core.config import MangaTranslatorConfig, PreprocessingConfig, RenderingConfig
from core.scaling import scale_font_size, scale_length, scale_scalar
from utils.exceptions import (
    CancellationError,
    CleaningError,
    FontError,
    ImageProcessingError,
    RenderingError,
    TranslationError,
)
from utils.logging import log_message

from .image.cleaning import clean_speech_bubbles, retry_cleaning_with_otsu
from .image.detection import detect_panels, detect_speech_bubbles
from .image.image_utils import (
    convert_image_to_target_mode,
    cv2_to_pil,
    pil_to_cv2,
    resize_to_max_side,
    save_image_with_compression,
    upscale_image,
    upscale_image_to_dimension,
)
from .image.sorting import sort_bubbles_by_reading_order, sort_panels_by_reading_order
from .ml.model_manager import get_model_manager
from .outside_text_processor import process_outside_text
from .services.translation import (
    call_translation_api_batch,
    prepare_bubble_images_for_translation,
)
from .text.placeholders import generate_test_placeholders
from .text.text_processing import is_latin_style_language
from .text.text_renderer import render_text_skia

if TYPE_CHECKING:
    from ui.cancellation import CancellationManager

ENABLE_COMPONENT_ORDER_DEBUG = False


def _debug_mask_bbox(mask):
    """Return full-image bbox for a debug mask, or None when empty/invalid."""
    normalized = (
        _normalize_debug_mask(mask, (mask.shape[1], mask.shape[0]))
        if isinstance(mask, np.ndarray) and mask.ndim >= 2
        else None
    )
    if normalized is None:
        try:
            mask_array = np.asarray(mask)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            if mask_array.ndim != 2:
                return None
            normalized = mask_array > 0
        except Exception:
            return None
    coords = np.where(normalized)
    if coords[0].size == 0 or coords[1].size == 0:
        return None
    return [
        int(coords[1].min()),
        int(coords[0].min()),
        int(coords[1].max()) + 1,
        int(coords[0].max()) + 1,
    ]


def get_image_encoding_params(pil_image_format: Optional[str]) -> Tuple[str, str]:
    """Returns (mime_type, cv2_ext) for a given PIL image format."""
    if pil_image_format and pil_image_format.upper() == "PNG":
        return "image/png", ".png"
    return "image/jpeg", ".jpg"


def _load_debug_font(size: int):
    """Load a bold-ish font for the debug overlay, falling back safely."""
    font_candidates = [
        "arialbd.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_dashed_rectangle(draw, bbox, color, width=2, dash=12, gap=7):
    """Draw a dashed rectangle matching the requested debug style."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return

    def _draw_dashed_line(start, end, horizontal=True):
        if horizontal:
            fixed = start[1]
            pos = start[0]
            limit = end[0]
            while pos < limit:
                seg_end = min(pos + dash, limit)
                draw.line((pos, fixed, seg_end, fixed), fill=color, width=width)
                pos += dash + gap
        else:
            fixed = start[0]
            pos = start[1]
            limit = end[1]
            while pos < limit:
                seg_end = min(pos + dash, limit)
                draw.line((fixed, pos, fixed, seg_end), fill=color, width=width)
                pos += dash + gap

    _draw_dashed_line((x0, y0), (x1, y0), horizontal=True)
    _draw_dashed_line((x0, y1), (x1, y1), horizontal=True)
    _draw_dashed_line((x0, y0), (x0, y1), horizontal=False)
    _draw_dashed_line((x1, y0), (x1, y1), horizontal=False)


def _draw_centered_index(draw, bbox, value, font, color):
    """Draw the index at the visual center of the box."""
    x0, y0, x1, y1 = bbox
    cx = int(round((x0 + x1) / 2))
    cy = int(round((y0 + y1) / 2))
    label = str(value)
    try:
        draw.text((cx, cy), label, fill=color, font=font, anchor="mm")
    except TypeError:
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (cx - (right - left) / 2, cy - (bottom - top) / 2),
            label,
            fill=color,
            font=font,
        )


def _normalize_debug_mask(mask, image_size):
    """Normalize a debug mask into a full-image boolean array."""
    if mask is None:
        return None

    try:
        mask_array = np.asarray(mask)
    except Exception:
        return None

    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    if mask_array.ndim != 2:
        return None

    width, height = image_size
    if mask_array.shape != (height, width):
        return None

    return mask_array > 0


def _apply_mask_debug_overlay(canvas, mask, color=(255, 0, 0, 84)):
    """Alpha-composite a semi-transparent mask overlay onto the debug canvas."""
    normalized_mask = _normalize_debug_mask(mask, canvas.size)
    if normalized_mask is None or not np.any(normalized_mask):
        return

    overlay = np.zeros((canvas.size[1], canvas.size[0], 4), dtype=np.uint8)
    overlay[normalized_mask] = color
    canvas.alpha_composite(Image.fromarray(overlay, mode="RGBA"))


def _write_component_order_debug_image(
    image_size,
    sorted_items,
    panels,
    bubble_masks,
    reading_direction,
    image_path,
    output_path,
    verbose=False,
):
    """Write a debug PNG showing panel order and merged text-element order."""
    width, height = image_size
    if width <= 0 or height <= 0:
        return

    canvas = Image.new("RGBA", (width, height), (238, 238, 238, 255))
    draw = ImageDraw.Draw(canvas)

    panel_color = (32, 63, 255)
    osb_color = (255, 0, 255)
    bubble_color = (34, 160, 34)
    index_color = (255, 0, 0)

    font_size = max(14, min(width, height) // 28)
    font = _load_debug_font(font_size)

    panel_order = (
        sort_panels_by_reading_order(panels, reading_direction) if panels else []
    )

    for item in sorted_items:
        if item.get("is_outside_text", False):
            continue
        bbox = tuple(int(round(v)) for v in item.get("bbox", (0, 0, 0, 0)))
        _apply_mask_debug_overlay(
            canvas, bubble_masks.get(bbox) if bubble_masks else None
        )

    for panel_index, panel_id in enumerate(panel_order, start=1):
        panel_bbox = tuple(int(round(v)) for v in panels[panel_id])
        draw.rectangle(panel_bbox, outline=panel_color, width=3)
        _draw_centered_index(draw, panel_bbox, panel_index, font, index_color)

    for item_index, item in enumerate(sorted_items, start=1):
        bbox = tuple(int(round(v)) for v in item.get("bbox", (0, 0, 0, 0)))
        if item.get("is_outside_text", False):
            draw.rectangle(bbox, outline=osb_color, width=2)
            draw_bbox = bbox
        else:
            mask_bbox = (
                _debug_mask_bbox(bubble_masks.get(bbox)) if bubble_masks else None
            )
            draw_bbox = tuple(mask_bbox) if mask_bbox is not None else bbox
            _draw_dashed_rectangle(draw, draw_bbox, bubble_color, width=2)
        _draw_centered_index(draw, draw_bbox, item_index, font, index_color)

    base_path = Path(output_path) if output_path else Path(image_path)
    debug_path = base_path.parent / f"{base_path.stem}.component-order-debug.png"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(debug_path, format="PNG")
    log_message(
        f"Wrote component-order debug image: {debug_path}",
        verbose=verbose,
        always_print=True,
    )


def _write_llm_crop_debug_images(
    sorted_items,
    image_path,
    output_path,
    verbose=False,
):
    """Save the exact image crops the LLM sees to a debug subfolder."""
    base_path = Path(output_path) if output_path else Path(image_path)
    crop_dir = base_path.parent / f"{base_path.stem}.llm-crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, item in enumerate(sorted_items, start=1):
        img_b64 = item.get("image_b64")
        if not img_b64:
            continue
        try:
            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_cv = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_cv is None:
                continue
            label = "osb" if item.get("is_outside_text", False) else "bubble"
            crop_path = crop_dir / f"{i:03d}_{label}.png"
            cv2.imwrite(str(crop_path), img_cv)
            count += 1
        except Exception:
            pass

    log_message(
        f"Wrote {count} LLM crop debug images to: {crop_dir}",
        verbose=verbose,
        always_print=True,
    )


def _resolve_pre_upscale_factor(
    pre_cfg: Optional[PreprocessingConfig],
    verbose: bool = False,
) -> float:
    if pre_cfg is None or not pre_cfg.enabled:
        return 1.0

    factor = max(1.0, min(float(pre_cfg.factor or 1.0), 8.0))
    if factor <= 1.01:
        return 1.0

    log_message(f"Initial upscaling enabled: {factor:.2f}x", verbose=verbose)
    return factor


def _apply_pre_upscale_if_needed(
    image: Image.Image,
    config: MangaTranslatorConfig,
    verbose: bool = False,
) -> Tuple[Image.Image, float]:
    factor = _resolve_pre_upscale_factor(
        getattr(config, "preprocessing", None), verbose
    )
    if factor == 1.0:
        return image, 1.0

    # Use the output upscale model setting for initial upscaling as well
    model_type = (
        getattr(config.output, "image_upscale_model", "model_lite")
        if hasattr(config, "output")
        else "model_lite"
    )
    upscaled = upscale_image(image, factor, model_type=model_type, verbose=verbose)
    return upscaled, factor


def translate_and_render(
    image_path: Union[str, Path],
    config: MangaTranslatorConfig,
    output_path: Optional[Union[str, Path]] = None,
    cancellation_manager: Optional["CancellationManager"] = None,
):
    start_time = time.time()
    image_path = Path(image_path)
    verbose = config.verbose
    device = config.device

    log_message(f"Using device: {device}", verbose=verbose)

    hf_token = config.outside_text.huggingface_token
    get_model_manager().set_hf_token(hf_token)

    try:
        pil_original = Image.open(image_path)
        image_format = pil_original.format
        mime_type, cv2_ext = get_image_encoding_params(image_format)
    except FileNotFoundError:
        log_message(f"Error: Input image not found at {image_path}", always_print=True)
        raise
    except Exception as e:
        log_message(f"Error opening image {image_path}: {e}", always_print=True)
        raise

    if cancellation_manager and cancellation_manager.is_cancelled():
        raise TranslationError("Process cancelled by user.")

    desired_format = config.output.output_format
    output_ext_for_mode = (
        Path(output_path).suffix.lower() if output_path else image_path.suffix.lower()
    )

    if desired_format == "jpeg" or (
        desired_format == "auto" and output_ext_for_mode in [".jpg", ".jpeg"]
    ):
        target_mode = "RGB"
    else:
        target_mode = "RGBA"

    pil_image_processed = convert_image_to_target_mode(
        pil_original, target_mode, verbose
    )
    pil_image_processed, _ = _apply_pre_upscale_if_needed(
        pil_image_processed, config, verbose
    )

    if config.upscaling_only:
        log_message(
            "Upscaling only mode - skipping detection and translation",
            always_print=True,
        )
        final_image_to_save = pil_image_processed
        if config.output.upscale_final_image:
            final_image_to_save = upscale_image(
                final_image_to_save,
                config.output.image_upscale_factor,
                model_type=config.output.image_upscale_model,
                verbose=verbose,
            )
        if output_path:
            if final_image_to_save.mode != target_mode:
                final_image_to_save = final_image_to_save.convert(target_mode)
            try:
                save_image_with_compression(
                    final_image_to_save,
                    output_path,
                    jpeg_quality=config.output.jpeg_quality,
                    png_compression=config.output.png_compression,
                    verbose=verbose,
                )
            except ImageProcessingError as e:
                log_message(f"Failed to save image: {e}", always_print=True)
                raise
        return final_image_to_save

    if config.preprocessing.auto_scale:
        width, height = pil_image_processed.size
        processing_scale = math.sqrt((width * height) / 1_000_000)
    else:
        processing_scale = 1.0

    get_cache().set_current_image(pil_image_processed, verbose)
    original_cv_image = pil_to_cv2(pil_image_processed)

    try:
        bubble_data, text_free_boxes = detect_speech_bubbles(
            image_path,
            config.yolo_model_path,
            config.detection.confidence,
            verbose=verbose,
            device=device,
            seg_model=config.detection.seg_model,
            conjoined_detection=config.detection.conjoined_detection,
            conjoined_confidence=config.detection.conjoined_confidence,
            image_override=pil_image_processed,
            osb_enabled=config.outside_text.enabled,
            osb_text_verification=config.detection.use_osb_text_verification,
            osb_text_hf_token=config.outside_text.huggingface_token,
            bubble_detector_model=config.detection.bubble_detector_model,
        )
    except Exception as e:
        log_message(f"Error during detection: {e}", always_print=True)
        bubble_data = []
        text_free_boxes = []

    panels = None
    debug_panels = None
    if config.detection.use_panel_sorting or ENABLE_COMPONENT_ORDER_DEBUG:
        try:
            debug_panels = detect_panels(
                image_path,
                confidence=config.detection.panel_confidence,
                device=device,
                verbose=verbose,
            )
        except Exception as e:
            log_message(
                f"Panel detection failed: {e}. Using global sorting.",
                always_print=True,
            )
            debug_panels = None
        if config.detection.use_panel_sorting:
            panels = debug_panels

    pil_image_processed, outside_text_data = process_outside_text(
        pil_image_processed,
        config,
        image_path,
        image_format,
        verbose,
        bubble_data=bubble_data,
        text_free_boxes=text_free_boxes,
        panels=panels,
    )
    original_cv_image = pil_to_cv2(pil_image_processed)

    full_image_b64 = None
    full_image_mime_type = None
    if config.translation.send_full_page_context:
        try:
            context_image_pil = cv2_to_pil(original_cv_image)
            effective_context_max_side = scale_length(
                config.translation.context_image_max_side_pixels,
                None,
                minimum=512,
                maximum=4096,
            )

            context_upscale_method = (
                "none" if config.test_mode else config.translation.upscale_method
            )

            if context_upscale_method in ("model", "model_lite"):
                model_manager = get_model_manager()
                if context_upscale_method == "model":
                    upscale_model = model_manager.load_upscale(verbose=verbose)
                else:
                    upscale_model = model_manager.load_upscale_lite(verbose=verbose)
                context_image_pil = upscale_image_to_dimension(
                    upscale_model,
                    context_image_pil,
                    effective_context_max_side,
                    config.device,
                    "max",
                    context_upscale_method,
                    verbose,
                )
                context_image_pil = resize_to_max_side(
                    context_image_pil,
                    effective_context_max_side,
                    verbose=verbose,
                )
                model_manager.clear_cache()
            elif context_upscale_method == "lanczos":
                context_image_pil = resize_to_max_side(
                    context_image_pil,
                    effective_context_max_side,
                    verbose=verbose,
                )

            context_image_cv = pil_to_cv2(context_image_pil)
            is_success, buffer = cv2.imencode(cv2_ext, context_image_cv)
            if not is_success:
                raise ImageProcessingError(f"Full image encoding to {cv2_ext} failed")
            full_image_b64 = base64.b64encode(buffer).decode("utf-8")
            full_image_mime_type = mime_type
        except Exception as e:
            log_message(
                f"Warning: Failed to encode full image context: {e}", always_print=True
            )

    if cancellation_manager and cancellation_manager.is_cancelled():
        raise CancellationError("Process cancelled by user.")

    final_image_to_save = pil_image_processed

    if not bubble_data and not outside_text_data:
        log_message("No speech bubbles or outside text detected", always_print=True)
    else:
        if bubble_data:
            try:
                use_otsu = config.cleaning.use_otsu_threshold
                cleaned_image_cv, processed_bubbles_info = clean_speech_bubbles(
                    pil_image_processed,
                    config.yolo_model_path,
                    config.detection.confidence,
                    pre_computed_detections=bubble_data,
                    device=device,
                    thresholding_value=config.cleaning.thresholding_value,
                    use_otsu_threshold=use_otsu,
                    roi_shrink_px=config.cleaning.roi_shrink_px,
                    verbose=verbose,
                    processing_scale=processing_scale,
                    conjoined_confidence=config.detection.conjoined_confidence,
                    inpaint_colored_bubbles=config.cleaning.inpaint_colored_bubbles,
                    flux_hf_token=config.outside_text.huggingface_token,
                    flux_num_inference_steps=config.outside_text.flux_num_inference_steps,
                    flux_residual_diff_threshold=config.outside_text.flux_residual_diff_threshold,
                    flux_seed=config.outside_text.seed,
                    osb_text_verification=config.detection.use_osb_text_verification,
                    osb_text_hf_token=config.outside_text.huggingface_token,
                    inpaint_method=config.outside_text.inpainting_method,
                    kontext_backend=config.outside_text.kontext_backend,
                    flux_low_vram=config.outside_text.flux_low_vram,
                    flux_luminance_correction=config.outside_text.flux_luminance_correction,
                    bubble_detector_model=config.detection.bubble_detector_model,
                )
            except Exception as e:
                log_message(f"Cleaning failed: {e}", always_print=True)
                cleaned_image_cv = original_cv_image.copy()
                processed_bubbles_info = []

            pil_cleaned_image = cv2_to_pil(cleaned_image_cv)
            if pil_cleaned_image.mode != target_mode:
                pil_cleaned_image = pil_cleaned_image.convert(target_mode)
            final_image_to_save = pil_cleaned_image
        else:
            processed_bubbles_info = []
            pil_cleaned_image = pil_image_processed
            if pil_cleaned_image.mode != target_mode:
                pil_cleaned_image = pil_cleaned_image.convert(target_mode)
            final_image_to_save = pil_cleaned_image

        if config.cleaning_only:
            log_message("Cleaning only mode - skipping translation", always_print=True)
        else:
            main_min_font = scale_font_size(
                config.rendering.min_font_size, processing_scale, minimum=4, maximum=256
            )
            main_max_font = scale_font_size(
                config.rendering.max_font_size,
                processing_scale,
                minimum=main_min_font,
                maximum=384,
            )
            padding_pixels = scale_scalar(
                config.rendering.padding_pixels,
                processing_scale,
                minimum=1.0,
                maximum=80.0,
            )
            osb_min_font = scale_font_size(
                config.outside_text.osb_min_font_size,
                processing_scale,
                minimum=4,
                maximum=512,
            )
            osb_max_font = scale_font_size(
                config.outside_text.osb_max_font_size,
                processing_scale,
                minimum=osb_min_font,
                maximum=640,
            )
            osb_outline_width = scale_scalar(
                config.outside_text.osb_outline_width,
                processing_scale,
                minimum=0.0,
                maximum=24.0,
            )

            if processed_bubbles_info:
                _mask_lut: Dict[tuple, Any] = {}
                for _info in processed_bubbles_info:
                    _bk = tuple(int(round(v)) for v in _info.get("bbox", ()))
                    if len(_bk) != 4:
                        continue
                    _m = _info.get("mask")
                    if _m is None:
                        _m = _info.get("base_mask")
                    if _m is not None:
                        _mask_lut[_bk] = _m
                for _b in bubble_data:
                    _bk = tuple(int(round(v)) for v in _b.get("bbox", ()))
                    if _bk in _mask_lut:
                        _b["sam_mask"] = _mask_lut[_bk]

            bubble_upscale_method = (
                "none" if config.test_mode else config.translation.upscale_method
            )

            model_manager = get_model_manager()
            upscale_model = None
            if bubble_upscale_method == "model":
                upscale_model = model_manager.load_upscale(verbose=verbose)
            elif bubble_upscale_method == "model_lite":
                upscale_model = model_manager.load_upscale_lite(verbose=verbose)

            bubble_data = prepare_bubble_images_for_translation(
                bubble_data,
                original_cv_image,
                upscale_model,
                config.device,
                mime_type,
                config.translation.bubble_min_side_pixels,
                bubble_upscale_method,
                config.translation.whiteout_conjoined_bubbles,
                verbose,
            )
            if upscale_model is not None:
                model_manager.clear_cache()

            valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
            
            reading_direction = config.translation.reading_direction
            if outside_text_data:
                all_text_data = valid_bubble_data + outside_text_data
            else:
                all_text_data = valid_bubble_data

            sorted_bubble_data = sort_bubbles_by_reading_order(
                all_text_data, reading_direction, panels=panels
            )

            bubble_images_b64 = [
                bubble["image_b64"]
                for bubble in sorted_bubble_data
                if "image_b64" in bubble
            ]
            bubble_mime_types = [
                bubble["mime_type"]
                for bubble in sorted_bubble_data
                if "image_b64" in bubble and "mime_type" in bubble
            ]
            translated_texts = []
            _provider_tag = f"[{config.translation.provider}:"
            
            if bubble_images_b64:
                try:
                    translated_texts = call_translation_api_batch(
                        config=config.translation,
                        images_b64=bubble_images_b64,
                        full_image_b64=full_image_b64 or "",
                        mime_types=bubble_mime_types,
                        full_image_mime_type=full_image_mime_type
                        or "image/jpeg",
                        bubble_metadata=sorted_bubble_data,
                        debug=verbose,
                    )
                except Exception as e:
                    translated_texts = ["[Translation Error]"] * len(sorted_bubble_data)

            bubble_render_info_map = {
                tuple(info["bbox"]): {
                    "color": info["color"],
                    "mask": info.get("mask"),
                    "base_mask": info.get("base_mask"),
                    "is_sam": info.get("is_sam", False),
                    "is_colored": info.get("is_colored", False),
                    "text_bbox": info.get("text_bbox"),
                    "text_color_bgr": info.get("text_color_bgr"),
                }
                for info in processed_bubbles_info
                if "bbox" in info and "color" in info and "mask" in info
            }

            if len(translated_texts) == len(sorted_bubble_data):
                for i, bubble in enumerate(sorted_bubble_data):
                    bubble["translation"] = translated_texts[i]
                    bbox = bubble["bbox"]
                    text = bubble.get("translation", "")
                    is_outside_text = bubble.get("is_outside_text", False)

                    if is_outside_text and text:
                        text = text.upper()
                        bubble["translation"] = text

                    if not text or text.startswith("[Translation Error"):
                        continue

                    if is_outside_text:
                        font_dir = (
                            config.outside_text.osb_font_dir
                            if config.outside_text.osb_font_dir
                            else config.rendering.font_dir
                        )
                        min_font = osb_min_font
                        max_font = osb_max_font
                        line_spacing = config.outside_text.osb_line_spacing
                        use_ligs = config.outside_text.osb_use_ligatures
                        cleaned_mask = None
                        is_dark_text = bubble.get("is_dark_text", True)
                        text_color_rgb = bubble.get("text_color_rgb", None)
                        bubble_color_bgr = (
                            (50, 50, 50) if is_dark_text else (255, 255, 255)
                        )
                        rotation_deg = 0.0
                        vertical_stack = False
                        outline_w = osb_outline_width
                        text_bg_rgb = None
                    else:
                        font_dir = config.rendering.font_dir
                        min_font = main_min_font
                        max_font = main_max_font
                        line_spacing = config.rendering.line_spacing_mult
                        use_ligs = config.rendering.use_ligatures
                        outline_w = 0.0
                        render_info = bubble_render_info_map.get(tuple(bbox))
                        bubble_color_bgr = (255, 255, 255)
                        cleaned_mask = None
                        text_color_rgb = None
                        text_bg_rgb = None
                        if render_info:
                            bubble_color_bgr = render_info["color"]
                            cleaned_mask = render_info.get("mask")
                            text_color_bgr_val = render_info.get("text_color_bgr")
                            if text_color_bgr_val:
                                text_color_rgb = (
                                    text_color_bgr_val[2],
                                    text_color_bgr_val[1],
                                    text_color_bgr_val[0],
                                )
                        vertical_stack = False
                        rotation_deg = 0.0

                    render_config = RenderingConfig(
                        min_font_size=min_font,
                        max_font_size=max_font,
                        line_spacing_mult=line_spacing,
                        use_ligatures=use_ligs,
                        outline_width=outline_w,
                        padding_pixels=padding_pixels,
                    )
                    
                    try:
                        rendered_image = render_text_skia(
                            pil_image=pil_cleaned_image,
                            text=text,
                            bbox=bbox,
                            font_dir=font_dir,
                            cleaned_mask=cleaned_mask,
                            bubble_color_bgr=bubble_color_bgr,
                            config=render_config,
                            verbose=verbose,
                            bubble_id=str(i + 1),
                            rotation_deg=rotation_deg,
                            vertical_stack=vertical_stack,
                            text_color_rgb=text_color_rgb,
                            raise_on_safe_error=False,
                            text_background_color=text_bg_rgb,
                        )
                        pil_cleaned_image = rendered_image
                        final_image_to_save = pil_cleaned_image
                    except Exception as e:
                        log_message(f"Text rendering failed: {e}", verbose=verbose)

    if config.output.upscale_final_image:
        final_image_to_save = upscale_image(
            final_image_to_save,
            config.output.image_upscale_factor,
            model_type=config.output.image_upscale_model,
            verbose=verbose,
        )

    if output_path:
        if final_image_to_save.mode != target_mode:
            final_image_to_save = final_image_to_save.convert(target_mode)
        try:
            save_image_with_compression(
                final_image_to_save,
                output_path,
                jpeg_quality=config.output.jpeg_quality,
                png_compression=config.output.png_compression,
                verbose=verbose,
            )
        except ImageProcessingError as e:
            log_message(f"Failed to save image: {e}", always_print=True)
            raise

    return final_image_to_save


def _resolve_output_path(
    img_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: MangaTranslatorConfig,
    preserve_structure: bool,
) -> Tuple[Path, str, str]:
    if preserve_structure:
        relative_path = img_path.relative_to(input_dir)
        output_subdir = output_dir / relative_path.parent
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f"{relative_path.stem}_translated"
        display_path = str(relative_path)
        error_key = str(relative_path)
    else:
        output_subdir = output_dir
        output_filename = f"{img_path.stem}_translated"
        display_path = img_path.name
        error_key = img_path.name

    original_ext = img_path.suffix.lower()
    desired_format = config.output.output_format
    if desired_format == "jpeg":
        output_ext = ".jpg"
    elif desired_format == "png":
        output_ext = ".png"
    elif desired_format == "auto":
        output_ext = original_ext
    else:
        output_ext = original_ext

    return output_subdir / f"{output_filename}{output_ext}", display_path, error_key


async def _batch_translate_parallel(
    image_files: List[Path],
    input_dir: Path,
    config: MangaTranslatorConfig,
    output_dir: Path,
    preserve_structure: bool,
    progress_callback: Optional[Callable[[float, str], None]],
    cancellation_manager: Optional["CancellationManager"],
) -> Dict[str, Any]:
    total_images = len(image_files)
    n_workers = config.parallel_requests
    results = {"success_count": 0, "error_count": 0, "errors": {}}

    first_img = image_files[0]
    first_output, first_display, first_key = _resolve_output_path(
        first_img, input_dir, output_dir, config, preserve_structure
    )
    try:
        translate_and_render(
            first_img, config, first_output, cancellation_manager=cancellation_manager
        )
        results["success_count"] += 1
    except Exception as e:
        results["error_count"] += 1
        results["errors"][first_key] = str(e)

    completed_count = 1
    if progress_callback:
        progress_callback(1 / total_images, f"Completed 1/{total_images} images")

    remaining = image_files[1:]
    if not remaining:
        return results

    sem = asyncio.Semaphore(n_workers)
    results_lock = threading.Lock()
    cancelled = False

    def _process_single(img_path: Path, index: int) -> Tuple[str, str]:
        output_path, display_path, error_key = _resolve_output_path(
            img_path, input_dir, output_dir, config, preserve_structure
        )
        translate_and_render(
            img_path, config, output_path, cancellation_manager=cancellation_manager
        )
        return display_path, error_key

    async def _worker(img_path: Path, index: int, executor: ThreadPoolExecutor):
        nonlocal completed_count, cancelled
        async with sem:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(executor, _process_single, img_path, index)
                with results_lock:
                    results["success_count"] += 1
                    completed_count += 1
                    count = completed_count
            except Exception as e:
                _, display_path, error_key = _resolve_output_path(
                    img_path, input_dir, output_dir, config, preserve_structure
                )
                with results_lock:
                    results["error_count"] += 1
                    results["errors"][error_key] = str(e)
                    completed_count += 1
                    count = completed_count

            if progress_callback:
                progress_callback(count / total_images, f"Completed {count}/{total_images} images")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        tasks = [_worker(img, i, executor) for i, img in enumerate(remaining, start=1)]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

    return results


def batch_translate_images(
    input_dir: Union[str, Path],
    config: MangaTranslatorConfig,
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    preserve_structure: bool = False,
    cancellation_manager: Optional["CancellationManager"] = None,
) -> Dict[str, Any]:
    
    input_dir = Path(input_dir)
    if not output_dir:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./output") / timestamp
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    if preserve_structure:
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
    else:
        image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    image_files.sort(key=lambda p: p.name.lower())
    if not image_files:
        return {"success_count": 0, "error_count": 0, "errors": {}}

    if config.parallel_requests > 1:
        results = asyncio.run(
            _batch_translate_parallel(
                image_files=image_files, input_dir=input_dir, config=config, output_dir=output_dir,
                preserve_structure=preserve_structure, progress_callback=progress_callback, cancellation_manager=cancellation_manager,
            )
        )
    else:
        results = {"success_count": 0, "error_count": 0, "errors": {}}
        for i, img_path in enumerate(image_files):
            try:
                output_path, display_path, error_key = _resolve_output_path(img_path, input_dir, output_dir, config, preserve_structure)
                translate_and_render(img_path, config, output_path, cancellation_manager=cancellation_manager)
                results["success_count"] += 1
            except Exception as e:
                results["error_count"] += 1
                results["errors"][error_key] = str(e)
            if progress_callback:
                progress_callback((i + 1) / len(image_files), f"Completed {i + 1}/{len(image_files)}")

    return results

# =====================================================================================
# 全新的高级批量处理架构 (Two-Pass Batch Architecture) - 支持导出/API直连/导入与智能原生竖排
# =====================================================================================

def render_vertical_text_pil(pil_image, text, bbox, font_path, max_font, text_color_rgb, padding_pixels=0, line_spacing_mult=1.0, ss_factor=1, outline_width=0.0, outline_color=(255,255,255), text_bbox=None):
    """
    终极竖排渲染器：
    1. 真实 OpenCV 内接矩形 + 15% 黄金留白收缩
    2. 【修正】严格遵守用户指定的字体，绝不使用任何备用/替换字体！
    3. 纯几何锚点偏移，标点完美对齐。
    """
    from PIL import ImageDraw, ImageFont, Image
    import re
    import math
    
    SS = max(1, int(ss_factor))
    text = text.replace(" ", "").replace("\n", "").replace("\r", "")
    if not text: return pil_image

    VERTICAL_SUBSTITUTIONS = {
        '「': '﹁', '」': '﹂', '『': '﹃', '』': '﹄',
        '（': '︵', '）': '︶', '【': '︻', '】': '︼',
        '《': '︽', '》': '︾', '〈': '︿', '〉': '﹀',
        'ー': '丨', '-': '丨', '—': '丨', '~': '丨',
        '…': '︙', '。': '︒', '、': '︑'
    }
    for hz, vt in VERTICAL_SUBSTITUTIONS.items():
        text = text.replace(hz, vt)

    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
    
    if text_bbox and len(text_bbox) == 4:
        tx1, ty1, tx2, ty2 = [int(v) for v in text_bbox]
        safe_w = max(10, (tx2 - tx1) * 0.7 - padding_pixels * 2)
        safe_h = max(10, (ty2 - ty1) * 0.7 - padding_pixels * 2)
        center_x = tx1 + (tx2 - tx1) / 2
        center_y = ty1 + (ty2 - ty1) / 2
    else:
        safe_w = max(10, box_w * 0.55 - padding_pixels * 2)
        safe_h = max(10, box_h * 0.55 - padding_pixels * 2)
        center_x = x1 + box_w / 2
        center_y = y1 + box_h / 2
        
    target_w = max(safe_w, box_w * 0.25)
    target_h = max(safe_h, box_h * 0.25)

    BASE_FSIZE = 100
    line_spacing = BASE_FSIZE * 1.15 * line_spacing_mult 
    char_spacing = BASE_FSIZE * 1.05
    
    N = len(text)
    best_c = 1
    max_scale = 0
    best_rows = N
    
    for c in range(1, N + 1):
        rows = math.ceil(N / c)
        text_w_px = c * line_spacing
        text_h_px = rows * char_spacing
        
        scale = min(target_w / text_w_px, target_h / text_h_px)
        if scale > max_scale:
            max_scale = scale
            best_c = c
            best_rows = rows

    effective_fsize = BASE_FSIZE * max_scale
    if effective_fsize > max_font:
        max_scale = max_font / BASE_FSIZE

    canvas_w = int(best_c * line_spacing)
    canvas_h = int(best_rows * char_spacing)
    ss_canvas = Image.new('RGBA', (canvas_w * SS, canvas_h * SS), (0,0,0,0))
    draw = ImageDraw.Draw(ss_canvas)
    
    try:
        # 只加载用户指定的唯一字体
        font = ImageFont.truetype(str(font_path), BASE_FSIZE * SS)
    except:
        font = ImageFont.load_default()

    out_w = int((outline_width / max(1, effective_fsize)) * BASE_FSIZE * SS) if outline_width > 0 else 0

    cols = [text[i:i + best_rows] for i in range(0, N, best_rows)]
    
    for col_idx, col_text in enumerate(cols):
        cx = (canvas_w - col_idx * line_spacing - line_spacing / 2) * SS
        col_h = len(col_text) * char_spacing
        current_y = ((canvas_h - col_h) / 2) * SS
        
        for char in col_text:
            is_latin = bool(re.match(r'[a-zA-Z0-9_]', char))
            cy = current_y + (char_spacing / 2) * SS
            
            if is_latin:
                temp_size = int(BASE_FSIZE * SS * 2.5)
                temp_img = Image.new('RGBA', (temp_size, temp_size), (0,0,0,0))
                temp_draw = ImageDraw.Draw(temp_img)
                center_coord = temp_size // 2
                
                if out_w > 0:
                    for dx in [-out_w, 0, out_w]:
                        for dy in [-out_w, 0, out_w]:
                            if dx != 0 or dy != 0:
                                temp_draw.text((center_coord+dx, center_coord+dy), char, font=font, fill=outline_color, anchor="mm")
                temp_draw.text((center_coord, center_coord), char, font=font, fill=text_color_rgb, anchor="mm")
                
                rotated = temp_img.rotate(-90, resample=Image.Resampling.BICUBIC, expand=True)
                paste_x = int(cx - rotated.width / 2)
                paste_y = int(cy - rotated.height / 2)
                ss_canvas.paste(rotated, (paste_x, paste_y), rotated)
            else:
                offset_x, offset_y = 0, 0
                
                if char in ['「', '﹁', '『', '﹃']:
                    offset_x, offset_y = BASE_FSIZE * SS * 0.1, -BASE_FSIZE * SS * 0.1
                elif char in ['」', '﹂', '』', '﹄']:
                    offset_x, offset_y = -BASE_FSIZE * SS * 0.1, BASE_FSIZE * SS * 0.1
                elif char in ['！', '？', '!', '?', '‼️', '⁉️', '⁈', '❕']:
                    offset_x, offset_y = BASE_FSIZE * SS * 0.22, 0
                    
                if out_w > 0:
                    for dx in [-out_w, 0, out_w]:
                        for dy in [-out_w, 0, out_w]:
                            if dx != 0 or dy != 0:
                                draw.text((cx + offset_x + dx, cy + offset_y + dy), char, font=font, fill=outline_color, anchor="mm")
                draw.text((cx + offset_x, cy + offset_y), char, font=font, fill=text_color_rgb, anchor="mm")
            
            current_y += char_spacing * SS

    final_w = int(canvas_w * max_scale)
    final_h = int(canvas_h * max_scale)
    
    if final_w > 0 and final_h > 0:
        ss_canvas = ss_canvas.resize((final_w, final_h), Image.Resampling.LANCZOS)
        paste_x = int(center_x - final_w / 2)
        paste_y = int(center_y - final_h / 2)
        pil_image = pil_image.convert("RGBA")
        pil_image.alpha_composite(ss_canvas, (paste_x, paste_y))
        pil_image = pil_image.convert("RGB")

    return pil_image
        
def extract_batch_script(input_dir, config, output_dir):
    """阶段一：提取全本漫画文本，生成带标签 [OSB]/[Bubble] 的剧本"""
    import os
    import json
    import base64
    import re
    from io import BytesIO
    from pathlib import Path
    from PIL import Image
    from core.image.ocr_detection import extract_text_with_manga_ocr, extract_text_with_paddle_ocr_vl
    from core.image.detection import detect_speech_bubbles, detect_panels
    from core.outside_text_processor import process_outside_text
    from core.image.sorting import sort_bubbles_by_reading_order
    from core.services.translation import prepare_bubble_images_for_translation
    from core.image.image_utils import pil_to_cv2
    from core.ml.model_manager import get_model_manager
    from utils.logging import log_message
    
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions])
    
    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")

    global_script = {"chapter_info": {"total_pages": len(image_files)}, "pages": {}}
    log_message(f"Starting Two-Pass Extraction for {len(image_files)} images...", always_print=True)
    
    for idx, img_path in enumerate(image_files):
        log_message(f"Extracting {idx+1}/{len(image_files)}: {img_path.name}", always_print=True)
        pil_image = Image.open(img_path).convert("RGB")
        
        bubble_data, text_free_boxes = detect_speech_bubbles(
            img_path, config.yolo_model_path, config.detection.confidence,
            device=config.device, seg_model=config.detection.seg_model,
            osb_enabled=config.outside_text.enabled,
            bubble_detector_model=config.detection.bubble_detector_model
        )
        panels = detect_panels(img_path, config.detection.panel_confidence, config.device) if config.detection.use_panel_sorting else None
        
        pil_image, outside_text_data = process_outside_text(
            pil_image, config, img_path, pil_image.format, verbose=config.verbose,
            bubble_data=bubble_data, text_free_boxes=text_free_boxes, panels=panels
        )
        
        original_cv_image = pil_to_cv2(pil_image)
        bubble_upscale_method = config.translation.upscale_method
        model_manager = get_model_manager()
        upscale_model = None
        
        if bubble_upscale_method == "model":
            upscale_model = model_manager.load_upscale(verbose=config.verbose)
        elif bubble_upscale_method == "model_lite":
            upscale_model = model_manager.load_upscale_lite(verbose=config.verbose)

        prepared_bubbles = prepare_bubble_images_for_translation(
            bubble_data, original_cv_image, upscale_model, config.device,
            "image/jpeg", config.translation.bubble_min_side_pixels,
            bubble_upscale_method, config.translation.whiteout_conjoined_bubbles, config.verbose
        )

        if upscale_model is not None:
            model_manager.clear_cache()
            
        all_text_data = prepared_bubbles + outside_text_data
        if not all_text_data:
            continue
            
        sorted_items = sort_bubbles_by_reading_order(all_text_data, config.translation.reading_direction, panels=panels)
        images_b64 = [item["image_b64"] for item in sorted_items if item.get("image_b64")]
        extracted_texts = []
        
        if images_b64:
            if config.translation.ocr_method == "LLM":
                from core.services.translation import _call_llm_endpoint
                llm_parts = []
                for i, b64 in enumerate(images_b64):
                    if b64.startswith('data:'):
                        mime = b64.split(';')[0].split(':')[1]
                        data = b64.split(',')[1]
                    else:
                        mime = "image/jpeg"
                        data = b64
                    llm_parts.append({"text": f"Bubble {i+1}:"})
                    llm_parts.append({"inline_data": {"mime_type": mime, "data": data}})
                
                sys_prompt = f"You are an expert OCR assistant. Extract the {config.translation.input_language} text from the provided image crops. Reply EXACTLY in this format:\n1: [text]\n2: [text]"
                try:
                    log_message(f"Calling LLM ({config.translation.model_name}) for OCR extraction...", always_print=True)
                    llm_response = _call_llm_endpoint(config.translation, llm_parts, "Transcribe the text.", config.verbose, system_prompt=sys_prompt)
                    text_dict = {}
                    for line in llm_response.split('\n'):
                        match = re.match(r'^(\d+)[:：]\s*(.*)', line.strip())
                        if match:
                            text_dict[int(match.group(1))] = match.group(2)
                    for i in range(len(images_b64)):
                        extracted_texts.append(text_dict.get(i+1, ""))
                except Exception as e:
                    log_message(f"LLM OCR API failed: {e}. Outputting empty strings.", always_print=True)
                    extracted_texts = [""] * len(images_b64)
            else:
                pil_images_for_ocr = []
                for b64 in images_b64:
                    if b64.startswith('data:image'):
                        b64 = b64.split(',')[1]
                    pil_images_for_ocr.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
                    
                if config.translation.ocr_method == "paddleocr-vl":
                    extracted_texts = extract_text_with_paddle_ocr_vl(pil_images_for_ocr)
                else:
                    extracted_texts = extract_text_with_manga_ocr(pil_images_for_ocr)
        
        page_items = []
        for i, item in enumerate(sorted_items):
            text = extracted_texts[i] if i < len(extracted_texts) else ""
            if text == "[OCR FAILED]": text = ""
            text = " ".join(text.split())
            page_items.append({
                "id": i + 1,
                "bbox": item["bbox"],
                "is_outside_text": item.get("is_outside_text", False),
                "original_text": text,
                "translated_text": ""
            })
        global_script["pages"][img_path.name] = page_items
        
    with open(output_dir / "manga_script.json", "w", encoding="utf-8") as f:
        json.dump(global_script, f, ensure_ascii=False, indent=2)
        
    system_prompt = """## ROLE (角色设定)
你是一位经验丰富的“漫画汉化组”资深润色主笔。你现在拿到的是**一整话（全本）**经过 OCR 提取的漫画日文台词剧本。你的任务是利用你强大的长文本记忆能力，统揽全局，将其本地化为极具画面感、情感张力和前后连贯的中文对白。

## 思维与规划指令 (针对 <think> 过程)
在生成最终译文前，请在后台逻辑中按以下顺序思考：
1. 上下文串联（跨页记忆）：通读全话文本，理解剧情走向、人物代词（你/我/他/它）的指代关系。前一页出现的伏笔或专有名词，后一页必须保持绝对统一。
2. 标签语境感知：原文中区分了 `[Bubble]` 和 `[OSB]`。
   - `[Bubble]`（气泡框内）：属于角色的正式对白、大喊或核心内心独白，需注重情绪爆发与语流连贯。
   - `[OSB]`（画框外文字）：通常是角色的背景碎碎念、路人吐槽、动作音效补充或作者的旁白注解。翻译应更碎片化、口语化，或是作为主台词的辅助情绪。
3. 角色语气重构：分辨当前台词是男主、女主、路人还是旁白。根据语境赋予极具感染力的口吻（傲娇、粗犷、冷静、诱惑等），不回避任何符合该语境的露骨、激进或市井词汇。
4. 视觉字数压缩：漫画的排版空间极小。在绝对不改变原意的前提下，进行“极限精简重组”，剔除啰嗦的修饰语，确保字数最短。

## 核心解析与输出规范 (最高优先级)
你的输出将被自动化代码直接解析，**任何格式错误都会导致程序崩溃！** 必须严格遵守以下法则：
1. 结构冻结：必须**原封不动地保留**所有的 `=== PAGE: filename ===` 分页标记。绝对不允许遗漏任何一页！
2. 序号绝对对齐：必须严格以 `数字: 译文` 格式输出。**严禁**合并相邻的台词，**严禁**拆分台词，**严禁**删除空行或跳过序号！输入有多少个编号，输出就必须有多少个编号。
3. 标签剔除：在最终输出的译文中，**请直接输出纯中文，禁止输出** `[Bubble]` 或 `[OSB]` 标签。（例如，原文为 `10: [Bubble] お！端田君！`，请直接输出 `10: 哦！端田同学！`）。
4. 绝对单行防断层：每一个编号对应的译文，**必须写在同一物理行内**。译文内部绝对不允许出现换行符（回车键），否则解析器将无法读取。
5. OCR 盲区处理：若原文中出现 `[OCR FAILED]` 请不要试图脑补，直接在译文中原样输出 `[OCR FAILED]` 或留空，请注意，不要在没有[OCR FAILED]的地方莫名输出[OCR FAILED]。
6. 有一些不断重复的明显是模型识别出了问题，此时你应该根据上下文和不断重复的内容推测实际的合理内容。

## 汉化润色原则
1. 拒绝翻译腔：严禁呆板直译。将日式的倒装句、半截话、省略语转化为极其地道、火辣、有生命力的中文表述。
2. 严格的中文标点规范：**严禁**在中文里使用日式非标符号组合（如 `～！`、`～？`）。请将其转化为标准的中文表达。同时不要大量重复一个相同的符号表达，这样不符合漫画的对话框模式。
   - 错误示范：妈，我回来啦～！ / 你很怕吧～？
   - 正确示范：妈，我回来啦——！ / 你很怕吧？（依靠语气词或破折号来表现拖长音，而不是波浪号叠加感叹号）。
3. 样式触发（按需使用）：
   - 使用 *文字* 触发斜体（用于内心独白、回忆、微弱的喘息、旁白）。
   - 使用 **文字** 触发粗体（用于大喊、怒吼、情绪爆发的重音强调、拟声词）。

## 输出示例
1: **你给我快点！**
2: *我知道不用你提醒...*
3: **老夫认错人了，简直一模一样！**

========== 剧本正文 ==========
"""

    with open(output_dir / "manga_script_original.txt", "w", encoding="utf-8") as f:
        f.write(system_prompt + "\n")
        for page_name, items in global_script["pages"].items():
            f.write(f"=== PAGE: {page_name} ===\n")
            for item in items:
                tag = "[OSB]" if item.get("is_outside_text", False) else "[Bubble]"
                f.write(f"{item['id']}: {tag} {item['original_text']}\n")
            f.write("\n")
            
    log_message(f"Extraction complete! Script saved.", always_print=True)

def parse_translated_txt(txt_path, json_path):
    import re
    import json
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(json_path, "r", encoding="utf-8") as f:
        global_script = json.load(f)
        
    current_page = None
    page_pattern = re.compile(r'=== PAGE:\s*(.*?)\s*===')
    text_pattern = re.compile(r'^(\d+)[:：]\s*(.*)$')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        page_match = page_pattern.search(line)
        if page_match:
            current_page = page_match.group(1)
            continue
            
        if current_page and current_page in global_script["pages"]:
            text_match = text_pattern.match(line)
            if text_match:
                item_id = int(text_match.group(1))
                trans_text = text_match.group(2).strip()
                
                # 核心修复：智能切除 LLM 可能会返回的 [OSB] 或 [Bubble] 标签，防止污染嵌字
                trans_text = re.sub(r'^\[.*?\]\s*', '', trans_text).strip()
                
                for item in global_script["pages"][current_page]:
                    if item["id"] == item_id:
                        item["translated_text"] = trans_text
                        break

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(global_script, f, ensure_ascii=False, indent=2)


def render_fallback_text_pil(pil_image, text, bbox, font_path, font_size, text_color_rgb, outline_width=0.0, outline_color=(255,255,255), text_bbox=None):
    """终极保底渲染器：当 Skia 罢工时，强行居中在气泡真实内部！"""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(str(font_path), int(font_size))
    except:
        font = ImageFont.load_default()
        
    # 同理，优先拿真实的内接矩形中心去画
    if text_bbox and len(text_bbox) == 4:
        tx1, ty1, tx2, ty2 = [int(v) for v in text_bbox]
        cx, cy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
    out_w = int(outline_width)
    
    if out_w > 0:
        for dx in [-out_w, 0, out_w]:
            for dy in [-out_w, 0, out_w]:
                if dx != 0 or dy != 0:
                    draw.text((cx+dx, cy+dy), text, font=font, fill=outline_color, anchor="mm")
    draw.text((cx, cy), text, font=font, fill=text_color_rgb, anchor="mm")
    return pil_image

def auto_translate_script_api(txt_path, config, output_txt_path):
    from core.services.translation import _call_llm_endpoint
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    system_prompt = f"你是一位资深的漫画本地化翻译专家。你的目标是将整话漫画剧本翻译为{config.translation.output_language}。\n请严格保持用户的 `=== PAGE: xxx ===` 标记以及序号格式返回，不要增删任何行。"
    if config.translation.special_instructions:
        system_prompt += f"\n\n术语表与特殊要求：\n{config.translation.special_instructions}"
        
    log_message(f"Sending chapter to {config.translation.provider} API for context-aware translation...", always_print=True)
    response = _call_llm_endpoint(config.translation, parts=[], prompt_text=content, debug=config.verbose, system_prompt=system_prompt)
    
    if not response:
        raise TranslationError("Global API Translation returned empty response.")
        
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(response)

def render_batch_from_script(input_dir, json_path, config, output_dir):
    import os
    import json
    import math
    from pathlib import Path
    from PIL import Image
    from core.image.detection import detect_speech_bubbles, detect_panels
    from core.outside_text_processor import process_outside_text
    from core.image.sorting import sort_bubbles_by_reading_order
    from core.image.cleaning import clean_speech_bubbles
    from core.image.image_utils import cv2_to_pil, pil_to_cv2, save_image_with_compression
    from core.text.text_renderer import render_text_skia
    from core.config import RenderingConfig
    from core.scaling import scale_font_size, scale_scalar
    from core.text.font_manager import find_font_variants
    from utils.logging import log_message

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, "r", encoding="utf-8") as f:
        global_script = json.load(f)
        
    pages_data = global_script.get("pages", {})
    if not pages_data:
        log_message("No pages found in script JSON.", always_print=True)
        return
        
    for img_name, items in pages_data.items():
        img_path = input_dir / img_name
        if not img_path.exists():
            continue
            
        log_message(f"Rendering page: {img_name}...", always_print=True)
        pil_image = Image.open(img_path).convert("RGB")
        
        if config.preprocessing.auto_scale:
            width, height = pil_image.size
            processing_scale = math.sqrt((width * height) / 1_000_000)
        else:
            processing_scale = 1.0

        log_message(f"Re-generating component masks for {img_name}...", always_print=True)
        bubble_data, text_free_boxes = detect_speech_bubbles(
            img_path, config.yolo_model_path, config.detection.confidence,
            device=config.device, seg_model=config.detection.seg_model,
            osb_enabled=config.outside_text.enabled,
            bubble_detector_model=config.detection.bubble_detector_model
        )
        panels = detect_panels(img_path, config.detection.panel_confidence, config.device) if config.detection.use_panel_sorting else None
        
        pil_image, outside_text_data = process_outside_text(
            pil_image, config, img_path, pil_image.format, verbose=config.verbose,
            bubble_data=bubble_data, text_free_boxes=text_free_boxes, panels=panels
        )
        
        all_text_data = bubble_data + outside_text_data
        sorted_items = sort_bubbles_by_reading_order(all_text_data, config.translation.reading_direction, panels=panels)
        
        json_translations = {item["id"]: item["translated_text"] for item in items}
        for i, bubble in enumerate(sorted_items):
            bubble["translation"] = json_translations.get(i + 1, "")
            
        cleaned_image_cv, processed_bubbles_info = clean_speech_bubbles(
            pil_image, config.yolo_model_path, config.detection.confidence,
            pre_computed_detections=bubble_data,
            device=config.device,
            thresholding_value=config.cleaning.thresholding_value,
            use_otsu_threshold=config.cleaning.use_otsu_threshold,
            roi_shrink_px=config.cleaning.roi_shrink_px,
            verbose=config.verbose,
            processing_scale=processing_scale,
            conjoined_confidence=config.detection.conjoined_confidence,
            inpaint_colored_bubbles=config.cleaning.inpaint_colored_bubbles,
            flux_hf_token=config.outside_text.huggingface_token,
            flux_num_inference_steps=config.outside_text.flux_num_inference_steps,
            flux_residual_diff_threshold=config.outside_text.flux_residual_diff_threshold,
            flux_seed=config.outside_text.seed,
            osb_text_verification=config.detection.use_osb_text_verification,
            inpaint_method=config.outside_text.inpainting_method,
            kontext_backend=config.outside_text.kontext_backend,
            flux_low_vram=config.outside_text.flux_low_vram,
            flux_luminance_correction=config.outside_text.flux_luminance_correction,
            bubble_detector_model=config.detection.bubble_detector_model,
        )
        
        pil_cleaned_image = cv2_to_pil(cleaned_image_cv)
        bubble_render_info_map = {tuple(info["bbox"]): info for info in processed_bubbles_info if "bbox" in info}
        final_image = pil_cleaned_image
        
        main_min_font = scale_font_size(config.rendering.min_font_size, processing_scale, minimum=4, maximum=256)
        main_max_font = scale_font_size(config.rendering.max_font_size, processing_scale, minimum=main_min_font, maximum=384)
        padding_pixels = scale_scalar(config.rendering.padding_pixels, processing_scale, minimum=1.0, maximum=80.0)
        osb_min_font = scale_font_size(config.outside_text.osb_min_font_size, processing_scale, minimum=4, maximum=512)
        osb_max_font = scale_font_size(config.outside_text.osb_max_font_size, processing_scale, minimum=osb_min_font, maximum=640)
        osb_outline_width = scale_scalar(config.outside_text.osb_outline_width, processing_scale, minimum=0.0, maximum=24.0)

        for i, bubble in enumerate(sorted_items):
            text = bubble.get("translation", "")
            if not text: continue
            
            bbox = bubble["bbox"]
            is_osb = bubble.get("is_outside_text", False)
            box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            is_originally_vertical = box_h > box_w * 1.05
            
            if is_osb:
                font_dir = config.outside_text.osb_font_dir if config.outside_text.osb_font_dir else config.rendering.font_dir
                min_font, max_font = osb_min_font, osb_max_font
                line_spacing = config.outside_text.osb_line_spacing
                use_ligs = config.outside_text.osb_use_ligatures
                cleaned_mask = None
                text_bbox = None
                text_color_rgb = bubble.get("text_color_rgb", None)
                if not text_color_rgb:
                    is_dark_text = bubble.get("is_dark_text", True)
                    text_color_rgb = (0, 0, 0) if is_dark_text else (255, 255, 255)
                bubble_color_bgr = (50, 50, 50) if text_color_rgb == (255,255,255) else (255, 255, 255)
                outline_color = (255,255,255) if text_color_rgb == (0,0,0) else (0,0,0)
                outline_w = osb_outline_width
            else:
                font_dir = config.rendering.font_dir
                min_font, max_font = main_min_font, main_max_font
                line_spacing = config.rendering.line_spacing_mult
                use_ligs = config.rendering.use_ligatures
                outline_w = 0.0
                
                render_info = bubble_render_info_map.get(tuple(bbox), {})
                bubble_color_bgr = render_info.get("color", (255, 255, 255))
                cleaned_mask = render_info.get("mask")
                
                # 提取底层算出的最安全内接矩形！
                text_bbox = render_info.get("text_bbox")
                
                text_color_bgr_val = render_info.get("text_color_bgr")
                if text_color_bgr_val:
                    text_color_rgb = (text_color_bgr_val[2], text_color_bgr_val[1], text_color_bgr_val[0])
                else:
                    lum = 0.299 * bubble_color_bgr[2] + 0.587 * bubble_color_bgr[1] + 0.114 * bubble_color_bgr[0]
                    text_color_rgb = (0, 0, 0) if lum > 128 else (255, 255, 255)
                outline_color = (0,0,0)

            if is_originally_vertical and not is_osb:
                log_message(f"Using high-quality vertical engine for bubble {i+1}", verbose=config.verbose)
                try:
                    font_variants = find_font_variants(font_dir, verbose=config.verbose)
                    regular_font_path = font_variants.get("regular", font_variants.get("bold", None))
                    
                    if not regular_font_path:
                        for f in Path(font_dir).glob("*.ttf"):
                            regular_font_path = str(f)
                            break
                            
                    if regular_font_path:
                        final_image = render_vertical_text_pil(
                            final_image, text, bbox, regular_font_path, 
                            max_font=max_font, 
                            text_color_rgb=text_color_rgb, 
                            padding_pixels=padding_pixels,
                            line_spacing_mult=line_spacing,
                            ss_factor=config.rendering.supersampling_factor,
                            outline_width=outline_w, outline_color=outline_color,
                            text_bbox=text_bbox # 传入关键参数！
                        )
                    else:
                        log_message(f"No valid .ttf found in {font_dir}", always_print=True)
                except Exception as e:
                    log_message(f"Vertical rendering failed for bubble {i+1}: {e}", always_print=True)
                continue
                
            render_config = RenderingConfig(
                min_font_size=min_font,
                max_font_size=max_font,
                line_spacing_mult=line_spacing,
                use_ligatures=use_ligs,
                outline_width=outline_w,
                padding_pixels=padding_pixels,
            )
            
            try:
                final_image = render_text_skia(
                    pil_image=final_image,
                    text=text.upper() if is_osb else text,
                    bbox=bbox,
                    font_dir=font_dir,
                    cleaned_mask=cleaned_mask,
                    bubble_color_bgr=bubble_color_bgr,
                    config=render_config,
                    verbose=config.verbose,
                    bubble_id=str(i+1),
                    vertical_stack=False,
                    text_color_rgb=text_color_rgb,
                    raise_on_safe_error=True
                )
            except Exception as e:
                log_message(f"Skia failed for bubble {i+1} ({e}). Forcing fallback render!", always_print=True)
                try:
                    font_variants = find_font_variants(font_dir, verbose=config.verbose)
                    fallback_font_path = font_variants.get("regular", font_variants.get("bold", None))
                    if not fallback_font_path:
                        for f in Path(font_dir).glob("*.ttf"):
                            fallback_font_path = str(f)
                            break
                    if fallback_font_path:
                        final_image = render_fallback_text_pil(
                            final_image, text, bbox, fallback_font_path, 
                            min_font, text_color_rgb, outline_w, outline_color, text_bbox=text_bbox
                        )
                except Exception as fallback_e:
                    log_message(f"Fallback also failed: {fallback_e}", always_print=True)
                
        output_file = output_dir / f"{img_path.stem}_translated{img_path.suffix}"
        save_image_with_compression(final_image, str(output_file), jpeg_quality=config.output.jpeg_quality)
        
    log_message("All images rendered successfully from script!", always_print=True)
    
def run_advanced_batch_pipeline(input_dir, config, output_dir, mode="export_only"):
    from pathlib import Path
    json_path = Path(output_dir) / "manga_script.json"
    original_txt = Path(output_dir) / "manga_script_original.txt"
    translated_txt = Path(output_dir) / "manga_script_translated.txt"
    
    if mode in ["export_only", "auto_api_full"]:
        extract_batch_script(input_dir, config, output_dir)
        
    if mode == "auto_api_full":
        auto_translate_script_api(original_txt, config, translated_txt)
        parse_translated_txt(translated_txt, json_path)
        render_batch_from_script(input_dir, json_path, config, output_dir)
        
    elif mode == "import_render":
        if not translated_txt.exists():
            raise FileNotFoundError("Error: Could not find manga_script_translated.txt in the output folder.")
        if not json_path.exists():
            raise FileNotFoundError("Error: Could not find manga_script.json. Please make sure you uploaded it in the UI.")
            
        parse_translated_txt(translated_txt, json_path)
        render_batch_from_script(input_dir, json_path, config, output_dir)