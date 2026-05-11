import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from PIL import Image

from core.config import MangaTranslatorConfig, RenderingConfig
from core.pipeline import batch_translate_images, translate_and_render
from core.validation import validate_batch_input_path, validate_core_inputs
from utils.exceptions import (
    CancellationError,
    CleaningError,
    FontError,
    ImageProcessingError,
    RenderingError,
    TranslationError,
    ValidationError,
)
from utils.logging import log_message

if TYPE_CHECKING:
    from ui.cancellation import CancellationManager


class LogicError(Exception):
    pass


def extract_zip_to_temp(
    zip_file_path: Union[str, Path],
) -> tuple[Path, tempfile.TemporaryDirectory]:
    """
    Extracts a ZIP archive to a temporary directory, preserving folder structure.

    Args:
        zip_file_path: Path to the ZIP file to extract.

    Returns:
        Tuple of (Path to extracted directory, TemporaryDirectory object).
        The caller must keep a reference to the TemporaryDirectory object
        to prevent premature cleanup, and call cleanup() when done.

    Raises:
        ValidationError: If the ZIP file is invalid or cannot be extracted.
        FileNotFoundError: If the ZIP file does not exist.
    """
    zip_path = Path(zip_file_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file '{zip_file_path}' does not exist.")

    if not zip_path.is_file():
        raise ValidationError(f"Path '{zip_file_path}' is not a file.")

    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir_obj.name)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir_path)

        log_message(
            f"Extracted ZIP archive '{zip_path.name}' to temporary directory",
            verbose=True,
        )

        return temp_dir_path, temp_dir_obj

    except zipfile.BadZipFile:
        temp_dir_obj.cleanup()
        raise ValidationError(f"File '{zip_file_path}' is not a valid ZIP archive.")
    except Exception as e:
        temp_dir_obj.cleanup()
        raise ValidationError(f"Failed to extract ZIP archive: {str(e)}")


def translate_manga_logic(
    image: Union[str, Path, Image.Image],
    config: MangaTranslatorConfig,
    selected_font_pack_name: str,
    models_dir: Path,
    fonts_base_dir: Path,
    output_base_dir: Path = Path("./output"),
    cancellation_manager: Optional["CancellationManager"] = None,
) -> tuple[Image.Image, Path]:
    """
    Processes a single manga image. Handles core validation, calls the core pipeline,
    and returns the processed image or raises standard exceptions.

    Args:
        image: Input image (path or PIL object).
        config: The main configuration object.
        selected_font_pack_name: Name of the font pack directory.
        models_dir: Path to the directory containing YOLO models.
        fonts_base_dir: Path to the base directory containing font pack subdirectories.
        output_base_dir: Base directory to save the output image.
        cancellation_manager: Optional manager to handle cancellation.

    Returns:
        A tuple containing:
            - The translated PIL Image object.
            - The Path object where the image was saved.

    Raises:
        FileNotFoundError: If required models or fonts are not found.
        ValidationError: If core configuration validation fails.
        ValueError: For general configuration issues.
        LogicError: For processing failures within this function.
        Exception: For unexpected errors during processing.
    """
    display_name = (
        Path(image).name if isinstance(image, (str, Path)) else "uploaded image"
    )
    try:
        # Create temporary config for validation since we only have the font pack name
        rendering_cfg_for_val = RenderingConfig(
            font_dir=selected_font_pack_name,
            max_font_size=config.rendering.max_font_size,
            min_font_size=config.rendering.min_font_size,
            line_spacing_mult=config.rendering.line_spacing_mult,
            font_hinting=config.rendering.font_hinting,
        )
        yolo_model_path, font_dir_path = validate_core_inputs(
            translation_cfg=config.translation,
            rendering_cfg=rendering_cfg_for_val,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            bubble_detector_model=config.detection.bubble_detector_model,
        )
        config.yolo_model_path = str(yolo_model_path)
        config.rendering.font_dir = str(font_dir_path)

    except (FileNotFoundError, ValidationError, ValueError) as e:
        raise e

    temp_image_path = None
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if isinstance(image, (str, Path)):
            input_path = Path(image)
            original_ext = input_path.suffix.lower()
            image_path_for_processing = str(input_path)
        elif isinstance(image, Image.Image):
            # PIL objects need to be saved to disk for processing
            original_ext = ".png"
            temp_image_path = Path(tempfile.mktemp(suffix=".png"))
            image.save(temp_image_path)
            image_path_for_processing = str(temp_image_path)

        output_format = config.output.output_format
        if output_format == "auto":
            output_ext = (
                original_ext
                if original_ext in [".jpg", ".jpeg", ".png", ".webp"]
                else ".png"
            )
        elif output_format == "png":
            output_ext = ".png"
        elif output_format == "jpeg":
            output_ext = ".jpg"
        else:
            output_ext = ".png"

        os.makedirs(output_base_dir, exist_ok=True)
        if isinstance(image, (str, Path)):
            original_name = input_path.stem
        else:
            # PIL.Image object - no original filename available
            original_name = "MangaTranslator"
        save_path = (
            output_base_dir / f"{original_name}_translated_{timestamp}{output_ext}"
        )

        translated_image = translate_and_render(
            image_path=image_path_for_processing,
            config=config,
            output_path=save_path,
            cancellation_manager=cancellation_manager,
        )

        if not translated_image:
            raise LogicError("Translation process failed to return an image.")

        return translated_image, save_path

    except (FontError, RenderingError) as e:
        raise LogicError(f"Text rendering failed: {str(e)}") from e
    except CleaningError as e:
        raise LogicError(f"Bubble cleaning failed: {str(e)}") from e
    except TranslationError as e:
        raise LogicError(f"Translation failed: {str(e)}") from e
    except ImageProcessingError as e:
        raise LogicError(f"Image processing failed: {str(e)}") from e
    except CancellationError as e:
        log_message(f"Translation cancelled for {display_name}", verbose=config.verbose)
        raise e
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise LogicError(
            f"An unexpected error occurred during translation: {str(e)}"
        ) from e
    finally:
        if temp_image_path and temp_image_path.exists():
            try:
                os.remove(temp_image_path)
            except Exception as e_clean:
                log_message(
                    f"Failed to clean up temporary file: {e_clean}", always_print=True
                )


def process_batch_logic(
    input_dir_or_files: Union[str, List[str], Dict[str, Any]],
    config: MangaTranslatorConfig,
    selected_font_pack_name: str,
    models_dir: Path,
    fonts_base_dir: Path,
    output_base_dir: Path = Path("./output"),
    gradio_progress: Any = None,
    cancellation_manager: Optional["CancellationManager"] = None,
    batch_workflow_mode: str = "标准模式 (逐页处理)",
    batch_large_directory_mode: bool = False,
    batch_script_upload: Optional[str] = None,
    batch_json_upload: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Processes a batch of manga images. Handles core validation, calls the core batch pipeline,
    and returns results or raises standard exceptions.

    Args:
        input_dir_or_files: Can be one of:
            - Path to input directory (str)
            - Path to ZIP archive (str ending in .zip)
            - List of file paths (List[str])
            - Dictionary with both "zip" and "files" keys to process both simultaneously (Dict[str, Any])
        config: The main configuration object.
        selected_font_pack_name: Name of the font pack directory.
        models_dir: Path to the directory containing YOLO models.
        fonts_base_dir: Path to the base directory containing font pack subdirectories.
        output_base_dir: Base directory to save the output images.
        gradio_progress: Optional Gradio Progress object for UI updates.

    Returns:
        A dictionary containing processing results:
        {
            "success_count": int,
            "error_count": int,
            "errors": Dict[str, str],
            "output_path": Path,
            "processing_time": float
        }

    Raises:
        FileNotFoundError: If required models or fonts are not found, or input dir is invalid.
        ValidationError: If core configuration validation fails.
        ValueError: For general configuration issues or invalid inputs.
        LogicError: For processing failures within this function.
        Exception: For unexpected errors during processing.
    """
    start_time = time.time()

    if batch_workflow_mode in ["高级模式 (仅导出未翻译脚本)", "高级模式 (导入已翻译脚本并渲染)"]:
        config.translation.provider = "OpenAI-Compatible"
        config.translation.model_name = "bypass-model"
        config.translation.openai_compatible_url = "http://bypass.local"

    try:
        # Create temporary config for validation since we only have the font pack name
        rendering_cfg_for_val = RenderingConfig(
            font_dir=selected_font_pack_name,
            max_font_size=config.rendering.max_font_size,
            min_font_size=config.rendering.min_font_size,
            line_spacing_mult=config.rendering.line_spacing_mult,
            font_hinting=config.rendering.font_hinting,
        )
        yolo_model_path, font_dir_path = validate_core_inputs(
            translation_cfg=config.translation,
            rendering_cfg=rendering_cfg_for_val,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            bubble_detector_model=config.detection.bubble_detector_model,
        )
        config.yolo_model_path = str(yolo_model_path)
        config.rendering.font_dir = str(font_dir_path)
    except (FileNotFoundError, ValidationError, ValueError) as e:
        raise e

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_output_path = output_base_dir / timestamp

    def _batch_progress_callback(value, desc="Processing..."):
        if gradio_progress is not None:
            gradio_progress(value, desc=desc)
        elif config.verbose:
            log_message(f"Progress: {desc} [{value * 100:.1f}%]", verbose=True)
        if cancellation_manager and cancellation_manager.is_cancelled():
            raise CancellationError("Batch process cancelled by user.")

    temp_dir_path_obj = None
    zip_temp_dir_obj = None
    try:
        process_dir = None
        preserve_structure = False

        if (
            isinstance(input_dir_or_files, dict)
            and "zip" in input_dir_or_files
            and "files" in input_dir_or_files
        ):
            zip_path_str = input_dir_or_files["zip"]
            files_list = input_dir_or_files["files"]

            temp_dir_path_obj = tempfile.TemporaryDirectory()
            combined_temp_dir = Path(temp_dir_path_obj.name)

            zip_path = Path(zip_path_str)
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file '{zip_path_str}' does not exist.")

            extracted_path, zip_temp_dir_obj = extract_zip_to_temp(zip_path)

            for item in extracted_path.iterdir():
                dest = combined_temp_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            if files_list:
                files_subdir = combined_temp_dir / "uploaded_files"
                files_subdir.mkdir(exist_ok=True)

                image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
                image_files_to_copy = []
                skipped_files = []
                for f_path_str in files_list:
                    p = Path(f_path_str)
                    if p.is_file() and p.suffix.lower() in image_extensions:
                        try:
                            with Image.open(p) as img:
                                img.verify()
                            image_files_to_copy.append(p)
                        except Exception as img_err:
                            skipped_files.append(f"{p.name} (Invalid: {img_err})")
                    else:
                        skipped_files.append(f"{p.name} (Not a supported image file)")

                if image_files_to_copy:
                    for img_file in image_files_to_copy:
                        try:
                            shutil.copy2(img_file, files_subdir / img_file.name)
                        except Exception as copy_err:
                            log_message(
                                f"Failed to copy {img_file.name}: {copy_err}",
                                always_print=True,
                            )

                if skipped_files:
                    log_message(
                        f"Skipped {len(skipped_files)} invalid files from directory upload",
                        verbose=config.verbose,
                    )

            process_dir = combined_temp_dir
            preserve_structure = True

        elif isinstance(input_dir_or_files, list):
            if not input_dir_or_files:
                raise ValidationError("No files provided for batch processing.")

            temp_dir_path_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir_path_obj.name)

            image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
            image_files_to_copy = []
            skipped_files = []
            for f_path_str in input_dir_or_files:
                p = Path(f_path_str)
                if p.is_file() and p.suffix.lower() in image_extensions:
                    try:
                        with Image.open(p) as img:
                            img.verify()
                        image_files_to_copy.append(p)
                    except Exception as img_err:
                        skipped_files.append(f"{p.name} (Invalid: {img_err})")
                else:
                    skipped_files.append(f"{p.name} (Not a supported image file)")

            if not image_files_to_copy:
                raise ValidationError(
                    f"No valid image files found in the selection. "
                    f"Skipped: {', '.join(skipped_files) if skipped_files else 'None'}"
                )

            if skipped_files:
                log_message(
                    f"Preparing {len(image_files_to_copy)} files (skipped {len(skipped_files)} invalid files)",
                    verbose=config.verbose,
                )
            else:
                log_message(
                    f"Preparing {len(image_files_to_copy)} files for batch processing",
                    verbose=config.verbose,
                )

            if image_files_to_copy:
                # Find common path to preserve directory structure
                flatten_files = True
                for img_file in image_files_to_copy:
                    try:
                        rel_path = Path(img_file.name)
                        parts = img_file.parts
                        if "gradio" in parts:
                            idx = parts.index("gradio")
                            if idx + 2 < len(parts):
                                extracted_rel_path = Path(*parts[idx+2:])
                                if extracted_rel_path.name == img_file.name:
                                    rel_path = extracted_rel_path
                                    if len(rel_path.parts) > 1:
                                        flatten_files = False

                        dest_path = temp_dir_path / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img_file, dest_path)
                        log_message(
                            f"Copied {img_file.name} to {dest_path}",
                            verbose=config.verbose,
                        )
                    except Exception as copy_err:
                        log_message(
                            f"Failed to copy {img_file.name}: {copy_err}", always_print=True
                        )
            process_dir = temp_dir_path
            preserve_structure = not flatten_files

        elif isinstance(input_dir_or_files, str):
            input_path = validate_batch_input_path(input_dir_or_files)

            if input_path.is_file() and input_path.suffix.lower() == ".zip":
                extracted_path, zip_temp_dir_obj = extract_zip_to_temp(input_path)
                process_dir = extracted_path
                preserve_structure = True
            elif input_path.is_dir():
                process_dir = input_path
                preserve_structure = False

        if not process_dir:
            raise LogicError("Could not determine processing directory.")

        # Determine original input name for output folder
        input_name = "batch_output"
        if isinstance(input_dir_or_files, str):
            input_name = Path(input_dir_or_files).stem
        elif isinstance(input_dir_or_files, dict) and "zip" in input_dir_or_files:
            input_name = Path(input_dir_or_files["zip"]).stem
        elif isinstance(input_dir_or_files, list) and input_dir_or_files:
            try:
                # Try to extract the top-level directory name from the uploaded files
                for f in input_dir_or_files:
                    p = Path(f)
                    parts = p.parts
                    if "gradio" in parts:
                        idx = parts.index("gradio")
                        if idx + 2 < len(parts):
                            # parts[idx] is 'gradio', parts[idx+1] is hash, parts[idx+2] is the folder name!
                            # if it's a file, it will be the filename, but if we have multiple files with the same parent, we can detect it.
                            parent_name = parts[idx+2]
                            if parent_name != p.name:
                                input_name = parent_name
                                break
            except Exception:
                pass

        mode_mapping = {
            "高级模式 (全图上下文自动关联 API)": "auto_api_full",
            "高级模式 (仅导出未翻译脚本)": "export_only",
            "高级模式 (导入已翻译脚本并渲染)": "import_render"
        }
        pipeline_mode = mode_mapping.get(batch_workflow_mode, "auto_api_full")

        # Helper to get unique output directory
        def get_unique_output_dir(base_dir: Path, name: str) -> Path:
            def _has_images(d: Path) -> bool:
                try:
                    return any(f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"] for f in d.rglob("*") if f.is_file())
                except Exception:
                    return False

            out_dir = base_dir / name
            if not out_dir.exists():
                return out_dir
            elif pipeline_mode == "import_render" and not _has_images(out_dir):
                return out_dir

            counter = 1
            while True:
                candidate = base_dir / f"{name}-{counter}"
                if not candidate.exists():
                    return candidate
                elif pipeline_mode == "import_render" and not _has_images(candidate):
                    return candidate
                counter += 1

        # Determine tasks based on Large Directory Mode
        tasks = []

        # If process_dir has exactly one folder (e.g. the top-level 'mahou' uploaded via Gradio) and no files, step into it
        # so Large Directory Mode correctly identifies the actual subfolders inside 'mahou'
        tmp_files = [f for f in process_dir.iterdir() if f.is_file()]
        tmp_dirs = [d for d in process_dir.iterdir() if d.is_dir()]
        if len(tmp_files) == 0 and len(tmp_dirs) == 1:
            process_dir = tmp_dirs[0]
            if len(process_dir.name) < 20: # ignore hash directories
                input_name = process_dir.name

        if batch_large_directory_mode:
            # Look for subdirectories in process_dir
            subdirs = [d for d in process_dir.iterdir() if d.is_dir()]
            if not subdirs:
                # Fallback if no subdirectories found
                tasks.append({
                    "input_dir": process_dir,
                    "output_dir": get_unique_output_dir(output_base_dir, process_dir.name),
                    "preserve_structure": preserve_structure
                })
            else:
                for subdir in subdirs:
                    tasks.append({
                        "input_dir": subdir,
                        "output_dir": get_unique_output_dir(output_base_dir, subdir.name),
                        "preserve_structure": False # Already isolated to a subdirectory
                    })
        else:
            tasks.append({
                "input_dir": process_dir,
                "output_dir": get_unique_output_dir(output_base_dir, input_name),
                "preserve_structure": preserve_structure
            })

        total_files = 0
        final_output_path = tasks[0]["output_dir"] if tasks else output_base_dir / timestamp

        for task_idx, task in enumerate(tasks):
            t_input_dir = task["input_dir"]
            t_output_dir = task["output_dir"]
            t_preserve = task["preserve_structure"]

            _batch_progress_callback(task_idx / max(1, len(tasks)), desc=f"Processing {t_input_dir.name}...")

            # =========================================================================
            # 核心分流逻辑：根据 UI 选择的 batch_workflow_mode 决定执行哪套管线
            # =========================================================================
            if batch_workflow_mode == "标准模式 (逐页处理)":
                # 兼容原版的逐页翻译模式
                _ = batch_translate_images(
                    input_dir=t_input_dir,
                    config=config,
                    output_dir=t_output_dir,
                    progress_callback=_batch_progress_callback,
                    preserve_structure=t_preserve,
                    cancellation_manager=cancellation_manager,
                )
            else:
                # 激活高级两遍处理管线 (Two-Pass Architecture)
                from core.pipeline import run_advanced_batch_pipeline

                # 如果是外部导入翻译文本模式
                if pipeline_mode == "import_render":
                    t_output_dir.mkdir(parents=True, exist_ok=True)
                    if batch_large_directory_mode:
                        # Auto-locate the scripts in the corresponding output directory or input directory
                        # 1. Check in previously created output directory without the '-X' suffix if we had to create a new one
                        original_out_dir = output_base_dir / t_input_dir.name
                        script_txt = original_out_dir / "manga_script_translated.txt"
                        script_json = original_out_dir / "manga_script.json"

                        # 2. Check in the current actual output directory just in case
                        if not script_txt.exists():
                            script_txt = t_output_dir / "manga_script_translated.txt"
                        if not script_json.exists():
                            script_json = t_output_dir / "manga_script.json"

                        # 3. Check in input directory
                        if not script_txt.exists():
                            script_txt = t_input_dir / "manga_script_translated.txt"
                        if not script_json.exists():
                            script_json = t_input_dir / "manga_script.json"

                        if not script_txt.exists() or not script_json.exists():
                            # Skip this subdir if scripts are not found
                            log_message(f"Skipping {t_input_dir.name}: missing manga_script_translated.txt or manga_script.json", always_print=True)
                            continue

                        # If found in input dir, copy to output dir for processing
                        if script_txt.parent != t_output_dir:
                            shutil.copy2(script_txt, t_output_dir / "manga_script_translated.txt")
                        if script_json.parent != t_output_dir:
                            shutil.copy2(script_json, t_output_dir / "manga_script.json")
                    else:
                        if not batch_script_upload:
                            raise ValidationError("Please upload the translated script file (TXT).")
                        if not batch_json_upload:
                            raise ValidationError("Please upload the manga_script.json coordinates file.")

                        shutil.copy2(batch_script_upload, t_output_dir / "manga_script_translated.txt")
                        shutil.copy2(batch_json_upload, t_output_dir / "manga_script.json")

                run_advanced_batch_pipeline(
                    input_dir=t_input_dir,
                    config=config,
                    output_dir=t_output_dir,
                    mode=pipeline_mode,
                    cancellation_manager=cancellation_manager
                )

            file_count = len(list(t_input_dir.glob("*.*"))) if pipeline_mode != "export_only" else 0
            total_files += file_count

        # 伪造一个 results 字典让前端 UI 画廊显示和统计不报错
        results = {
            "success_count": total_files,
            "error_count": 0,
            "errors": {},
            "output_path": final_output_path
        }

        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        results["output_path"] = batch_output_path

        log_message(
            f"Batch completed: {results['success_count']} success, "
            f"{results['error_count']} errors in {processing_time:.2f}s",
            verbose=config.verbose,
        )
        if results["errors"]:
            log_message("Processing errors:", verbose=config.verbose)
            for fname, err in results["errors"].items():
                log_message(f"  {fname}: {err}", verbose=config.verbose)

        return results

    except (FontError, RenderingError) as e:
        raise LogicError(
            f"Text rendering failed during batch processing: {str(e)}"
        ) from e
    except CleaningError as e:
        raise LogicError(
            f"Bubble cleaning failed during batch processing: {str(e)}"
        ) from e
    except TranslationError as e:
        raise LogicError(f"Translation failed during batch processing: {str(e)}") from e
    except ImageProcessingError as e:
        raise LogicError(
            f"Image processing failed during batch processing: {str(e)}"
        ) from e
    except CancellationError as e:
        raise e
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise LogicError(
            f"An unexpected error occurred during batch processing: {str(e)}"
        ) from e

    finally:
        if temp_dir_path_obj:
            try:
                temp_dir_path_obj.cleanup()
                log_message("Cleaned up temp directory", verbose=config.verbose)
            except Exception as e_clean:
                log_message(
                    f"Failed to clean up temp directory: {e_clean}",
                    always_print=True,
                )
        if zip_temp_dir_obj:
            try:
                zip_temp_dir_obj.cleanup()
                log_message(
                    "Cleaned up ZIP extraction temp directory", verbose=config.verbose
                )
            except Exception as e_clean:
                log_message(
                    f"Failed to clean up ZIP extraction temp directory: {e_clean}",
                    always_print=True,
                )
