import functools
from pathlib import Path
from typing import Any

import gradio as gr

from . import callbacks, settings_manager, utils

_ALPHABETICAL_LANGUAGES = [
    "Afrikaans",
    "Albanian",
    "Arabic",
    "Armenian",
    "Bengali",
    "Bosnian",
    "Bulgarian",
    "Catalan",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Persian (Farsi)",
    "Finnish",
    "French",
    "Galician",
    "Georgian",
    "German",
    "Greek",
    "Gujarati",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Icelandic",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kannada",
    "Korean",
    "Latvian",
    "Lithuanian",
    "Malay",
    "Marathi",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Romanian",
    "Russian",
    "Serbian (Cyrillic)",
    "Serbian (Latin)",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tamil",
    "Telugu",
    "Filipino (Tagalog)",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Uzbek",
    "Vietnamese",
    "Welsh",
]

SOURCE_LANGUAGES = [
    "Japanese",
    "Korean",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
] + [
    lang
    for lang in _ALPHABETICAL_LANGUAGES
    if lang
    not in ["Japanese", "Korean", "Chinese (Simplified)", "Chinese (Traditional)"]
]

TARGET_LANGUAGES = ["English"] + [
    lang for lang in _ALPHABETICAL_LANGUAGES if lang != "English"
]

# Languages supported by PaddleOCR-VL-1.5 (53 of the 59 in _ALPHABETICAL_LANGUAGES)
_PADDLE_OCR_VL_UNSUPPORTED = frozenset(
    ["Armenian", "Georgian", "Gujarati", "Hebrew", "Kannada", "Punjabi"]
)
PADDLE_OCR_VL_LANGUAGES = [
    lang for lang in _ALPHABETICAL_LANGUAGES if lang not in _PADDLE_OCR_VL_UNSUPPORTED
]

js_credits = """
function() {
    const footer = document.querySelector('footer');
    if (footer) {
        // Check if credits already exist
        if (footer.parentNode.querySelector('.mangatl-credits')) {
            return;
        }
        const newContent = document.createElement('div');
        newContent.className = 'mangatl-credits'; // Add a class for identification
        newContent.innerHTML = 'made by <a href="https://github.com/meangrinch">grinnch</a> with ❤️'; // credits

        newContent.style.textAlign = 'center';
        newContent.style.paddingTop = '50px';
        newContent.style.color = 'lightgray';

        // Style the hyperlink
        const link = newContent.querySelector('a');
        if (link) {
            link.style.color = 'gray';
            link.style.textDecoration = 'underline';
        }

        footer.parentNode.insertBefore(newContent, footer);
    }
}
"""

js_status_fade = """
() => {
    // Find the specific config status element by its ID
    const statusElement = document.getElementById('config_status_message');  // Config status

    // Apply fade logic only to the config status element
    if (statusElement) {
        if (statusElement && statusElement.textContent.trim() !== "") {
            clearTimeout(statusElement.fadeTimer);
            clearTimeout(statusElement.resetTimer);

            statusElement.style.display = 'block';
            statusElement.style.transition = 'none';
            statusElement.style.opacity = '1';

            const fadeDelay = 3000;
            const fadeDuration = 1000;

            statusElement.fadeTimer = setTimeout(() => {
                statusElement.style.transition = `opacity ${fadeDuration}ms ease-out`;
                statusElement.style.opacity = '0';

                statusElement.resetTimer = setTimeout(() => {
                    statusElement.style.display = 'none';
                    statusElement.style.opacity = '1';
                    statusElement.style.transition = 'none';
                }, fadeDuration);

            }, fadeDelay);
        } else {
            // Ensure hidden if empty
            statusElement.style.display = 'none';
        }
    }
}
"""

js_refresh_button_reset = """
() => {
    setTimeout(() => {
        const refreshButton = document.querySelector('.config-refresh-button button');
         if (refreshButton) {
            refreshButton.textContent = 'Refresh Models / Fonts';
            refreshButton.disabled = false;
        }
    }, 100); // Small delay to ensure Gradio update cycle completes
}
"""

js_refresh_button_processing = """
() => {
    const refreshButton = document.querySelector('.config-refresh-button button');
    if (refreshButton) {
        refreshButton.textContent = 'Refreshing...';
        refreshButton.disabled = true;
    }
    return []; // Required for JS function input/output
}
"""


js_reset_status_height = """
() => {
    setTimeout(() => {
        const ids = ['#translator_status_message textarea', '#batch_status_message textarea'];
        ids.forEach(selector => {
            const el = document.querySelector(selector);
            if (el) {
                el.style.height = '';
                el.style.removeProperty('height');
            }
        });
    }, 100);
}
"""


def create_layout(
    models_dir: Path, fonts_base_dir: Path, target_device: Any
) -> gr.Blocks:
    """Creates the Gradio UI layout and connects callbacks."""

    with gr.Blocks(
        title="MangaTranslator", js=js_credits, css_paths="style.css"
    ) as app:

        gr.Markdown("# 漫画翻译器 (MangaTranslator)")

        font_choices, initial_default_font = utils.get_available_font_packs(
            fonts_base_dir
        )
        saved_settings = settings_manager.get_saved_settings()

        saved_font_pack = saved_settings.get("font_pack")
        default_font = (
            saved_font_pack
            if saved_font_pack in font_choices
            else (initial_default_font if initial_default_font else None)
        )
        batch_saved_font_pack = saved_settings.get("batch_font_pack")
        batch_default_font = (
            batch_saved_font_pack
            if batch_saved_font_pack in font_choices
            else (initial_default_font if initial_default_font else None)
        )

        saved_osb_font_pack = saved_settings.get("outside_text_osb_font_pack", "")
        if saved_osb_font_pack not in ([""] + font_choices):
            saved_osb_font_pack = ""
        saved_batch_osb_font_pack = saved_settings.get(
            "batch_outside_text_osb_font_pack",
            saved_osb_font_pack,
        )
        if saved_batch_osb_font_pack not in ([""] + font_choices):
            saved_batch_osb_font_pack = ""
        batch_default_bubble_detector_model = saved_settings.get(
            "batch_bubble_detector_model",
            saved_settings.get("bubble_detector_model", "yolo_1"),
        )
        batch_default_padding_pixels = saved_settings.get(
            "batch_padding_pixels",
            saved_settings.get("padding_pixels", 5.0),
        )
        batch_default_outside_text_enabled = saved_settings.get(
            "batch_outside_text_enabled",
            saved_settings.get("outside_text_enabled", False),
        )
        batch_default_reading_direction = saved_settings.get(
            "batch_reading_direction",
            saved_settings.get("reading_direction", "rtl"),
        )

        initial_provider = saved_settings.get(
            "provider", settings_manager.DEFAULT_SETTINGS["provider"]
        )
        initial_model_name = saved_settings.get("model_name")

        if initial_provider == "OpenRouter" or initial_provider == "OpenAI-Compatible":
            initial_models_choices = [initial_model_name] if initial_model_name else []
        else:
            initial_models_choices = settings_manager.PROVIDER_MODELS.get(
                initial_provider, []
            )

        saved_max_tokens = saved_settings.get("max_tokens")
        if saved_max_tokens is not None:
            initial_max_tokens = saved_max_tokens
        else:
            is_reasoning = utils.is_reasoning_model(
                initial_provider, initial_model_name
            )
            initial_max_tokens = 16384 if is_reasoning else 4096

        # Calculate initial max_tokens maximum based on provider/model
        initial_max_tokens_cap = utils.get_max_tokens_cap(
            initial_provider, initial_model_name
        )
        initial_max_tokens_maximum = (
            initial_max_tokens_cap if initial_max_tokens_cap is not None else 63488
        )

        # --- Define UI Components ---
        with gr.Tabs():
            with gr.TabItem("翻译器"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="filepath",
                            label="上传图片",
                            show_download_button=False,
                            image_mode=None,
                            elem_id="translator_input_image",
                        )
                        font_dropdown = gr.Dropdown(
                            choices=font_choices,
                            label="文本字体",
                            value=default_font,
                            filterable=False,
                        )
                        outside_text_osb_font_pack = gr.Dropdown(
                            value=saved_osb_font_pack,
                            choices=[""] + font_choices,
                            label="框外字字体",
                            info="画外音/拟声词默认字体；留空则使用文本字体。",
                            filterable=False,
                        )
                        with gr.Accordion("生成参数", open=True):
                            bubble_detector_model = gr.Radio(
                                choices=["yolo_1", "yolo_2", "yolo_3"],
                                value=lambda k="bubble_detector_model", d="yolo_1": settings_manager.get_saved_settings().get(k, d),
                                label="气泡检测模型",
                                info="选择本次生成使用的主气泡检测模型。",
                            )
                            config_reading_direction = gr.Radio(
                                choices=["rtl", "ltr"],
                                label="阅读方向 (Reading Direction)",
                                value=lambda k="reading_direction", d="rtl": settings_manager.get_saved_settings().get(k, d),
                                info="气泡排序方向（rtl=日漫从右到左，ltr=美漫从左到右）。",
                                elem_id="translator_reading_direction",
                            )
                            padding_pixels = gr.Slider(
                                0,
                                50,
                                value=lambda k="padding_pixels", d=5.0: settings_manager.get_saved_settings().get(k, d),
                                step=1,
                                label="预留空白像素 (Padding Pixels)",
                                info="文字距离气泡边缘的安全预留像素值。",
                            )
                            outside_text_enabled = gr.Checkbox(
                                value=lambda k="outside_text_enabled", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="启用画外音检测 (OSB Detection)",
                                info="检测、清理并翻译气泡外文字、拟声词或旁白。",
                            )
                        with gr.Accordion("翻译设定", open=True):
                            # Hidden state to store original language selection
                            original_language_state = gr.State(
                                value=lambda k="input_language", d="Japanese": settings_manager.get_saved_settings().get(k, d)
                            )
                            input_language = gr.Dropdown(
                                SOURCE_LANGUAGES,
                                label="源语言 (Source Language)",
                                value=lambda k="input_language", d="Japanese": settings_manager.get_saved_settings().get(k, d),
                                allow_custom_value=True,
                            )
                            output_language = gr.Dropdown(
                                TARGET_LANGUAGES,
                                label="目标语言 (Target Language)",
                                value=lambda k="output_language", d="English": settings_manager.get_saved_settings().get(k, d),
                                allow_custom_value=True,
                            )
                        special_instructions = gr.Textbox(
                            label="提示词 / 特殊指令 (Prompt)",
                            placeholder="给大语言模型提供可选的背景设定、角色名、排版格式要求等...",
                            value=lambda k="special_instructions", d="": settings_manager.get_saved_settings().get(k, d),
                            lines=1,
                            max_lines=10,
                            elem_id="translator_special_instructions",
                        )
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            type="pil",
                            label="翻译后图片",
                            interactive=False,
                            elem_id="translator_output_image",
                        )
                        status_message = gr.Textbox(
                            label="运行状态",
                            interactive=False,
                            elem_id="translator_status_message",
                        )
                        with gr.Row():
                            translate_button = gr.Button("开始翻译", variant="primary")
                            clear_button = gr.Button("清除")
                            cancel_button = gr.Button(
                                "取消", variant="stop", visible=False
                            )

            with gr.TabItem("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_files = gr.File(
                            label="上传图片或文件夹",
                            file_count="directory",
                            file_types=["image"],
                            type="filepath",
                        )
                        input_zip = gr.File(
                            label="上传 ZIP 压缩包 (保留目录结构)",
                            file_count="single",
                            file_types=[".zip"],
                            type="filepath",
                        )
                        batch_font_dropdown = gr.Dropdown(
                            choices=font_choices,
                            label="文本字体",
                            value=batch_default_font,
                            filterable=False,
                        )
                        batch_outside_text_osb_font_pack = gr.Dropdown(
                            value=saved_batch_osb_font_pack,
                            choices=[""] + font_choices,
                            label="框外字字体",
                            info="批量画外音/拟声词默认字体；留空则使用文本字体。",
                            filterable=False,
                        )
                        with gr.Accordion("生成参数", open=True):
                            batch_bubble_detector_model = gr.Radio(
                                choices=["yolo_1", "yolo_2", "yolo_3"],
                                value=batch_default_bubble_detector_model,
                                label="气泡检测模型",
                                info="选择本次批量生成使用的主气泡检测模型。",
                            )
                            batch_reading_direction = gr.Radio(
                                choices=["rtl", "ltr"],
                                label="阅读方向 (Reading Direction)",
                                value=batch_default_reading_direction,
                                info="批量气泡排序方向（rtl=日漫从右到左，ltr=美漫从左到右）。",
                                elem_id="batch_reading_direction",
                            )
                            batch_padding_pixels = gr.Slider(
                                0,
                                50,
                                value=batch_default_padding_pixels,
                                step=1,
                                label="预留空白像素 (Padding Pixels)",
                                info="文字距离气泡边缘的安全预留像素值。",
                            )
                            batch_outside_text_enabled = gr.Checkbox(
                                value=batch_default_outside_text_enabled,
                                label="启用画外音检测 (OSB Detection)",
                                info="检测、清理并翻译气泡外文字、拟声词或旁白。",
                            )
                        with gr.Accordion("翻译设定", open=True):
                            # Hidden state to store original language selection
                            batch_original_language_state = gr.State(
                                value=lambda k="batch_input_language", d="Japanese": settings_manager.get_saved_settings().get(k, d)
                            )
                            batch_input_language = gr.Dropdown(
                                SOURCE_LANGUAGES,
                                label="源语言 (Source Language)",
                                value=lambda k="batch_input_language", d="Japanese": settings_manager.get_saved_settings().get(k, d),
                                allow_custom_value=True,
                            )
                            batch_output_language = gr.Dropdown(
                                TARGET_LANGUAGES,
                                label="目标语言 (Target Language)",
                                value=lambda k="batch_output_language", d="English": settings_manager.get_saved_settings().get(k, d),
                                allow_custom_value=True,
                            )
                        batch_special_instructions = gr.Textbox(
                            label="提示词 / 特殊指令 (Prompt)",
                            placeholder="给大语言模型提供可选的背景设定、角色名、排版格式要求等...",
                            value=lambda k="batch_special_instructions", d="": settings_manager.get_saved_settings().get(k, d),
                            lines=1,
                            max_lines=10,
                            elem_id="batch_special_instructions",
                        )
                        batch_parallel_requests = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=int(saved_settings.get("batch_parallel_requests", 1)),
                            step=1,
                            label="并发请求数 (Parallel Requests)",
                            info="同时处理的图片数量。提高可加快速度，但消耗更多显存和网络资源。",
                        )
                        batch_previous_context_image_count = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=int(
                                (
                                    saved_settings.get(
                                        "batch_previous_context_image_count", 0
                                    )
                                    if saved_settings.get(
                                        "send_full_page_context", True
                                    )
                                    else 0
                                )
                            ),
                            step=1,
                            label="上一页上下文图像数",
                            info="批量翻译时发送前面若干页的全页图像作为剧情上下文。仅 LLM OCR 且启用全页上下文时生效。",
                            interactive=(
                                saved_settings.get("send_full_page_context", True)
                                and saved_settings.get("ocr_method", "LLM") == "LLM"
                            ),
                        )
                        batch_previous_context_text_count = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=int(
                                saved_settings.get(
                                    "batch_previous_context_text_count", 3
                                )
                            ),
                            step=1,
                            label="上一页 OCR 文本数",
                            info="批量翻译时发送前面若干页的 OCR 文本记录作为剧情上下文。可在并发时等待必要前页文本。",
                        )
                        batch_workflow_mode = gr.Radio(
                            choices=[
                                "标准模式 (逐页处理)",
                                "高级模式 (全图上下文自动关联 API)",
                                "高级模式 (仅导出未翻译脚本)",
                                "高级模式 (导入已翻译脚本并渲染)",
                            ],
                            value="标准模式 (逐页处理)",
                            label="批量工作流模式",
                            info="选择你想要的翻译流程。'Standard' 是默认的逐页处理。'Advanced' 利用二次扫描架构获取全章上下文（需配合两步走工作流）。",
                        )
                        batch_large_directory_mode = gr.Checkbox(
                            value=False,
                            label="大目录模式",
                            info="自动处理第一级所有子文件夹（每个子文件夹作为一个独立输出）",
                        )
                        batch_large_directory_path = gr.Textbox(
                            label="输入大目录所在路径 (绝对路径)",
                            placeholder="例如: D:\\manga\\mahou",
                            visible=False,
                            interactive=True,
                        )
                        batch_script_upload = gr.File(
                            label="上传翻译后脚本 (TXT)",
                            file_types=[".txt"],
                            type="filepath",
                            visible=False,
                        )

                        batch_json_upload = gr.File(
                            label="上传坐标信息 (manga_script.json)",
                            file_types=[".json"],
                            type="filepath",
                            visible=False,
                        )

                        def _update_batch_visibility(mode, ldm):
                            is_import = (mode == "高级模式 (导入已翻译脚本并渲染)") and not ldm
                            return (
                                gr.update(visible=is_import),
                                gr.update(visible=is_import),
                                gr.update(visible=not ldm),
                                gr.update(visible=not ldm),
                                gr.update(visible=ldm, interactive=True)
                            )

                        batch_workflow_mode.change(
                            fn=_update_batch_visibility,
                            inputs=[batch_workflow_mode, batch_large_directory_mode],
                            outputs=[batch_script_upload, batch_json_upload, input_files, input_zip, batch_large_directory_path],
                            queue=False,
                        )
                        batch_large_directory_mode.change(
                            fn=_update_batch_visibility,
                            inputs=[batch_workflow_mode, batch_large_directory_mode],
                            outputs=[batch_script_upload, batch_json_upload, input_files, input_zip, batch_large_directory_path],
                            queue=False,
                        )

                    with gr.Column(scale=1):
                        batch_output_gallery = gr.Gallery(
                            label="翻译后图库",
                            show_label=True,
                            columns=4,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                        )
                        batch_status_message = gr.Textbox(
                            label="运行状态",
                            interactive=False,
                            elem_id="batch_status_message",
                        )
                        with gr.Row():
                            batch_process_button = gr.Button(
                                "开始批量翻译", variant="primary"
                            )
                            batch_clear_button = gr.Button("清除")
                            batch_cancel_button = gr.Button(
                                "取消", variant="stop", visible=False
                            )

            with gr.TabItem("核心设置", elem_id="settings-tab-container"):
                config_initial_provider = initial_provider
                config_initial_model_name = initial_model_name
                config_initial_models_choices = initial_models_choices

                with gr.Row(elem_id="config-button-row"):
                    save_config_btn = gr.Button(
                        "保存设置", variant="primary", scale=2
                    )
                    import_config_btn = gr.UploadButton(
                        "导入设置", file_types=[".json"], variant="secondary", scale=1
                    )
                    export_config_btn = gr.Button(
                        "导出设置", variant="secondary", scale=1
                    )
                    reset_defaults_btn = gr.Button(
                        "恢复默认设置", variant="secondary", scale=1
                    )
                export_config_file = gr.File(label="导出的配置文件", visible=False)

                # Assign specific ID for JS targeting
                config_status = gr.Markdown(elem_id="config_status_message")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, elem_id="settings-nav"):
                        nav_buttons = []
                        setting_groups = []
                        nav_button_detection = gr.Button(
                            "检测 (Detection)",
                            elem_classes=["nav-button", "nav-button-selected"],
                        )
                        nav_buttons.append(nav_button_detection)
                        nav_button_cleaning = gr.Button(
                            "擦除 (Cleaning)", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_cleaning)
                        nav_button_translation = gr.Button(
                            "翻译 (Translation)", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_translation)
                        nav_button_rendering = gr.Button(
                            "渲染 (Rendering)", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_rendering)
                        nav_button_outside_text = gr.Button(
                            "拟声词 (OSB Text)", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_outside_text)
                        nav_button_output = gr.Button(
                            "输出 (Output)", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_output)
                        nav_button_other = gr.Button("其他 (Other)", elem_classes="nav-button")
                        nav_buttons.append(nav_button_other)

                    with gr.Column(scale=4, elem_id="config-content-area"):
                        # --- Detection Settings ---
                        with gr.Group(
                            visible=True, elem_classes="settings-group"
                        ) as group_detection:
                            gr.Markdown("### 对话框检测 (Speech Bubble Detection)")
                            settings_bubble_detector_model = gr.Radio(
                                choices=["yolo_1", "yolo_2", "yolo_3"],
                                visible=False,
                                value=lambda k="bubble_detector_model", d="yolo_1": settings_manager.get_saved_settings().get(k, d),
                                label="对话框检测模型",
                                info=(
                                    "Primary detector. yolo_3 uses "
                                    "ogkalu/comic-speech-bubble-detector-yolov8m."
                                ),
                            )
                            confidence = gr.Slider(
                                0.1,
                                1.0,
                                value=lambda k="confidence", d=0.6: settings_manager.get_saved_settings().get(k, d),
                                step=0.05,
                                label="对话框置信度阈值",
                                info="值越低检测到的对话框越多，但可能包含误判。",
                            )
                            conjoined_detection_checkbox = gr.Checkbox(
                                value=lambda k="conjoined_detection", d=True: settings_manager.get_saved_settings().get(k, d),
                                label="连体气泡分割 (Conjoined Bubble Detection)",
                                info=None,
                            )
                            conjoined_confidence = gr.Slider(
                                0.1,
                                1.0,
                                value=lambda k="conjoined_confidence", d=0.35: settings_manager.get_saved_settings().get(k, d),
                                step=0.05,
                                label="连体气泡置信度阈值",
                                info="提高阈值可过滤误判，但可能漏掉部分连体气泡。",
                                interactive=saved_settings.get(
                                    "conjoined_detection", True
                                ),
                            )
                            use_panel_sorting_checkbox = gr.Checkbox(
                                value=lambda k="use_panel_sorting", d=True: settings_manager.get_saved_settings().get(k, d),
                                label="使用基于分镜的排序 (Panel-aware Sorting)",
                                info=None,
                            )
                            panel_confidence = gr.Slider(
                                0.05,
                                1.0,
                                value=lambda k="panel_confidence", d=0.25: settings_manager.get_saved_settings().get(k, d),
                                step=0.05,
                                label="分镜置信度阈值",
                                info="提高阈值可过滤误判，但可能漏掉部分分镜。",
                                interactive=saved_settings.get(
                                    "use_panel_sorting", True
                                ),
                            )
                            seg_model = gr.Radio(
                                choices=["sam3", "sam2", "yolo"],
                                value=lambda k="seg_model", d="yolo": settings_manager.get_saved_settings().get(k, d),
                                label="图像分割模型 (Segmentation Model)",
                                info=None,
                            )
                            osb_text_verification_checkbox = gr.Checkbox(
                                value=lambda k="use_osb_text_verification", d=True: settings_manager.get_saved_settings().get(k, d),
                                label="使用 AnimeText 验证气泡",
                                info=None,
                            )
                        setting_groups.append(group_detection)

                        # --- Cleaning Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_cleaning:
                            gr.Markdown("### 掩码清理与优化 (Mask Cleaning & Refinement)")
                            thresholding_value = gr.Slider(
                                0,
                                255,
                                value=lambda k="thresholding_value", d=190: settings_manager.get_saved_settings().get(k, d),
                                step=1,
                                label="固定二值化阈值",
                                info=None,
                                interactive=not saved_settings.get(
                                    "use_otsu_threshold", False
                                ),
                            )
                            use_otsu_threshold = gr.Checkbox(
                                value=lambda k="use_otsu_threshold", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="强制使用自动阈值 (Otsu's Method)",
                                info=None,
                            )
                            roi_shrink_px = gr.Slider(
                                0,
                                10,
                                value=lambda k="roi_shrink_px", d=5: settings_manager.get_saved_settings().get(k, d),
                                step=1,
                                label="收缩阈值运算区域 (px)",
                                info=None,
                            )

                        setting_groups.append(group_cleaning)

                        # --- Translation Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_translation:
                            gr.Markdown("### 文本识别与翻译 (OCR & Translation)")
                            config_translation_mode = gr.Radio(
                                choices=["one-step", "two-step"],
                                label="翻译模式",
                                value=lambda k="translation_mode", d=settings_manager.DEFAULT_SETTINGS[
                                        "translation_mode"
                                    ],: settings_manager.get_saved_settings().get(k, d),
                                info=None,
                                elem_id="config_translation_mode",
                            )
                            initial_ocr_method = saved_settings.get(
                                "ocr_method",
                                settings_manager.DEFAULT_SETTINGS.get(
                                    "ocr_method", "LLM"
                                ),
                            )
                            ocr_method_radio = gr.Radio(
                                choices=["LLM", "paddleocr-vl"],
                                label="OCR 方法 (光学字符识别)",
                                value=initial_ocr_method,
                                info=None,
                                elem_id="ocr_method_radio",
                                interactive=saved_settings.get(
                                    "translation_mode",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "translation_mode"
                                    ],
                                )
                                != "one-step",
                            )

                            gr.Markdown("### 大模型设置 (LLM Settings)")
                            available_providers = utils.get_available_providers(
                                initial_ocr_method
                            )
                            initial_provider_value = (
                                config_initial_provider
                                if config_initial_provider in available_providers
                                else (
                                    available_providers[0]
                                    if available_providers
                                    else "Google"
                                )
                            )
                            if initial_provider_value != config_initial_provider:
                                config_initial_provider = initial_provider_value
                            provider_selector = gr.Radio(
                                choices=available_providers,
                                label="大语言模型提供商 (Provider)",
                                value=initial_provider_value,
                                elem_id="provider_selector",
                            )
                            provider_state = gr.State(
                                value=initial_provider_value,
                            )
                            google_api_key = gr.Textbox(
                                label="Google API 密钥",
                                placeholder="Enter Google AI Studio API key (starts with AI...)",
                                type="password",
                                value=lambda k="google_api_key", d="": settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Google"),
                                elem_id="google_api_key",
                                info="将保存在本地。也可通过环境变量设置。",
                            )
                            openai_api_key = gr.Textbox(
                                label="OpenAI API 密钥",
                                placeholder="Enter OpenAI API key (starts with sk-...)",
                                type="password",
                                value=lambda k="openai_api_key", d="": settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenAI"),
                                elem_id="openai_api_key",
                                info="将保存在本地。也可通过环境变量设置。",
                            )
                            anthropic_api_key = gr.Textbox(
                                label="Anthropic API 密钥",
                                placeholder="Enter Anthropic API key (starts with sk-ant-...)",
                                type="password",
                                value=lambda k="anthropic_api_key", d="": settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Anthropic"),
                                elem_id="anthropic_api_key",
                                info="将保存在本地。也可通过环境变量设置。",
                            )
                            xai_api_key = gr.Textbox(
                                label="xAI API 密钥",
                                placeholder="Enter xAI API key (starts with xai-...)",
                                type="password",
                                value=lambda k="xai_api_key", d="": settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(config_initial_provider == "xAI"),
                                elem_id="xai_api_key",
                                info="将保存在本地。也可通过环境变量设置。",
                            )
                            deepseek_api_key = gr.Textbox(
                                label="DeepSeek API 密钥",
                                placeholder="Enter DeepSeek API key (starts with sk-...)",
                                type="password",
                                value=lambda k="deepseek_api_key", d="": settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(config_initial_provider == "DeepSeek"),
                                elem_id="deepseek_api_key",
                                info="将保存在本地。也可通过环境变量设置。",
                            )
                            zai_api_key = gr.Textbox(
                                label="Z.ai API 密钥",
                                placeholder="Enter Z.ai API key",
                                type="password",
                                value=lambda k="zai_api_key", d="": settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Z.ai"),
                                elem_id="zai_api_key",
                                info="将保存在本地。也可通过环境变量设置。",
                            )
                            moonshot_api_key = gr.Textbox(
                                label="Moonshot (Kimi) API 密钥",
                                placeholder="Enter Moonshot API key (starts with sk-...)",
                                type="password",
                                value=lambda k="moonshot_api_key", d="": settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Moonshot AI"),
                                elem_id="moonshot_api_key",
                                info="将保存在本地。也可通过环境变量设置。",
                            )
                            openrouter_api_key = gr.Textbox(
                                label="OpenRouter API 密钥",
                                placeholder="Enter OpenRouter API key (starts with sk-or-...)",
                                type="password",
                                value=lambda k="openrouter_api_key", d="": settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenRouter"),
                                elem_id="openrouter_api_key",
                                info="将保存在本地。也可通过环境变量设置。",
                            )
                            openai_compatible_url_input = gr.Textbox(
                                label="OpenAI 兼容接口 Base URL",
                                placeholder="Enter Base URL (e.g., http://localhost:1234/v1)",
                                type="text",
                                value=lambda k="openai_compatible_url", d=settings_manager.DEFAULT_SETTINGS[
                                        "openai_compatible_url"
                                    ],: settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(
                                    config_initial_provider == "OpenAI-Compatible"
                                ),
                                elem_id="openai_compatible_url_input",
                                info="OpenAI 兼容接口的 Base URL 地址（例如本地部署的 http://localhost:1234/v1）。",
                            )
                            openai_compatible_api_key_input = gr.Textbox(
                                label="OpenAI 兼容接口 API 密钥 (可选)",
                                placeholder="Enter API key if required",
                                type="password",
                                value=lambda k="openai_compatible_api_key", d="": settings_manager.get_saved_settings().get(k, d),
                                show_copy_button=False,
                                visible=(
                                    config_initial_provider == "OpenAI-Compatible"
                                ),
                                elem_id="openai_compatible_api_key_input",
                                info="将保存在本地。也可通过环境变量设置。",
                            )
                            config_model_name = gr.Dropdown(
                                choices=config_initial_models_choices,
                                label="选择模型 (Model)",
                                value=config_initial_model_name,
                                info="Select the specific model for the chosen provider.",
                                elem_id="config_model_name",
                                allow_custom_value=True,
                            )
                            (
                                _initial_reasoning_effort_visible,
                                _initial_reasoning_effort_choices,
                                _initial_reasoning_effort_default,
                            ) = utils.get_reasoning_effort_config(
                                config_initial_provider, config_initial_model_name
                            )

                            _initial_reasoning_effort_value = saved_settings.get(
                                "reasoning_effort"
                            )
                            if _initial_reasoning_effort_value is None:
                                _initial_reasoning_effort_value = (
                                    _initial_reasoning_effort_default
                                )
                            elif (
                                _initial_reasoning_effort_choices
                                and _initial_reasoning_effort_value
                                not in _initial_reasoning_effort_choices
                            ):
                                _initial_reasoning_effort_value = (
                                    _initial_reasoning_effort_default
                                )
                            elif not _initial_reasoning_effort_choices:
                                _initial_reasoning_effort_value = None

                            _initial_reasoning_effort_info = (
                                utils.get_reasoning_effort_info_text(
                                    config_initial_provider,
                                    config_initial_model_name,
                                    _initial_reasoning_effort_choices,
                                )
                            )

                            _initial_reasoning_effort_label = (
                                utils.get_reasoning_effort_label(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )

                            reasoning_effort_dropdown = gr.Radio(
                                choices=_initial_reasoning_effort_choices,
                                label=_initial_reasoning_effort_label,
                                value=_initial_reasoning_effort_value,
                                info=_initial_reasoning_effort_info,
                                visible=_initial_reasoning_effort_visible,
                                elem_id="reasoning_effort_dropdown",
                            )

                            # Effort dropdown (Claude Opus 4.5/4.6/4.7 and Sonnet 4.6)
                            (
                                _initial_effort_visible,
                                _initial_effort_choices,
                                _initial_effort_default,
                            ) = utils.get_effort_config(
                                config_initial_provider, config_initial_model_name
                            )
                            _initial_effort_value = saved_settings.get("effort")
                            if _initial_effort_value is None:
                                _initial_effort_value = _initial_effort_default
                            elif (
                                _initial_effort_choices
                                and _initial_effort_value not in _initial_effort_choices
                            ):
                                _initial_effort_value = _initial_effort_default
                            elif not _initial_effort_choices:
                                _initial_effort_value = None

                            effort_dropdown = gr.Radio(
                                choices=_initial_effort_choices,
                                label="Effort",
                                value=_initial_effort_value,
                                info="Controls token spending eagerness. Opus 4.5/4.6/4.7, Sonnet 4.6 only.",
                                visible=_initial_effort_visible,
                                elem_id="effort_dropdown",
                            )

                            # Verbosity dropdown (GPT-5 series only)
                            (
                                _initial_verbosity_visible,
                                _initial_verbosity_choices,
                                _initial_verbosity_default,
                            ) = utils.get_verbosity_config(
                                config_initial_provider, config_initial_model_name
                            )
                            _initial_verbosity_value = saved_settings.get("verbosity")
                            if _initial_verbosity_value is None:
                                _initial_verbosity_value = _initial_verbosity_default
                            elif (
                                _initial_verbosity_choices
                                and _initial_verbosity_value
                                not in _initial_verbosity_choices
                            ):
                                _initial_verbosity_value = _initial_verbosity_default
                            elif not _initial_verbosity_choices:
                                _initial_verbosity_value = None

                            verbosity_dropdown = gr.Radio(
                                choices=_initial_verbosity_choices,
                                label="Verbosity",
                                value=_initial_verbosity_value,
                                info="Controls response verbosity. GPT-5 series only.",
                                visible=_initial_verbosity_visible,
                                elem_id="verbosity_dropdown",
                            )

                            _initial_enable_web_search_visible = (
                                config_initial_provider
                                not in ("OpenAI-Compatible", "DeepSeek")
                            )
                            (
                                _initial_enable_web_search_label,
                                _initial_enable_web_search_info,
                            ) = utils.get_enable_web_search_label_and_info(
                                config_initial_provider
                                if _initial_enable_web_search_visible
                                else "Google"
                            )

                            enable_web_search_checkbox = gr.Checkbox(
                                label=_initial_enable_web_search_label,
                                value=lambda k="enable_web_search", d=False: settings_manager.get_saved_settings().get(k, d),
                                info=_initial_enable_web_search_info,
                                visible=_initial_enable_web_search_visible,
                                elem_id="enable_web_search_checkbox",
                            )

                            # Compute initial visibility for enable_code_execution
                            _initial_enable_code_execution_visible = (
                                utils.is_code_execution_visible(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )

                            enable_code_execution_checkbox = gr.Checkbox(
                                label="Enable Code Execution with Images",
                                value=lambda k="enable_code_execution", d=False: settings_manager.get_saved_settings().get(k, d),
                                info="Allow Gemini 3 Flash to zoom and inspect image details using code execution.",
                                visible=_initial_enable_code_execution_visible,
                                interactive=initial_ocr_method
                                != "paddleocr-vl",
                                elem_id="enable_code_execution_checkbox",
                            )


                            # Compute initial visibility for image_detail (OpenAI only)
                            _id_visible, _id_choices, _id_default, _id_info = (
                                utils.get_image_detail_config(
                                    config_initial_provider, config_initial_model_name
                                )
                            )
                            initial_image_detail_value=lambda k="image_detail", d=_id_default: settings_manager.get_saved_settings().get(k, d)

                            image_detail_dropdown = gr.Dropdown(
                                label="Image Detail Level",
                                choices=_id_choices,
                                value=initial_image_detail_value,
                                info=_id_info,
                                visible=_id_visible,
                                interactive=initial_ocr_method != "paddleocr-vl",
                                elem_id="image_detail_dropdown",
                            )

                            # Compute initial visibility for media_resolution (Google/xAI providers only)
                            _mr_bubbles_visible_init, _, _ = (
                                utils.get_media_resolution_config(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )
                            _initial_media_resolution_visible = (
                                config_initial_provider == "Google"
                                and not _mr_bubbles_visible_init
                            )
                            initial_media_resolution_value=lambda k="media_resolution", d="auto": settings_manager.get_saved_settings().get(k, d)

                            media_resolution_dropdown = gr.Radio(
                                label="Media Resolution",
                                choices=["auto", "high", "medium", "low"],
                                value=initial_media_resolution_value,
                                info="Resolution for Gemini to process bubble/context images.",
                                visible=_initial_media_resolution_visible,
                                elem_id="media_resolution_dropdown",
                            )

                            # Compute initial visibility for Gemini 3 and xAI specific media resolution options
                            _mr_bubbles_visible, _mr_choices, _mr_info_base = (
                                utils.get_media_resolution_config(
                                    config_initial_provider, config_initial_model_name
                                )
                            )
                            _mr_bubbles_info = _mr_info_base.replace(
                                "process images", "process bubble images"
                            )
                            _mr_context_info = _mr_info_base.replace(
                                "process images", "process context (full page) images"
                            )

                            initial_media_resolution_bubbles_value=lambda k="media_resolution_bubbles", d="auto": settings_manager.get_saved_settings().get(k, d)
                            initial_media_resolution_context_value=lambda k="media_resolution_context", d="auto": settings_manager.get_saved_settings().get(k, d)

                            media_resolution_bubbles_dropdown = gr.Radio(
                                label="Media Resolution (Bubbles)",
                                choices=_mr_choices,
                                value=initial_media_resolution_bubbles_value,
                                info=_mr_bubbles_info,
                                visible=_mr_bubbles_visible,
                                elem_id="media_resolution_bubbles_dropdown",
                            )

                            media_resolution_context_dropdown = gr.Radio(
                                label="Media Resolution (Context)",
                                choices=_mr_choices,
                                value=initial_media_resolution_context_value,
                                info=_mr_context_info,
                                visible=_mr_bubbles_visible,
                                elem_id="media_resolution_context_dropdown",
                            )

                            temperature = gr.Slider(
                                0,
                                2.0,
                                value=lambda k="temperature", d=0.1: settings_manager.get_saved_settings().get(k, d),
                                step=0.05,
                                label="生成随机性 (Temperature)",
                                info="控制生成文本的创造性。值越低越稳定确定；值越高越随机、发散。",
                                elem_id="config_temperature",
                            )
                            top_p = gr.Slider(
                                0,
                                1,
                                value=lambda k="top_p", d=0.95: settings_manager.get_saved_settings().get(k, d),
                                step=0.05,
                                label="核心采样 (Top P)",
                                info="控制采样多样性。值越低越聚焦于高概率词；值越高越随机。",
                                elem_id="config_top_p",
                            )
                            top_k = gr.Slider(
                                0,
                                64,
                                value=lambda k="top_k", d=64: settings_manager.get_saved_settings().get(k, d),
                                step=1,
                                label="候选词限制 (Top K)",
                                info="将每一步的候选词池限制在概率最高的 K 个词。",
                                interactive=(
                                    config_initial_provider
                                    not in ("OpenAI", "xAI", "DeepSeek", "Moonshot AI")
                                ),
                                elem_id="config_top_k",
                            )
                            max_tokens = gr.Slider(
                                2048,
                                initial_max_tokens_maximum,
                                value=initial_max_tokens,
                                step=1024,
                                label="最大输出长度 (Max Tokens)",
                                info="大模型一次性返回的最大 Token 数量。如果发现翻译长图时被截断，请尝试调大此值。",
                                elem_id="config_max_tokens",
                            )

                            gr.Markdown("### 上下文与图像超分 (Context & Upscaling)")
                            send_full_page_context = gr.Checkbox(
                                value=lambda k="send_full_page_context", d=True: settings_manager.get_saved_settings().get(k, d),
                                label="将全页图像作为上下文发送",
                                info=None,
                                interactive=initial_ocr_method
                                != "paddleocr-vl",
                            )
                            whiteout_conjoined_bubbles = gr.Checkbox(
                                value=lambda k="whiteout_conjoined_bubbles", d=True: settings_manager.get_saved_settings().get(k, d),
                                label="涂白相连气泡 (防重叠翻译)",
                                info=None,
                            )
                            upscale_method = gr.Radio(
                                choices=[
                                    ("Model", "model"),
                                    ("Model (Lite)", "model_lite"),
                                    ("LANCZOS", "lanczos"),
                                    ("None", "none"),
                                ],
                                value=lambda k="upscale_method", d="model_lite": settings_manager.get_saved_settings().get(k, d),
                                label="气泡/全页缩放方法 (Upscale Method)",
                                info=None,
                            )
                            initial_upscale_method = saved_settings.get(
                                "upscale_method", "model_lite"
                            )
                            sliders_interactive = initial_upscale_method != "none"
                            bubble_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=lambda k="bubble_min_side_pixels", d=128: settings_manager.get_saved_settings().get(k, d),
                                step=16,
                                label="气泡裁剪图最小边长 (像素)",
                                info=None,
                                elem_id="config_bubble_min_side_pixels",
                                interactive=sliders_interactive,
                            )
                            context_image_max_side_pixels = gr.Slider(
                                512,
                                2560,
                                value=lambda k="context_image_max_side_pixels", d=1024: settings_manager.get_saved_settings().get(k, d),
                                step=128,
                                label="全页上下文最大边长 (像素)",
                                info=None,
                                elem_id="config_context_image_max_side_pixels",
                                interactive=sliders_interactive,
                            )
                            osb_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=lambda k="osb_min_side_pixels", d=128: settings_manager.get_saved_settings().get(k, d),
                                step=16,
                                label="画外音区域最小边长 (像素)",
                                info=None,
                                elem_id="config_osb_min_side_pixels",
                                interactive=sliders_interactive,
                            )
                        setting_groups.append(group_translation)

                        # --- Rendering Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_rendering:
                            gr.Markdown("### Font Rendering")
                            max_font_size = gr.State(1000)
                            min_font_size = gr.State(4)
                            line_spacing_mult = gr.Slider(
                                0.5,
                                2.0,
                                value=lambda k="line_spacing_mult", d=1.0: settings_manager.get_saved_settings().get(k, d),
                                step=0.05,
                                label="行距倍率 (Line Spacing)",
                                info="调整多行文本之间的垂直间距（1.0 = 标准）。",
                            )
                            use_subpixel_rendering = gr.Checkbox(
                                value=lambda k="use_subpixel_rendering", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="使用子像素渲染 (Subpixel Rendering)",
                                info="改善 RGB 显示器上的文字清晰度。如果使用 OLED 屏幕请禁用。",
                            )
                            font_hinting = gr.Radio(
                                choices=["none", "slight", "normal", "full"],
                                value=lambda k="font_hinting", d="none": settings_manager.get_saved_settings().get(k, d),
                                label="字体微调 (Font Hinting)",
                                info="调整字形轮廓以适应像素网格。'无 (None)' 通常最适合高分辨率图像。",
                            )
                            use_ligatures = gr.Checkbox(
                                value=lambda k="use_ligatures", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="使用标准连字 (Ligatures)",
                                info="允许将常见的字母组合渲染为单个字形（前提是字体支持）。",
                            )
                            pure_black_text = gr.Checkbox(
                                value=lambda k="pure_black_text", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="强制纯黑字体 (Pure Black Text)",
                                info="启用后所有气泡和文字都会被强制使用纯黑色渲染（(0, 0, 0)），不再尝试根据背景自动反色。",
                            )
                            gr.Markdown("### Text Layout")
                            detach_trailing_ellipsis = gr.Checkbox(
                                value=lambda k="detach_trailing_punctuation", d=True: settings_manager.get_saved_settings().get(k, settings_manager.get_saved_settings().get("detach_trailing_ellipsis", d)),
                                label="分离句尾标点",
                                info="将句尾省略号、问号、感叹号等标点簇移至新行，以改善文本排版换行。",
                            )
                            auto_vertical_text = gr.Checkbox(
                                value=lambda k="auto_vertical_text", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="高气泡自动竖排",
                                info="在细长气泡中自动尝试竖排；手动脚本 direction:vertical 仍优先采用。",
                            )
                            hyphenate_before_scaling = gr.Checkbox(
                                value=lambda k="hyphenate_before_scaling", d=True: settings_manager.get_saved_settings().get(k, d),
                                label="允许长单词使用连字符 (-)",
                                info="尝试在因过长而需要缩小字体前，插入连字符以折行。",
                            )
                            hyphen_penalty = gr.Slider(
                                100,
                                2000,
                                value=lambda k="hyphen_penalty", d=1000.0: settings_manager.get_saved_settings().get(k, d),
                                step=100,
                                label="连字符惩罚 (Hyphen Penalty)",
                                info="文本排版时增加对连字符的惩罚权重。调高可减少连字符的使用。",
                                interactive=saved_settings.get(
                                    "hyphenate_before_scaling", True
                                ),
                            )
                            hyphenation_min_word_length = gr.Slider(
                                6,
                                10,
                                value=lambda k="hyphenation_min_word_length", d=8: settings_manager.get_saved_settings().get(k, d),
                                step=1,
                                label="连字符最小单词长度",
                                info="允许使用连字符的最小单词长度。",
                                interactive=saved_settings.get(
                                    "hyphenate_before_scaling", True
                                ),
                            )
                            badness_exponent = gr.Slider(
                                2.0,
                                4.0,
                                value=lambda k="badness_exponent", d=3.0: settings_manager.get_saved_settings().get(k, d),
                                step=0.5,
                                label="不良度指数 (Badness Exponent)",
                                info="控制排版对齐松紧的惩罚指数。增加可避免行间隙过于松散。",
                            )
                            settings_padding_pixels = gr.Slider(0, 50, value=lambda k="padding_pixels", d=5.0: settings_manager.get_saved_settings().get(k, d), step=1, visible=False, label="预留空白像素 (Padding Pixels)", info="文字距离气泡边缘的安全预留像素值。增加此值可让文字更往中心聚拢。")
                            supersampling_factor = gr.Slider(
                                1,
                                16,
                                value=lambda k="supersampling_factor", d=4: settings_manager.get_saved_settings().get(k, d),
                                step=1,
                                label="超采样倍数 (Supersampling Factor)",
                                info="以 N 倍分辨率渲染文字后缩放以获取平滑边缘。值越高品质越好但占用更多显存。1 = 关闭。",
                            )
                        setting_groups.append(group_rendering)

                        # --- Outside Text Removal Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_outside_text:
                            gr.Markdown("### Outside Speech Bubble Text")
                            outside_text_huggingface_token = gr.Textbox(
                                value=lambda k="outside_text_huggingface_token", d="": settings_manager.get_saved_settings().get(k, d),
                                label="HuggingFace Token (部分功能必需)",
                                type="password",
                                info="下载检测模型 (如 SAM 3, Flux Kontext 等) 所需。也可通过系统环境变量配置。",
                            )
                            settings_outside_text_enabled = gr.Checkbox(
                                value=lambda k="outside_text_enabled", d=False: settings_manager.get_saved_settings().get(k, d),
                                visible=False,
                                label="启用画外音检测 (OSB Detection)",
                                info="检测、涂白并翻译气泡外/无边框的文字（画外音、拟声词等）。",
                            )

                            # Wrap all settings except the enable checkbox and token in a Column with visibility control
                            with gr.Column(
                                visible=True
                            ) as outside_text_settings_wrapper:
                                gr.Markdown("### Detection")
                                outside_text_osb_confidence = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=lambda k="outside_text_osb_confidence", d=0.6: settings_manager.get_saved_settings().get(k, d),
                                    step=0.05,
                                    label="画外音检测置信度",
                                    info="调低可检测更多文本，但也可能增加误判。",
                                )
                                outside_text_bbox_expansion_percent = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=lambda k="outside_text_bbox_expansion_percent", d=0.1: settings_manager.get_saved_settings().get(k, d),
                                    step=0.05,
                                    label="检测框扩展比例",
                                    info="画外音检测框向外放大的百分比。调高可捕获周围更多底色作为上下文。",
                                )
                                outside_text_text_box_proximity_ratio = gr.Slider(
                                    0.01,
                                    0.1,
                                    value=lambda k="outside_text_text_box_proximity_ratio", d=0.02: settings_manager.get_saved_settings().get(k, d),
                                    step=0.01,
                                    label="文本框合并距离比",
                                    info="用于合并临近文本框的距离比例。增加可将距离较远的零散字块合并。",
                                )
                                outside_text_enable_page_number_filtering = gr.Checkbox(
                                    value=lambda k="outside_text_enable_page_number_filtering", d=False,: settings_manager.get_saved_settings().get(k, d),
                                    label="过滤页码",
                                    info="识别页面边缘文字并丢弃可能的页码。会略微拖慢速度，且可能有误判。",
                                )
                                outside_text_page_filter_margin_threshold = gr.Slider(
                                    0.0,
                                    0.3,
                                    value=lambda k="outside_text_page_filter_margin_threshold", d=0.1,: settings_manager.get_saved_settings().get(k, d),
                                    step=0.01,
                                    label="页码边缘比例",
                                    info="触发页码过滤的垂直边缘比例阈值。",
                                    interactive=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                )
                                outside_text_page_filter_min_area_ratio = gr.Slider(
                                    0.0,
                                    0.2,
                                    value=lambda k="outside_text_page_filter_min_area_ratio", d=0.05,: settings_manager.get_saved_settings().get(k, d),
                                    step=0.01,
                                    label="页码最小面积比",
                                    info="触发页码过滤的最小面积比例。",
                                    interactive=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                )
                                gr.Markdown("### Inpainting")
                                outside_text_inpainting_method = gr.Radio(
                                    value=lambda k="outside_text_inpainting_method", d="flux_klein_4b",: settings_manager.get_saved_settings().get(k, d),
                                    choices=[
                                        ("Flux.2 Klein 9B", "flux_klein_9b"),
                                        ("Flux.2 Klein 4B", "flux_klein_4b"),
                                        ("Flux.1 Kontext (12B)", "flux_kontext"),
                                        ("OpenCV", "opencv"),
                                        ("None (text background)", "none"),
                                    ],
                                    label="涂白算法 (Inpainting Method)",
                                    info="Klein 模型较快但可能有轻微色偏。Kontext 无色偏，但模型更大且更慢。",
                                )
                                _initial_method = saved_settings.get(
                                    "outside_text_inpainting_method", "flux_klein_4b"
                                )
                                _is_kontext = _initial_method == "flux_kontext"
                                outside_text_kontext_backend = gr.Radio(
                                    choices=[
                                        ("SDNQ (cross-platform)", "sdnq"),
                                        ("Nunchaku (CUDA)", "nunchaku"),
                                    ],
                                    value=lambda k="outside_text_kontext_backend", d="sdnq": settings_manager.get_saved_settings().get(k, d),
                                    label="Kontext Backend",
                                    info=None,
                                    visible=_is_kontext,
                                )
                                _is_klein_model = _initial_method in (
                                    "flux_klein_9b",
                                    "flux_klein_4b",
                                )
                                _initial_backend = saved_settings.get(
                                    "outside_text_kontext_backend", "sdnq"
                                )
                                _show_low_vram = _is_klein_model or (
                                    _is_kontext and _initial_backend == "sdnq"
                                )
                                outside_text_flux_low_vram = gr.Checkbox(
                                    value=lambda k="outside_text_flux_low_vram", d=False: settings_manager.get_saved_settings().get(k, d),
                                    label="低显存模式 (Low VRAM)",
                                    info="启用 CPU 内存卸载以降低显存占用（处理会变慢）。",
                                    visible=_show_low_vram,
                                )
                                outside_text_flux_num_inference_steps = gr.Slider(
                                    1,
                                    (
                                        30
                                        if saved_settings.get(
                                            "outside_text_inpainting_method",
                                            "flux_klein_4b",
                                        )
                                        == "flux_kontext"
                                        else 12
                                    ),
                                    value=lambda k="outside_text_flux_num_inference_steps", d=4: settings_manager.get_saved_settings().get(k, d),
                                    step=1,
                                    label="步数 (Steps)",
                                    info="推断步数。Klein 推荐 4 步。Kontext 推荐 6-15 步。",
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    != "opencv",
                                )
                                _is_klein_for_lum = saved_settings.get(
                                    "outside_text_inpainting_method",
                                    "flux_klein_4b",
                                ) in ("flux_klein_9b", "flux_klein_4b")
                                outside_text_flux_luminance_correction = gr.Checkbox(
                                    value=lambda k="outside_text_flux_luminance_correction", d=True: settings_manager.get_saved_settings().get(k, d),
                                    label="亮度校正 (Luminance Correction)",
                                    info="尝试使涂白生成的补丁亮度匹配周围的自然环境。",
                                    visible=_is_klein_for_lum,
                                    interactive=(
                                        _is_klein_for_lum
                                        and saved_settings.get(
                                            "outside_text_flux_upscale_small_crops",
                                            True,
                                        )
                                        and not saved_settings.get(
                                            "outside_text_flux_group_regions",
                                            False,
                                        )
                                    ),
                                )
                                _is_flux_for_klein_options = saved_settings.get(
                                    "outside_text_inpainting_method",
                                    "flux_klein_4b",
                                ) not in ("opencv", "none")
                                outside_text_flux_upscale_small_crops = gr.Checkbox(
                                    value=lambda k="outside_text_flux_upscale_small_crops", d=True: settings_manager.get_saved_settings().get(k, d),
                                    label="Klein 小裁剪放大到约 1MP",
                                    info="小区域送入 Flux Klein 前先放大以提升清理质量。关闭后仅会在超过 4MP 时缩小。",
                                    visible=_is_flux_for_klein_options,
                                    interactive=_is_klein_for_lum,
                                )
                                outside_text_flux_group_regions = gr.State(
                                    value=saved_settings.get(
                                        "outside_text_flux_group_regions", False
                                    )
                                )
                                outside_text_flux_residual_diff_threshold = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=lambda k="outside_text_flux_residual_diff_threshold", d=0.15,: settings_manager.get_saved_settings().get(k, d),
                                    step=0.01,
                                    label="残差阈值 (Residual Diff Threshold)",
                                    info="Flux Kontext 首块缓存阈值。越高越快，但画质稍差。",
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    == "flux_kontext",
                                )
                                outside_text_seed = gr.Number(
                                    value=lambda k="outside_text_seed", d=1: settings_manager.get_saved_settings().get(k, d),
                                    label="随机种子 (Seed)",
                                    info="固定涂白结果的随机种子（-1 = 随机）。",
                                    precision=0,
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    not in ("opencv", "none"),
                                )
                                inpaint_colored_bubbles = gr.Checkbox(
                                    value=lambda k="inpaint_colored_bubbles", d=False: settings_manager.get_saved_settings().get(k, d),
                                    label="使用高级模型涂白非纯色气泡",
                                    info="当气泡内部不是纯白/纯黑（如彩色/带渐变灰阶）时，启用 Flux 进行无痕涂白。",
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    not in ("opencv", "none"),
                                )

                                gr.Markdown("### Font Rendering")
                                outside_text_osb_render_expansion_narrow_multiplier = gr.Slider(
                                    1.0,
                                    3.0,
                                    value=lambda k="outside_text_osb_render_expansion_narrow_multiplier", d=1.0: settings_manager.get_saved_settings().get(k, d),
                                    step=0.1,
                                    label="狭长形状渲染放大倍数 (Narrow/Tall Expansion)",
                                    info="按此倍数向外放大细长的画外音渲染框，改善细长框导致字体过小的问题。",
                                )
                                outside_text_osb_render_expansion_aspect_ratio_threshold = gr.Slider(
                                    0.05,
                                    1.0,
                                    value=lambda k="outside_text_osb_render_expansion_aspect_ratio_threshold", d=0.4: settings_manager.get_saved_settings().get(k, d),
                                    step=0.01,
                                    label="狭长形状宽长比阈值",
                                    info="当检测框的 宽/高 比例低于此值时，会被判定为细长形状并应用上述放大倍数。",
                                )
                                outside_text_osb_render_expansion_tiny_multiplier = gr.Slider(
                                    1.0,
                                    3.0,
                                    value=lambda k="outside_text_osb_render_expansion_tiny_multiplier", d=1.0: settings_manager.get_saved_settings().get(k, d),
                                    step=0.1,
                                    label="微小形状渲染放大倍数 (Tiny Expansion)",
                                    info="按此倍数向外放大极其微小的画外音渲染框。",
                                )
                                outside_text_osb_render_expansion_area_ratio_threshold = gr.Slider(
                                    0.0,
                                    0.05,
                                    value=lambda k="outside_text_osb_render_expansion_area_ratio_threshold", d=0.005: settings_manager.get_saved_settings().get(k, d),
                                    step=0.001,
                                    label="微小形状面积比阈值",
                                    info="当检测框占全图的面积比例低于此值时，会被判定为微小形状并应用上述放大倍数。",
                                )
                                settings_outside_text_osb_font_pack = gr.Dropdown(
                                    value=saved_osb_font_pack,
                                    choices=[""] + font_choices,
                                    label="文本字体",
                                    info="画外音/拟声词专用翻译字体（留空则默认使用主字体）。",
                                    visible=False,
                                )
                                outside_text_osb_max_font_size = gr.State(1000)
                                outside_text_osb_min_font_size = gr.State(4)
                                outside_text_osb_line_spacing = gr.Slider(
                                    0.5,
                                    2.0,
                                    value=lambda k="outside_text_osb_line_spacing", d=1.0: settings_manager.get_saved_settings().get(k, d),
                                    step=0.05,
                                    label="行距倍率 (Line Spacing)",
                                    info=None,
                                )
                                outside_text_osb_use_subpixel_rendering = gr.Checkbox(
                                    value=lambda k="outside_text_osb_use_subpixel_rendering", d=True: settings_manager.get_saved_settings().get(k, d),
                                    label="使用子像素渲染 (Subpixel Rendering)",
                                    info="改善 RGB 显示器上的文字清晰度。如果使用 OLED 屏幕请禁用。",
                                )
                                outside_text_osb_font_hinting = gr.Radio(
                                    choices=["none", "slight", "normal", "full"],
                                    value=lambda k="outside_text_osb_font_hinting", d="none": settings_manager.get_saved_settings().get(k, d),
                                    label="字体微调 (Font Hinting)",
                                    info="调整字形轮廓以适应像素网格。'无 (None)' 通常最适合高分辨率图像。",
                                )
                                outside_text_osb_use_ligatures = gr.Checkbox(
                                    value=lambda k="outside_text_osb_use_ligatures", d=False: settings_manager.get_saved_settings().get(k, d),
                                    label="使用标准连字 (Ligatures)",
                                    info="允许将常见的字母组合渲染为单个字形（前提是字体支持）。",
                                )
                                outside_text_osb_outline_width = gr.Slider(
                                    0,
                                    10,
                                    value=lambda k="outside_text_osb_outline_width", d=3.0: settings_manager.get_saved_settings().get(k, d),
                                    step=0.5,
                                    label="外发光描边宽度 (px)",
                                    info="画外音/拟声词文字外发光的宽度。",
                                )
                        setting_groups.append(group_outside_text)

                        # --- Output Settings ---
                        image_upscale_mode_default = saved_settings.get(
                            "image_upscale_mode", "off"
                        )
                        if image_upscale_mode_default not in {
                            "off",
                            "initial",
                            "final",
                        }:
                            image_upscale_mode_default = "off"
                        image_upscale_factor_default = float(
                            saved_settings.get("image_upscale_factor", 2.0)
                        )

                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_output:
                            gr.Markdown("### Output Format")
                            output_format = gr.Radio(
                                choices=["auto", "png", "jpeg"],
                                label="图像输出格式",
                                value=lambda k="output_format", d="png": settings_manager.get_saved_settings().get(k, d),
                                info="'auto' 为自动继承原图格式（若未知则默认 png）。",
                            )
                            jpeg_quality = gr.Slider(
                                1,
                                100,
                                value=lambda k="jpeg_quality", d=95: settings_manager.get_saved_settings().get(k, d),
                                step=1,
                                label="JPEG 品质",
                                info="调高可提升画质，但文件体积更大。",
                                interactive=saved_settings.get("output_format", "png")
                                != "png",
                            )
                            png_compression = gr.Slider(
                                0,
                                6,
                                value=lambda k="png_compression", d=2: settings_manager.get_saved_settings().get(k, d),
                                step=1,
                                label="PNG 压缩等级",
                                info="使用 OxiPNG 压缩。调高可减小体积，但处理更慢。",
                                interactive=saved_settings.get("output_format", "png")
                                != "jpeg",
                            )
                            gr.Markdown("### Upscaling")
                            image_upscale_mode = gr.Radio(
                                choices=["off", "initial", "final"],
                                value=image_upscale_mode_default,
                                label="图像超分处理模式",
                                info="决定是超分'初始原图'还是'翻译后的最终图'。'初始'文字边缘更清晰，但全图风格可能不一致。",
                            )
                            image_upscale_model = gr.Radio(
                                choices=[
                                    ("Model", "model"),
                                    ("Model (Lite)", "model_lite"),
                                ],
                                value=lambda k="image_upscale_model", d="model_lite": settings_manager.get_saved_settings().get(k, d),
                                label="超分模型 (Upscaling Model)",
                                info="超分放大的模型。Model 画质最佳，Model (Lite) 稍逊但更省显存速度更快。",
                                interactive=image_upscale_mode_default != "off",
                            )
                            image_upscale_factor = gr.Slider(
                                1.0,
                                8.0,
                                value=image_upscale_factor_default,
                                step=0.1,
                                label="超分倍率 (Upscale Factor)",
                                info="超分放大的倍数（支持非整数）。",
                                interactive=image_upscale_mode_default != "off",
                            )
                            auto_scale = gr.Checkbox(
                                value=lambda k="auto_scale", d=True: settings_manager.get_saved_settings().get(k, d),
                                label="自动匹配图像分辨率缩放参数 (Auto-Scale)",
                                info="根据图像分辨率自动缩放内核、字体等参数，确保多分辨率下表现一致。",
                            )
                        setting_groups.append(group_output)

                        # --- Other Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_other:
                            gr.Markdown("### Other")
                            refresh_resources_button = gr.Button(
                                "Refresh Models / Fonts",
                                variant="secondary",
                                elem_classes="config-button",
                            )
                            unload_models_button = gr.Button(
                                "Force Unload Models",
                                variant="secondary",
                                elem_classes="config-button",
                            )
                            verbose = gr.Checkbox(
                                value=lambda k="verbose", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="启用终端详细日志输出 (Verbose Logging)",
                                info="Enable verbose logging in console.",
                            )
                            cleaning_only_toggle = gr.Checkbox(
                                value=lambda k="cleaning_only", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="仅涂白模式 (Cleaning-only Mode)",
                                info="跳过翻译和文字排版，只输出涂白清除掉日文的纯净底图。",
                                interactive=not (
                                    saved_settings.get("test_mode", False)
                                    or saved_settings.get("upscaling_only", False)
                                ),
                            )
                            upscaling_only_toggle = gr.Checkbox(
                                value=lambda k="upscaling_only", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="仅超分模式 (Upscaling-only Mode)",
                                info="跳过一切汉化管线，单纯将图片无损超分辨率放大。",
                                interactive=not (
                                    saved_settings.get("cleaning_only", False)
                                    or saved_settings.get("test_mode", False)
                                ),
                            )
                            test_mode_toggle = gr.Checkbox(
                                value=lambda k="test_mode", d=False: settings_manager.get_saved_settings().get(k, d),
                                label="排版测试模式 (Test Mode)",
                                info="跳过大模型翻译请求并节省 Token，直接使用无意义的假文(Lorem Ipsum)填充气泡以测试排版效果。",
                                interactive=not (
                                    saved_settings.get("cleaning_only", False)
                                    or saved_settings.get("upscaling_only", False)
                                ),
                            )
                        setting_groups.append(group_other)

        # --- Define Event Handlers ---
        save_config_inputs = [
            confidence,
            conjoined_confidence,
            panel_confidence,
            seg_model,
            bubble_detector_model,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_reading_direction,
            config_translation_mode,
            ocr_method_radio,
            max_font_size,
            min_font_size,
            line_spacing_mult,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            pure_black_text,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            upscaling_only_toggle,
            test_mode_toggle,
            input_language,
            output_language,
            font_dropdown,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            batch_outside_text_osb_font_pack,
            enable_web_search_checkbox,
            enable_code_execution_checkbox,
            image_detail_dropdown,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            verbosity_dropdown,
            send_full_page_context,
            whiteout_conjoined_bubbles,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            detach_trailing_ellipsis,
            auto_vertical_text,
            special_instructions,
            batch_special_instructions,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            padding_pixels,
            supersampling_factor,
            outside_text_enabled,
            outside_text_seed,
            outside_text_inpainting_method,
            outside_text_kontext_backend,
            outside_text_flux_low_vram,
            outside_text_flux_num_inference_steps,
            outside_text_flux_luminance_correction,
            outside_text_flux_upscale_small_crops,
            outside_text_flux_group_regions,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_osb_render_expansion_narrow_multiplier,
            outside_text_osb_render_expansion_tiny_multiplier,
            outside_text_osb_render_expansion_aspect_ratio_threshold,
            outside_text_osb_render_expansion_area_ratio_threshold,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_parallel_requests,
            batch_previous_context_image_count,
            batch_previous_context_text_count,
            batch_bubble_detector_model,
            batch_reading_direction,
            batch_padding_pixels,
            batch_outside_text_enabled,
        ]

        reset_outputs = [
            confidence,
            conjoined_confidence,
            panel_confidence,
            seg_model,
            bubble_detector_model,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_reading_direction,
            config_translation_mode,
            ocr_method_radio,
            max_font_size,
            min_font_size,
            line_spacing_mult,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            pure_black_text,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            upscaling_only_toggle,
            test_mode_toggle,
            input_language,
            output_language,
            font_dropdown,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            batch_outside_text_osb_font_pack,
            enable_web_search_checkbox,
            enable_code_execution_checkbox,
            image_detail_dropdown,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            verbosity_dropdown,
            config_status,
            send_full_page_context,
            whiteout_conjoined_bubbles,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            detach_trailing_ellipsis,
            auto_vertical_text,
            special_instructions,
            batch_special_instructions,
            outside_text_enabled,
            outside_text_seed,
            outside_text_inpainting_method,
            outside_text_kontext_backend,
            outside_text_flux_low_vram,
            outside_text_flux_num_inference_steps,
            outside_text_flux_luminance_correction,
            outside_text_flux_upscale_small_crops,
            outside_text_flux_group_regions,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_osb_render_expansion_narrow_multiplier,
            outside_text_osb_render_expansion_tiny_multiplier,
            outside_text_osb_render_expansion_aspect_ratio_threshold,
            outside_text_osb_render_expansion_area_ratio_threshold,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_parallel_requests,
            batch_bubble_detector_model,
            batch_reading_direction,
            batch_padding_pixels,
            batch_outside_text_enabled,
            batch_previous_context_image_count,
            batch_previous_context_text_count,
        ]

        translate_inputs = [
            input_image,
            confidence,
            conjoined_confidence,
            panel_confidence,
            seg_model,
            bubble_detector_model,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_reading_direction,
            config_translation_mode,
            ocr_method_radio,
            input_language,
            output_language,
            font_dropdown,
            max_font_size,
            min_font_size,
            line_spacing_mult,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            pure_black_text,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            upscaling_only_toggle,
            test_mode_toggle,
            enable_web_search_checkbox,
            enable_code_execution_checkbox,
            image_detail_dropdown,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            verbosity_dropdown,
            send_full_page_context,
            whiteout_conjoined_bubbles,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            detach_trailing_ellipsis,
            auto_vertical_text,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            padding_pixels,
            supersampling_factor,
            outside_text_enabled,
            outside_text_seed,
            outside_text_inpainting_method,
            outside_text_kontext_backend,
            outside_text_flux_low_vram,
            outside_text_flux_num_inference_steps,
            outside_text_flux_luminance_correction,
            outside_text_flux_upscale_small_crops,
            outside_text_flux_group_regions,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_osb_render_expansion_narrow_multiplier,
            outside_text_osb_render_expansion_tiny_multiplier,
            outside_text_osb_render_expansion_aspect_ratio_threshold,
            outside_text_osb_render_expansion_area_ratio_threshold,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            batch_outside_text_osb_font_pack,
            special_instructions,
            batch_special_instructions,
            batch_parallel_requests,
            batch_previous_context_image_count,
            batch_previous_context_text_count,
            batch_bubble_detector_model,
            batch_reading_direction,
            batch_padding_pixels,
            batch_outside_text_enabled,
        ]

        batch_inputs = [
            input_files,
            input_zip,
            confidence,
            conjoined_confidence,
            panel_confidence,
            seg_model,
            bubble_detector_model,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_reading_direction,
            config_translation_mode,
            ocr_method_radio,
            input_language,
            output_language,
            font_dropdown,
            max_font_size,
            min_font_size,
            line_spacing_mult,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            pure_black_text,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            upscaling_only_toggle,
            test_mode_toggle,
            enable_web_search_checkbox,
            enable_code_execution_checkbox,
            image_detail_dropdown,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            verbosity_dropdown,
            send_full_page_context,
            whiteout_conjoined_bubbles,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            detach_trailing_ellipsis,
            auto_vertical_text,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            padding_pixels,
            supersampling_factor,
            outside_text_enabled,
            outside_text_seed,
            outside_text_inpainting_method,
            outside_text_kontext_backend,
            outside_text_flux_low_vram,
            outside_text_flux_num_inference_steps,
            outside_text_flux_luminance_correction,
            outside_text_flux_upscale_small_crops,
            outside_text_flux_group_regions,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_osb_render_expansion_narrow_multiplier,
            outside_text_osb_render_expansion_tiny_multiplier,
            outside_text_osb_render_expansion_aspect_ratio_threshold,
            outside_text_osb_render_expansion_area_ratio_threshold,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            batch_outside_text_osb_font_pack,
            special_instructions,
            batch_special_instructions,
            batch_parallel_requests,
            batch_previous_context_image_count,
            batch_previous_context_text_count,
            batch_bubble_detector_model,
            batch_reading_direction,
            batch_padding_pixels,
            batch_outside_text_enabled,
            batch_workflow_mode,
            batch_large_directory_mode,
            batch_large_directory_path,
            batch_script_upload,
            batch_json_upload,
        ]

        # Config Tab Navigation & Updates
        output_components_for_switch = setting_groups + nav_buttons
        nav_button_detection.click(
            fn=lambda idx=0: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_cleaning.click(
            fn=lambda idx=1: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_translation.click(
            fn=lambda idx=2: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_rendering.click(
            fn=lambda idx=3: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_outside_text.click(
            fn=lambda idx=4: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_output.click(
            fn=lambda idx=5: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_other.click(
            fn=lambda idx=6: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )

        output_format.change(
            fn=callbacks.handle_output_format_change,
            inputs=output_format,
            outputs=[jpeg_quality, png_compression],
            queue=False,
        )

        image_upscale_mode.change(
            fn=lambda mode: (
                gr.update(interactive=mode != "off"),
                gr.update(interactive=mode != "off"),
            ),
            inputs=image_upscale_mode,
            outputs=[image_upscale_factor, image_upscale_model],
            queue=False,
        )

        provider_selector.change(
            fn=callbacks.handle_provider_change,
            inputs=[provider_selector, temperature, ocr_method_radio],
            outputs=[
                google_api_key,
                openai_api_key,
                anthropic_api_key,
                xai_api_key,
                deepseek_api_key,
                zai_api_key,
                moonshot_api_key,
                openrouter_api_key,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
                config_model_name,
                temperature,
                top_p,
                top_k,
                max_tokens,
                enable_web_search_checkbox,
                enable_code_execution_checkbox,
            image_detail_dropdown,
                media_resolution_dropdown,
                media_resolution_bubbles_dropdown,
                media_resolution_context_dropdown,
                reasoning_effort_dropdown,
                effort_dropdown,
                verbosity_dropdown,
            ],
            queue=False,
        ).then(  # Trigger model fetch *after* provider change updates visibility etc.
            fn=lambda prov, url, key, ocr_method: (
                utils.fetch_and_update_compatible_models(url, key, force_refresh=True)
                if prov == "OpenAI-Compatible"
                else (
                    utils.fetch_and_update_openrouter_models(ocr_method=ocr_method)
                    if prov == "OpenRouter"
                    else gr.update()
                )
            ),
            inputs=[
                provider_selector,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
                ocr_method_radio,
            ],
            outputs=[config_model_name],
            queue=True,  # Allow fetching to happen in the background
        )

        # Keep provider_state in sync with provider_selector for manual changes
        provider_selector.change(
            fn=lambda p: p,
            inputs=[provider_selector],
            outputs=[provider_state],
            queue=False,
        )

        config_model_name.change(
            fn=callbacks.handle_model_change,
            inputs=[provider_selector, config_model_name, temperature],
            outputs=[
                temperature,
                top_p,
                top_k,
                max_tokens,
                enable_web_search_checkbox,
                enable_code_execution_checkbox,
            image_detail_dropdown,
                media_resolution_dropdown,
                media_resolution_bubbles_dropdown,
                media_resolution_context_dropdown,
                reasoning_effort_dropdown,
                effort_dropdown,
                verbosity_dropdown,
            ],
            queue=False,
        )

        # Reasoning effort change → update temp/top_p slider interactivity
        reasoning_effort_dropdown.change(
            fn=callbacks.handle_reasoning_effort_change,
            inputs=[provider_selector, config_model_name, reasoning_effort_dropdown],
            outputs=[temperature, top_p],
            queue=False,
        )

        # Thresholding checkbox change handler
        use_otsu_threshold.change(
            fn=callbacks.handle_thresholding_change,
            inputs=use_otsu_threshold,
            outputs=thresholding_value,
            queue=False,
        )

        # Hyphenation checkbox change handler
        hyphenate_before_scaling.change(
            fn=callbacks.handle_hyphenation_change,
            inputs=hyphenate_before_scaling,
            outputs=[hyphen_penalty, hyphenation_min_word_length],
            queue=False,
        )

        # Cleaning-only, Upscaling-only, and Test mode mutual exclusivity handlers
        cleaning_only_toggle.change(
            fn=callbacks.handle_cleaning_only_change,
            inputs=cleaning_only_toggle,
            outputs=[upscaling_only_toggle, test_mode_toggle],
            queue=False,
        )

        # Upscaling-only toggle change handler
        upscaling_only_toggle.change(
            fn=callbacks.handle_upscaling_only_change,
            inputs=upscaling_only_toggle,
            outputs=[cleaning_only_toggle, test_mode_toggle],
            queue=False,
        )

        # Test mode toggle change handler
        test_mode_toggle.change(
            fn=callbacks.handle_test_mode_change,
            inputs=test_mode_toggle,
            outputs=[cleaning_only_toggle, upscaling_only_toggle],
            queue=False,
        )

        # OSB enable/disable handler
        outside_text_enabled.change(
            fn=lambda x: gr.update(visible=True),
            inputs=outside_text_enabled,
            outputs=outside_text_settings_wrapper,
            queue=False,
        )

        # Page-number filtering toggle -> enable/disable related sliders
        outside_text_enable_page_number_filtering.change(
            fn=lambda enabled: (
                gr.update(interactive=enabled),
                gr.update(interactive=enabled),
            ),
            inputs=outside_text_enable_page_number_filtering,
            outputs=[
                outside_text_page_filter_margin_threshold,
                outside_text_page_filter_min_area_ratio,
            ],
            queue=False,
        )

        # Inpainting method change -> enable/disable controls and adjust steps range
        def _update_inpainting_controls(
            method: str,
            current_backend: str,
            current_steps: int,
            upscale_small_crops: bool,
        ):
            """Update controls based on inpainting method selection."""
            is_opencv = method == "opencv"
            is_none = method == "none"
            is_no_flux = is_opencv or is_none
            is_kontext = method == "flux_kontext"
            is_klein = method in ("flux_klein_9b", "flux_klein_4b")
            is_flux_for_klein_options = not is_no_flux

            if is_kontext:
                max_steps = 30
                default_steps = 8
            else:
                max_steps = 12
                default_steps = 4

            show_low_vram = is_klein or (is_kontext and current_backend == "sdnq")
            residual_interactive = is_kontext and current_backend == "nunchaku"
            luminance_interactive = is_klein and bool(upscale_small_crops)

            return (
                gr.update(visible=is_kontext),
                gr.update(visible=show_low_vram),
                gr.update(
                    interactive=(not is_no_flux),
                    maximum=max_steps,
                    value=default_steps,
                ),
                gr.update(
                    visible=is_klein,
                    interactive=luminance_interactive,
                    value=luminance_interactive,
                ),
                gr.update(
                    visible=is_flux_for_klein_options,
                    interactive=is_klein,
                ),
                gr.update(interactive=residual_interactive),
                gr.update(interactive=(not is_no_flux)),
                gr.update(interactive=(not is_no_flux)),
            )

        outside_text_inpainting_method.change(
            fn=_update_inpainting_controls,
            inputs=[
                outside_text_inpainting_method,
                outside_text_kontext_backend,
                outside_text_flux_num_inference_steps,
                outside_text_flux_upscale_small_crops,
            ],
            outputs=[
                outside_text_kontext_backend,
                outside_text_flux_low_vram,
                outside_text_flux_num_inference_steps,
                outside_text_flux_luminance_correction,
                outside_text_flux_upscale_small_crops,
                outside_text_flux_residual_diff_threshold,
                outside_text_seed,
                inpaint_colored_bubbles,
            ],
            queue=False,
        )

        def _update_luminance_interactivity(upscale_small_crops: bool, method: str):
            is_klein = method in ("flux_klein_9b", "flux_klein_4b")
            interactive = is_klein and bool(upscale_small_crops)
            return gr.update(interactive=interactive, value=interactive)

        outside_text_flux_upscale_small_crops.change(
            fn=_update_luminance_interactivity,
            inputs=[outside_text_flux_upscale_small_crops, outside_text_inpainting_method],
            outputs=outside_text_flux_luminance_correction,
            queue=False,
        )

        # Kontext backend change -> update Low VRAM and Residual diff visibility
        def _update_kontext_backend_controls(backend: str):
            """Update controls when Kontext backend changes."""
            is_sdnq = backend == "sdnq"
            return (
                gr.update(visible=is_sdnq),
                gr.update(interactive=(not is_sdnq)),
            )

        outside_text_kontext_backend.change(
            fn=_update_kontext_backend_controls,
            inputs=outside_text_kontext_backend,
            outputs=[
                outside_text_flux_low_vram,
                outside_text_flux_residual_diff_threshold,
            ],
            queue=False,
        )

        outside_text_flux_luminance_correction.change(
            fn=callbacks.handle_luminance_correction_change,
            inputs=outside_text_flux_luminance_correction,
            outputs=None,
            queue=False,
        )

        # Conjoined detection change handler - clears SAM cache
        conjoined_detection_checkbox.change(
            fn=callbacks.handle_conjoined_detection_change,
            inputs=conjoined_detection_checkbox,
            outputs=conjoined_confidence,
            queue=False,
        )

        # Panel sorting change handler
        use_panel_sorting_checkbox.change(
            fn=lambda enabled: gr.update(interactive=enabled),
            inputs=use_panel_sorting_checkbox,
            outputs=panel_confidence,
            queue=False,
        )

        # Confidence threshold change handlers - clear YOLO cache
        confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=confidence,
            outputs=None,
            queue=False,
        )

        conjoined_confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=conjoined_confidence,
            outputs=None,
            queue=False,
        )

        panel_confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=panel_confidence,
            outputs=None,
            queue=False,
        )

        outside_text_osb_confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=outside_text_osb_confidence,
            outputs=None,
            queue=False,
        )

        # Translation mode change handler - disable OCR selection when one-step
        config_translation_mode.change(
            fn=callbacks.handle_translation_mode_change,
            inputs=[config_translation_mode, ocr_method_radio],
            outputs=ocr_method_radio,
            queue=False,
        )

        # OCR method change handler
        ocr_method_radio.change(
            fn=callbacks.handle_ocr_method_change,
            inputs=[
                ocr_method_radio,
                input_language,
                original_language_state,
                batch_input_language,
                batch_original_language_state,
                provider_state,
                config_model_name,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
            ],
            outputs=[
                provider_selector,
                input_language,
                original_language_state,
                batch_input_language,
                batch_original_language_state,
                send_full_page_context,
                batch_previous_context_image_count,
                whiteout_conjoined_bubbles,
                enable_code_execution_checkbox,
                media_resolution_dropdown,
                media_resolution_bubbles_dropdown,
                media_resolution_context_dropdown,
                config_model_name,
                provider_state,
            ],
            queue=False,
        )

        send_full_page_context.change(
            fn=lambda enabled: (
                gr.update(interactive=True)
                if enabled
                else gr.update(value=0, interactive=False)
            ),
            inputs=send_full_page_context,
            outputs=batch_previous_context_image_count,
            queue=False,
        )

        # Upscale method change handler
        upscale_method.change(
            fn=lambda x: [
                gr.update(interactive=x != "none"),
                gr.update(interactive=x != "none"),
                gr.update(interactive=x != "none"),
            ],
            inputs=upscale_method,
            outputs=[
                bubble_min_side_pixels,
                context_image_max_side_pixels,
                osb_min_side_pixels,
            ],
            queue=False,
        )

        # Config Save/Reset Buttons
        save_config_btn.click(
            fn=callbacks.handle_save_config_click,
            inputs=save_config_inputs,
            outputs=[config_status],
            queue=False,
        ).then(fn=None, inputs=None, outputs=None, js=js_status_fade, queue=False)

        reset_defaults_btn.click(
            fn=functools.partial(
                callbacks.handle_reset_defaults_click,
                fonts_base_dir=fonts_base_dir,
            ),
            inputs=[],
            outputs=reset_outputs,
            queue=False,
        ).then(fn=None, inputs=None, outputs=None, js=js_status_fade, queue=False)

        # Refresh Button
        refresh_outputs = [
            font_dropdown,
            batch_font_dropdown,
            outside_text_osb_font_pack,
            batch_outside_text_osb_font_pack,
        ]
        refresh_resources_button.click(
            fn=functools.partial(
                callbacks.handle_refresh_resources_click,
                fonts_base_dir=fonts_base_dir,
            ),
            inputs=[],
            outputs=refresh_outputs,
            js=js_refresh_button_processing,
        ).then(fn=None, inputs=None, outputs=None, js=js_refresh_button_reset)

        # Unload Models Button
        unload_models_button.click(
            fn=callbacks.handle_unload_models_click,
            inputs=[],
            outputs=[],
        )

        # Translator Tab Button
        clear_button.click(
            fn=lambda: (None, None, gr.update(value="", lines=1)),
            outputs=[input_image, output_image, status_message],
            queue=False,
        ).then(fn=None, js=js_reset_status_height, queue=False)
        batch_clear_button.click(
            fn=lambda: (None, None, None, gr.update(value="", lines=1)),
            outputs=[
                input_files,
                input_zip,
                batch_output_gallery,
                batch_status_message,
            ],
            queue=False,
        ).then(fn=None, js=js_reset_status_height, queue=False)
        translate_event = translate_button.click(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=True,
                button_text_processing="Translating...",
                button_text_idle="Translate",
            ),
            outputs=[
                translate_button,
                clear_button,
                cancel_button,
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
            ],
            queue=False,
        ).then(
            fn=functools.partial(
                callbacks.handle_translate_click,
                models_dir=models_dir,
                fonts_base_dir=fonts_base_dir,
                target_device=target_device,
            ),
            inputs=translate_inputs,
            outputs=[output_image, status_message],
        )
        translate_event.then(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=False,
                button_text_processing="Translating...",
                button_text_idle="Translate",
            ),
            outputs=[
                translate_button,
                clear_button,
                cancel_button,
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
            ],
            queue=False,
        ).then(fn=None, js=js_reset_status_height, queue=False)

        cancel_button.click(
            fn=callbacks.cancel_process,
            cancels=translate_event,
            queue=False,
        )

        # Batch Tab Button
        batch_event = batch_process_button.click(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=True,
                button_text_processing="Processing...",
                button_text_idle="Start Batch Translating",
            ),
            outputs=[
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
                translate_button,
                clear_button,
                cancel_button,
            ],
            queue=False,
        ).then(
            fn=functools.partial(
                callbacks.handle_batch_click,
                models_dir=models_dir,
                fonts_base_dir=fonts_base_dir,
                target_device=target_device,
            ),
            inputs=batch_inputs,
            outputs=[batch_output_gallery, batch_status_message],
        )
        batch_event.then(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=False,
                button_text_processing="Processing...",
                button_text_idle="Start Batch Translating",
            ),
            outputs=[
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
                translate_button,
                clear_button,
                cancel_button,
            ],
            queue=False,
        ).then(fn=None, js=js_reset_status_height, queue=False)

        batch_cancel_button.click(
            fn=callbacks.cancel_process,
            cancels=batch_event,
            queue=False,
        )

        app.load(
            fn=callbacks.handle_app_load,
            inputs=[
                provider_selector,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
            ],
            outputs=[config_model_name],
            queue=False,
        )

        export_config_btn.click(
            fn=callbacks.handle_export_config,
            inputs=[],
            outputs=[export_config_file],
            queue=False,
        )
        import_config_btn.upload(
            fn=callbacks.handle_import_config,
            inputs=[import_config_btn],
            outputs=[config_status],
            queue=False,
        )

    return app


