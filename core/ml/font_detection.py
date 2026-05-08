import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import json
from dataclasses import dataclass
from typing import List, Optional

FONT_COUNT = 6150
REGRESSION_START = FONT_COUNT + 2
REGRESSION_DIM = 10

@dataclass
class TopFont:
    index: int
    score: float

@dataclass
class NamedFontPrediction:
    index: int
    name: str
    language: Optional[str]
    probability: float
    serif: bool

@dataclass
class FontPrediction:
    top_fonts: List[TopFont]
    named_fonts: List[NamedFontPrediction]
    direction: str
    text_color: tuple
    stroke_color: tuple
    font_size_px: float
    stroke_width_px: float
    line_height: float
    angle_deg: float

class FontDetectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50()
        self.model.fc = nn.Linear(2048, FONT_COUNT + REGRESSION_DIM + 2)

    def forward(self, x):
        return self.model(x)

    @classmethod
    def from_pretrained(cls, safetensors_path):
        import safetensors.torch
        model = cls()
        state_dict = safetensors.torch.load_file(safetensors_path)

        # fix keys by stripping model._orig_mod.model.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model._orig_mod.model.'):
                new_state_dict[k.replace('model._orig_mod.model.', '')] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        return model

class FontDetector:
    def __init__(self, device="cuda"):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download('fffonion/yuzumarker-font-detection', r'yuzumarker-font-detection.safetensors', token=os.environ.get('HF_TOKEN'))
        self.model = FontDetectorModel.from_pretrained(path).to(device).eval()
        self.device = device

        labels_path = hf_hub_download('fffonion/yuzumarker-font-detection', r'font-labels-ex.json', token=os.environ.get('HF_TOKEN'))
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

    @torch.no_grad()
    def inference(self, pil_images, top_k=5):
        import numpy as np
        if not pil_images:
            return []

        processed = []
        original_sizes = []
        for img in pil_images:
            original_sizes.append(img.width)
            # resize to 512x512
            img_resized = img.resize((512, 512), resample=3)
            tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
            processed.append(tensor)

        batch = torch.stack(processed).to(self.device)
        logits = self.model(batch)
        rows = logits.cpu().numpy()

        predictions = []
        for row, width in zip(rows, original_sizes):
            font_logits = row[:FONT_COUNT]
            exp_logits = np.exp(font_logits - np.max(font_logits))
            softmax = exp_logits / np.sum(exp_logits)

            top_indices = np.argsort(softmax)[::-1][:min(top_k, FONT_COUNT)]
            ranked = [TopFont(index=int(idx), score=float(softmax[idx])) for idx in top_indices]

            named_fonts = []
            for tf in ranked:
                if tf.index < len(self.labels):
                    label = self.labels[tf.index]
                    named_fonts.append(NamedFontPrediction(
                        index=tf.index,
                        name=label.get('path', ''),
                        language=label.get('language'),
                        probability=tf.score,
                        serif=label.get('serif', False)
                    ))

            direction = "Vertical" if row[FONT_COUNT + 1] > row[FONT_COUNT] else "Horizontal"

            regression = row[REGRESSION_START:REGRESSION_START + REGRESSION_DIM]

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            def clamp01(x):
                return max(0.0, min(1.0, float(x)))

            reg_sig = [sigmoid(v) for v in regression]

            text_color = (
                int(round(clamp01(reg_sig[2]) * 255)), # R
                int(round(clamp01(reg_sig[1]) * 255)), # G
                int(round(clamp01(reg_sig[0]) * 255)), # B
            )

            font_size_px = clamp01(reg_sig[3]) * width
            stroke_width_px = clamp01(reg_sig[4]) * width

            stroke_color = (
                int(round(clamp01(reg_sig[7]) * 255)), # R
                int(round(clamp01(reg_sig[6]) * 255)), # G
                int(round(clamp01(reg_sig[5]) * 255)), # B
            )

            line_spacing_px = clamp01(reg_sig[8]) * width
            line_height = 1.0 + (line_spacing_px / font_size_px) if font_size_px > 0 else 1.2

            angle_deg = (reg_sig[9] - 0.5) * 180.0

            predictions.append(FontPrediction(
                top_fonts=ranked,
                named_fonts=named_fonts,
                direction=direction,
                text_color=text_color,
                stroke_color=stroke_color,
                font_size_px=font_size_px,
                stroke_width_px=stroke_width_px,
                line_height=line_height,
                angle_deg=angle_deg
            ))

        return predictions
