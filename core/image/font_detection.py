import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms

REPO_ID = "gyrojeff/YuzuMarker.FontDetection"
DEFAULT_CKPT = "name=4x-epoch=84-step=1649340.ckpt"

class FontDetector:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def _load_model(self) -> nn.Module:
        print(f"Downloading/Loading {DEFAULT_CKPT} from {REPO_ID} ...")
        ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=DEFAULT_CKPT)

        # Load the state dict
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" not in state:
            raise RuntimeError("Unexpected checkpoint format: missing state_dict")
        state_dict = state["state_dict"]

        # The state dict has a prefix `model._orig_mod.model.`
        # We need to strip this to match standard resnet keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model._orig_mod.model."):
                new_key = k.replace("model._orig_mod.model.", "")
                new_state_dict[new_key] = v

        # Initialize standard ResNet50
        model = resnet50(weights=None)
        # Modify the fc layer to match the 6162 output classes
        model.fc = nn.Linear(model.fc.in_features, 6162)

        # Load the stripped state dict
        model.load_state_dict(new_state_dict)
        return model

    @torch.no_grad()
    def infer(self, pil_image: Image.Image) -> dict:
        """
        Runs font detection on a cropped image of text.
        Returns a dictionary containing predicted styling properties.
        """
        # Ensure RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        tensor_img = self.transform(pil_image).unsqueeze(0).to(self.device)

        logits = self.model(tensor_img)
        # The output dimension is 6162.
        # Indices 0..6150 are font logits.
        # Indices 6150..6152 are direction logits (horizontal vs vertical)
        # Indices 6152..6162 are regression parameters for color/stroke

        FONT_COUNT = 6150
        REGRESSION_START = FONT_COUNT + 2

        regression_logits = logits[0, REGRESSION_START:REGRESSION_START+10]
        # Apply sigmoid to convert to 0-1 scale
        regression = torch.sigmoid(regression_logits).cpu().numpy()

        # Parse colors and stroke parameters
        def clamp01(v):
            return max(0.0, min(1.0, float(v)))

        text_color_rgb = (
            int(round(clamp01(regression[0]) * 255.0)),
            int(round(clamp01(regression[1]) * 255.0)),
            int(round(clamp01(regression[2]) * 255.0)),
        )

        # font_size_px = clamp01(regression[3]) * width
        stroke_width_ratio = clamp01(regression[4])

        stroke_color_rgb = (
            int(round(clamp01(regression[5]) * 255.0)),
            int(round(clamp01(regression[6]) * 255.0)),
            int(round(clamp01(regression[7]) * 255.0)),
        )

        # We can also do color clamping like Koharu does (pure black/white if close)
        def clamp_bw(c):
            # check if gray
            is_gray = max(c) - min(c) <= 10
            t_black = 60 if is_gray else 12
            if c[0] <= t_black and c[1] <= t_black and c[2] <= t_black:
                return (0, 0, 0)
            t_white = 255 - (60 if is_gray else 12)
            if c[0] >= t_white and c[1] >= t_white and c[2] >= t_white:
                return (255, 255, 255)
            return c

        def colors_similar(a, b):
            return all(abs(a[i] - b[i]) <= 16 for i in range(3))

        text_color_rgb = clamp_bw(text_color_rgb)
        stroke_color_rgb = clamp_bw(stroke_color_rgb)

        if stroke_width_ratio > 0.0 and colors_similar(text_color_rgb, stroke_color_rgb):
            stroke_width_ratio = 0.0
            stroke_color_rgb = text_color_rgb

        return {
            "text_color_rgb": text_color_rgb,
            "stroke_color_rgb": stroke_color_rgb,
            "stroke_width_ratio": stroke_width_ratio
        }

# Global singleton
_font_detector_instance = None

def get_font_detector(device: torch.device) -> FontDetector:
    global _font_detector_instance
    if _font_detector_instance is None:
        _font_detector_instance = FontDetector(device)
    return _font_detector_instance
