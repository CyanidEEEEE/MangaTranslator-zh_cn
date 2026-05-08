import os
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_CHANNELS = [3, 32, 56, 80, 192, 328]
DECODER_CHANNELS = [256, 128, 64, 32, 16]

class EncoderConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, activation="silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == "silu":
            x = F.silu(x)
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_channels, reduced_channels, 1)
        self.conv_expand = nn.Conv2d(reduced_channels, in_channels, 1)

    def forward(self, x):
        scale = x.mean(dim=(2, 3), keepdim=True)
        scale = F.silu(self.conv_reduce(scale))
        scale = torch.sigmoid(self.conv_expand(scale))
        return x * scale

class EdgeResidual(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, stride):
        super().__init__()
        self.conv_exp = EncoderConvNormAct(in_channels, hidden_channels, 3, stride, 1, 1, "silu")
        self.conv_pwl = EncoderConvNormAct(hidden_channels, out_channels, 1, 1, 0, 1, "identity")
        self.residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        y = self.conv_pwl(self.conv_exp(x))
        return x + y if self.residual else y

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, stride):
        super().__init__()
        self.conv_pw = EncoderConvNormAct(in_channels, hidden_channels, 1, 1, 0, 1, "silu")
        self.conv_dw = EncoderConvNormAct(hidden_channels, hidden_channels, 3, stride, 1, hidden_channels, "silu")
        self.se = SqueezeExcite(hidden_channels, max(1, in_channels // 4))
        self.conv_pwl = EncoderConvNormAct(hidden_channels, out_channels, 1, 1, 0, 1, "identity")
        self.residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        y = self.conv_pw(x)
        y = self.conv_dw(y)
        y = self.se(y)
        y = self.conv_pwl(y)
        return x + y if self.residual else y

class EfficientNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = EncoderConvNormAct(3, 32, 3, 2, 1, 1, "silu")

        stage_specs = [
            [(32, 32, 32, 1, False), (32, 32, 32, 1, False), (32, 32, 32, 1, False)],
            [(32, 128, 56, 2, False), (56, 224, 56, 1, False), (56, 224, 56, 1, False), (56, 224, 56, 1, False), (56, 224, 56, 1, False)],
            [(56, 224, 80, 2, False), (80, 320, 80, 1, False), (80, 320, 80, 1, False), (80, 320, 80, 1, False), (80, 320, 80, 1, False)],
            [(80, 320, 152, 2, True), (152, 608, 152, 1, True), (152, 608, 152, 1, True), (152, 608, 152, 1, True), (152, 608, 152, 1, True), (152, 608, 152, 1, True), (152, 608, 152, 1, True), (152, 608, 152, 1, True)],
            [(152, 912, 192, 1, True)] + [(192, 1152, 192, 1, True)] * 14,
            [(192, 1152, 328, 2, True)] + [(328, 1968, 328, 1, True)] * 23,
        ]

        self.blocks = nn.ModuleList()
        for stage in stage_specs:
            stage_blocks = nn.ModuleList()
            for in_c, hid_c, out_c, stride, is_inv in stage:
                if is_inv:
                    stage_blocks.append(InvertedResidual(in_c, hid_c, out_c, stride))
                else:
                    stage_blocks.append(EdgeResidual(in_c, hid_c, out_c, stride))
            self.blocks.append(stage_blocks)

    def forward(self, x):
        h = self.conv_stem(x)
        stage_outputs = []
        for stage in self.blocks:
            for block in stage:
                h = block(h)
            stage_outputs.append(h)
        return [x, stage_outputs[0], stage_outputs[1], stage_outputs[2], stage_outputs[4], stage_outputs[5]]

class Scse(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        red = max(1, in_channels // 16)
        self.cSE = nn.Sequential(
            nn.Conv2d(in_channels, red, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red, in_channels, 1, bias=False)
        )
        self.sSE = nn.Conv2d(in_channels, 1, 1, bias=False)

    def forward(self, x):
        c = x.mean(dim=(2, 3), keepdim=True)
        c = torch.sigmoid(self.cSE(c))
        s = torch.sigmoid(self.sSE(x))
        return x * c + x * s

class DecoderConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        groups = self.get_groups(out_channels)
        self.norm = nn.GroupNorm(groups, out_channels, eps=1e-5)

    def get_groups(self, c):
        if c >= 8 and c % 8 == 0: return 8
        for g in range(min(c, 8), 1, -1):
            if c % g == 0: return g
        return 1

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)), inplace=True)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = DecoderConvRelu(in_channels + skip_channels, out_channels)
        self.attention1 = Scse(in_channels + skip_channels)
        self.conv2 = DecoderConvRelu(out_channels, out_channels)
        self.attention2 = Scse(out_channels)

    def forward(self, x, skip=None):
        h = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            h = torch.cat([h, skip], dim=1)
            h = self.attention1(h)
        h = self.conv1(h)
        h = self.conv2(h)
        return self.attention2(h)

class UnetPlusPlusDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_ch = ENCODER_CHANNELS[1:][::-1]
        in_ch = [encoder_ch[0]] + DECODER_CHANNELS[:-1]
        skip_ch = encoder_ch[1:] + [0]
        out_ch = DECODER_CHANNELS

        self.blocks = nn.ModuleDict()
        for layer_idx in range(len(in_ch) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    b_in, b_skip, b_out = in_ch[layer_idx], skip_ch[layer_idx] * (layer_idx + 1), out_ch[layer_idx]
                else:
                    b_in, b_skip, b_out = skip_ch[layer_idx - 1], skip_ch[layer_idx] * (layer_idx + 1 - depth_idx), skip_ch[layer_idx]
                self.blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(b_in, b_skip, b_out)

        self.blocks[f"x_0_{len(in_ch)-1}"] = DecoderBlock(in_ch[-1], 0, out_ch[-1])
        self.in_channels_list = in_ch
        self.depth = len(in_ch) - 1

    def forward(self, features):
        feats = features[1:][::-1]
        dense_x = {}
        for layer_idx in range(len(self.in_channels_list) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    key = f"x_{depth_idx}_{depth_idx}"
                    dense_x[key] = self.blocks[key](feats[depth_idx], feats[depth_idx + 1])
                else:
                    dense_layer = depth_idx + layer_idx
                    cat_feats = [dense_x[f"x_{idx}_{dense_layer}"] for idx in range(depth_idx + 1, dense_layer + 1)]
                    cat_feats.append(feats[dense_layer + 1])
                    skip = torch.cat(cat_feats, dim=1)
                    key = f"x_{depth_idx}_{dense_layer}"
                    dense_x[key] = self.blocks[key](dense_x[f"x_{depth_idx}_{dense_layer - 1}"], skip)

        return self.blocks[f"x_0_{self.depth}"](dense_x[f"x_0_{self.depth - 1}"])

class MangaTextSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNetEncoder()
        self.decoder = UnetPlusPlusDecoder()
        self.segmentation_head = nn.Sequential(nn.Conv2d(16, 1, 3, 1, 1))

    def forward(self, x):
        features = self.encoder(x)
        decoded = self.decoder(features)
        return self.segmentation_head(decoded)

    @classmethod
    def from_pretrained(cls, safetensors_path):
        import safetensors.torch
        model = cls()
        state_dict = safetensors.torch.load_file(safetensors_path)

        # fix keys to match pytorch implementation exactly
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('encoder.model.', 'encoder.')
            k = k.replace('cSE.1.weight', 'cSE.0.weight').replace('cSE.1.bias', 'cSE.0.bias')
            k = k.replace('cSE.3.weight', 'cSE.2.weight').replace('cSE.3.bias', 'cSE.2.bias')
            k = k.replace('attention1.attention.', 'attention1.')
            k = k.replace('attention2.attention.', 'attention2.')

            # Stem
            if k.startswith('encoder.conv_stem.weight'): k = k.replace('encoder.conv_stem.weight', 'encoder.conv_stem.conv.weight')
            elif k.startswith('encoder.bn1.'): k = k.replace('encoder.bn1.', 'encoder.conv_stem.bn.')

            # EdgeResidual / InvertedResidual
            if '.conv_exp.weight' in k: k = k.replace('.conv_exp.weight', '.conv_exp.conv.weight')
            elif '.bn1.' in k and 'blocks' in k:
                # determine if it's EdgeResidual (has conv_exp) or InvertedResidual (has conv_pw)
                # EdgeResidual is blocks.0.x and non-inverted ones.
                block_prefix = '.'.join(k.split('.')[:3])
                if '.0.0.' in k or '.0.1.' in k or '.0.2.' in k or '.1.0.' in k or '.1.1.' in k or '.1.2.' in k or '.1.3.' in k or '.1.4.' in k or '.2.0.' in k or '.2.1.' in k or '.2.2.' in k or '.2.3.' in k or '.2.4.' in k:
                    k = k.replace('.bn1.', '.conv_exp.bn.')
                else:
                    k = k.replace('.bn1.', '.conv_pw.bn.')

            if '.conv_pwl.weight' in k: k = k.replace('.conv_pwl.weight', '.conv_pwl.conv.weight')
            if '.conv_pw.weight' in k: k = k.replace('.conv_pw.weight', '.conv_pw.conv.weight')
            if '.conv_dw.weight' in k: k = k.replace('.conv_dw.weight', '.conv_dw.conv.weight')

            elif '.bn2.' in k:
                if '.0.0.' in k or '.0.1.' in k or '.0.2.' in k or '.1.0.' in k or '.1.1.' in k or '.1.2.' in k or '.1.3.' in k or '.1.4.' in k or '.2.0.' in k or '.2.1.' in k or '.2.2.' in k or '.2.3.' in k or '.2.4.' in k:
                    k = k.replace('.bn2.', '.conv_pwl.bn.')
                else:
                    k = k.replace('.bn2.', '.conv_dw.bn.')

            if '.bn3.' in k: k = k.replace('.bn3.', '.conv_pwl.bn.')

            # Decoder
            k = k.replace('.conv1.0.', '.conv1.conv.')
            k = k.replace('.conv1.1.', '.conv1.norm.')
            k = k.replace('.conv2.0.', '.conv2.conv.')
            k = k.replace('.conv2.1.', '.conv2.norm.')
            k = k.replace('.sSE.0.', '.sSE.')

            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        return model

class MangaTextSegmenter:
    def __init__(self, device="cuda"):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download('mayocream/manga-text-segmentation-2025', 'model.safetensors', token=os.environ.get('HF_TOKEN'))
        self.model = MangaTextSegmentationModel.from_pretrained(path).to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def inference(self, pil_image):
        import numpy as np
        import cv2
        original_w, original_h = pil_image.size

        # Scale logic
        max_pixels = 1536 * 1536
        area = original_w * original_h
        if area <= max_pixels:
            resized_w, resized_h = original_w, original_h
        else:
            scale = (max_pixels / area) ** 0.5
            resized_w = int(max(1, original_w * scale))
            resized_h = int(max(1, original_h * scale))
            while resized_w * resized_h > max_pixels:
                if resized_w >= resized_h and resized_w > 1:
                    resized_w -= 1
                elif resized_h > 1:
                    resized_h -= 1
                else:
                    break

        img = pil_image.resize((resized_w, resized_h), resample=3) # BICUBIC
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        img_t = (img_t - self.mean) / self.std

        pad_h = (32 - resized_h % 32) % 32
        pad_w = (32 - resized_w % 32) % 32
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h))

        logits = self.model(img_t)
        probs = torch.sigmoid(logits)[0, 0, :resized_h, :resized_w]

        if resized_w != original_w or resized_h != original_h:
            probs = probs.unsqueeze(0).unsqueeze(0)
            probs = F.interpolate(probs, size=(original_h, original_w), mode='bilinear', align_corners=False)
            probs = probs.squeeze(0).squeeze(0)

        return probs.cpu().numpy()
