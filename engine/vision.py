"""
Uncensored Local Vision Engine.
Combines Deep Visual Captioning (BLIP) with Optical Character Recognition (Tesseract OCR)
to provide 100% unmoderated, unbiased visual perception without GPU VRAM contention.
"""
from __future__ import annotations

import io
import os
import re
import base64
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple

try:  # Pillow is optional: without it, image analysis reports a clear error
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on host environment
    Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False

# Global cached model instances
_processor = None
_model = None

class VisionEngine:
    @classmethod
    def get_model(cls):
        """Loads and caches the vision model on CPU."""
        global _processor, _model
        if _processor is None or _model is None:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print("[*] Loading uncensored local vision model (BLIP) on CPU...")
            _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            _model.eval()
            print("[+] Vision model ready on CPU.")
        return _processor, _model

    @staticmethod
    def run_ocr(img: Image.Image) -> str:
        """Extracts optical character text, signs, code, and labels using Tesseract."""
        try:
            # Save to temporary in-memory PNG
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                # Convert to RGB and resize moderately for fast OCR
                img.convert("RGB").save(tmp_path)
            
            res = subprocess.run(
                ["tesseract", tmp_path, "stdout", "--oem", "1", "-l", "eng"],
                capture_output=True,
                text=True,
                timeout=5
            )
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            text = res.stdout.strip()
            # Clean up whitespace
            cleaned = re.sub(r"\n\s*\n", "\n", text).strip()
            return cleaned
        except Exception as e:
            return ""

    @classmethod
    def generate_caption(cls, img: Image.Image, prompt_hint: str = "") -> str:
        """Generates an objective, uncensored visual description of the image."""
        try:
            processor, model = cls.get_model()
            img_rgb = img.convert("RGB")
            # Downscale if massive to save memory
            max_dim = 768
            if max(img_rgb.size) > max_dim:
                img_rgb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            if prompt_hint.strip():
                # Conditional captioning / VQA style
                inputs = processor(img_rgb, prompt_hint.strip(), return_tensors="pt")
            else:
                # Unconditional captioning
                inputs = processor(img_rgb, return_tensors="pt")

            import torch
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=60, num_beams=3)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            return caption
        except Exception as e:
            return f"Image uploaded (visual analysis error: {e})"

    @classmethod
    def analyze(cls, image_path_or_base64: str, user_prompt: str = "") -> Dict[str, str]:
        """
        Full analysis pipeline:
        Takes an image file path or base64 data string,
        runs OCR and visual captioning, and returns structured context.
        """
        try:
            if not PIL_AVAILABLE:
                raise RuntimeError("Pillow (PIL) is not installed for this Python interpreter")
            if image_path_or_base64.startswith("data:image"):
                # Data URL
                header, encoded = image_path_or_base64.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                img = Image.open(io.BytesIO(img_bytes))
            elif len(image_path_or_base64) > 300 and not os.path.exists(image_path_or_base64):
                # Raw base64 string
                img_bytes = base64.b64decode(image_path_or_base64)
                img = Image.open(io.BytesIO(img_bytes))
            else:
                img = Image.open(image_path_or_base64)

            # 1. Optical Character Recognition (OCR)
            ocr_text = cls.run_ocr(img)

            # 2. General visual caption
            general_caption = cls.generate_caption(img)

            # 3. Targeted visual query if user prompt is specific
            targeted_caption = ""
            if len(user_prompt.split()) > 3 and "?" in user_prompt:
                targeted_caption = cls.generate_caption(img, prompt_hint=f"a photo of")

            # Build high-density prompt block
            lines = ["[ATTACHED IMAGE VISUAL ANALYSIS - UNCENSORED & OBJECTIVE]"]
            if general_caption:
                lines.append(f"- Scene & Visual Description: {general_caption}")
            if ocr_text:
                lines.append(f"- Extracted Text from Image (OCR):\n```\n{ocr_text[:800]}\n```")

            combined_block = "\n".join(lines)
            return {
                "context_block": combined_block,
                "caption": general_caption,
                "ocr_text": ocr_text
            }
        except Exception as e:
            return {
                "context_block": f"[ATTACHED IMAGE: Analysis failed: {e}]",
                "caption": "",
                "ocr_text": ""
            }
