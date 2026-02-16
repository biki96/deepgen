from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from PIL import Image
from xtuner.registry import BUILDER

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "configs" / "deepgen_scb.py")


class DeepGenPipeline:
    def __init__(
        self,
        checkpoint: str | None = None,
        config_path: str = _DEFAULT_CONFIG,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.device = torch.device(device)

        config = Config.fromfile(config_path)
        self.model = BUILDER.build(config.model)

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, weights_only=True)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing parameters: {missing}")
            if unexpected:
                print(f"Unexpected parameters: {unexpected}")

        self.model.to(device=self.device, dtype=self.model.dtype)
        self.model.eval()

        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def text2image(
        self,
        prompt: str | list[str],
        *,
        cfg_prompt: str = "",
        cfg_scale: float = 4.0,
        num_steps: int = 50,
        height: int = 512,
        width: int = 512,
        num_images: int = 1,
        progress_bar: bool = False,
    ) -> list[Image.Image]:
        if isinstance(prompt, list):
            prompts = [p.strip() for p in prompt]
        else:
            prompts = [prompt.strip()] * num_images
        cfg_prompts = [cfg_prompt] * len(prompts)

        samples = self.model.generate(
            prompt=prompts,
            cfg_prompt=cfg_prompts,
            pixel_values_src=None,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            progress_bar=progress_bar,
            generator=self.generator,
            height=height,
            width=width,
        )
        return self._postprocess(samples)

    def image2image(
        self,
        prompt: str | list[str],
        image: Image.Image,
        *,
        cfg_prompt: str = "blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.",
        cfg_scale: float = 4.0,
        num_steps: int = 50,
        height: int = 512,
        width: int = 512,
        num_images: int = 1,
        progress_bar: bool = False,
    ) -> list[Image.Image]:
        if isinstance(prompt, list):
            prompts = [p.strip() for p in prompt]
        else:
            prompts = [prompt.strip()] * num_images
        cfg_prompts = [cfg_prompt] * len(prompts)

        src_tensor = self._process_image(image, height, width)
        pixel_values_src = torch.stack([src_tensor[None]] * len(prompts)).to(self.model.dtype)

        samples = self.model.generate(
            prompt=prompts,
            cfg_prompt=cfg_prompts,
            pixel_values_src=pixel_values_src,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            progress_bar=progress_bar,
            generator=self.generator,
            height=height,
            width=width,
        )
        return self._postprocess(samples)

    @staticmethod
    def _process_image(image: Image.Image, height: int, width: int) -> torch.Tensor:
        image = image.resize(size=(width, height))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        return pixel_values.permute(2, 0, 1)

    @staticmethod
    def _postprocess(samples: torch.Tensor) -> list[Image.Image]:
        images = samples.permute(0, 2, 3, 1)
        images = torch.clamp(127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        return [Image.fromarray(img) for img in images]
