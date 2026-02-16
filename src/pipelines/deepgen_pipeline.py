from __future__ import annotations

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from einops import rearrange
from mmengine.config import Config
from PIL import Image
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER


class DeepGenPipeline:
    def __init__(
        self,
        checkpoint: str | None = None,
        config_path: str = "configs/models/deepgen_scb.py",
        seed: int = 42,
    ):
        self.accelerator = Accelerator()

        message = [f"Hello this is GPU {self.accelerator.process_index}"]
        messages = gather_object(message)
        self.accelerator.print(f"Number of gpus: {self.accelerator.num_processes}")
        self.accelerator.print(messages)

        config = Config.fromfile(config_path)
        print(f"Device: {self.accelerator.device}", flush=True)

        self.model = BUILDER.build(config.model)
        if checkpoint is not None:
            if checkpoint.endswith(".pt"):
                state_dict = torch.load(checkpoint)
            else:
                state_dict = guess_load_checkpoint(checkpoint)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            self.accelerator.print(f"Unexpected parameters: {unexpected}")

        self.model = self.model.to(device=self.accelerator.device)
        self.model = self.model.to(self.model.dtype)
        self.model.eval()

        self.generator = torch.Generator(device=self.model.device).manual_seed(seed)

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
            progress_bar=False,
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
            progress_bar=False,
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
        pixel_values = rearrange(pixel_values, "h w c -> c h w")
        return pixel_values

    @staticmethod
    def _postprocess(samples: torch.Tensor) -> list[Image.Image]:
        images = rearrange(samples, "b c h w -> b h w c")
        images = torch.clamp(127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        return [Image.fromarray(img) for img in images]
