import argparse
import os

from PIL import Image

from src.pipelines import DeepGenPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Using the red color, draw one continuous path from the green start to the red end along walkable white cells only. Do not cross walls.",
    )
    parser.add_argument("--src_img", type=str, default="UniREditBench/original_image/maze/1.png")
    parser.add_argument(
        "--cfg_prompt",
        type=str,
        default="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.",
    )
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--output", type=str, default="output.jpg")

    args = parser.parse_args()

    pipe = DeepGenPipeline(checkpoint=args.checkpoint, seed=args.seed)

    if pipe.accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    src_img = Image.open(args.src_img).convert("RGB")

    images = pipe.image2image(
        prompt=args.prompt,
        image=src_img,
        cfg_prompt=args.cfg_prompt,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        num_images=args.num_images,
    )

    for i, image in enumerate(images):
        image.save(f"{args.output}/case_{i}.png")
