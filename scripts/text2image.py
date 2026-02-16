import argparse
import os

from src.pipelines import DeepGenPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A quiet bookstore with a sign that says 'READ'. A coffee cup on the table with the word 'MORNING'.",
    )
    parser.add_argument("--cfg_prompt", type=str, default="")
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

    images = pipe.text2image(
        prompt=args.prompt,
        cfg_prompt=args.cfg_prompt,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        num_images=args.num_images,
    )

    for i, image in enumerate(images):
        image.save(f"{args.output}/case_{i}.png")
