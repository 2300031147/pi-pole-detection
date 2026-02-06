from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .config import load_config
from .detect import detect_from_image, result_to_dict


def _iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based pole voltage detection.")
    parser.add_argument("--image", type=Path, help="Path to a single image.")
    parser.add_argument("--image-dir", type=Path, help="Directory of images to process.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument("--output", type=Path, help="Output JSON path.")
    parser.add_argument("--output-dir", type=Path, help="Directory for JSON outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image and not args.image_dir:
        raise SystemExit("Provide --image or --image-dir.")
    if args.image and args.image_dir:
        raise SystemExit("Use either --image or --image-dir, not both.")

    config = load_config(args.config)

    if args.image:
        result = detect_from_image(args.image, config)
        payload = result_to_dict(result)
        output_path = args.output or Path("outputs/result.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    output_dir = args.output_dir or Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in _iter_images(args.image_dir):
        result = detect_from_image(image_path, config)
        payload = result_to_dict(result)
        output_path = output_dir / f"{image_path.stem}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
