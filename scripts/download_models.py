#!/usr/bin/env python3
"""Download OmniParser model weights from HuggingFace.

This script downloads the required model weights for OmniParser:
- icon_detect: YOLOv8-based icon detection model
- icon_caption_florence: Florence-2-based icon captioning model

Usage:
    python scripts/download_models.py [--target-dir weights]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_huggingface_cli():
    """Check if huggingface-cli is installed."""
    try:
        result = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_huggingface_hub():
    """Install huggingface_hub package."""
    print("Installing huggingface_hub...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "huggingface_hub"],
        check=True,
    )


def download_file(repo_id: str, filename: str, local_dir: Path):
    """Download a single file from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        filename: File to download
        local_dir: Local directory to save to
    """
    print(f"  Downloading: {filename}")
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            repo_id,
            filename,
            "--local-dir",
            str(local_dir),
        ],
        check=True,
    )


def download_omniparser_weights(target_dir: str = "weights"):
    """Download OmniParser v2.0 weights from HuggingFace.

    Args:
        target_dir: Directory to save weights
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    repo_id = "microsoft/OmniParser-v2.0"

    print(f"Downloading OmniParser weights to: {target}")
    print(f"Repository: {repo_id}")
    print()

    # Icon detection model files
    icon_detect_files = [
        "icon_detect/train_args.yaml",
        "icon_detect/model.pt",
        "icon_detect/model.yaml",
    ]

    print("Downloading icon detection model...")
    for f in icon_detect_files:
        download_file(repo_id, f, target)

    # Icon caption model files (Florence-2)
    icon_caption_files = [
        "icon_caption/config.json",
        "icon_caption/generation_config.json",
        "icon_caption/model.safetensors",
        "icon_caption/preprocessor_config.json",
        "icon_caption/processing_florence2.py",
        "icon_caption/tokenizer.json",
        "icon_caption/tokenizer_config.json",
    ]

    print("\nDownloading icon caption model...")
    for f in icon_caption_files:
        try:
            download_file(repo_id, f, target)
        except subprocess.CalledProcessError:
            print(f"  Warning: Could not download {f}")

    # Rename icon_caption to icon_caption_florence for clarity
    icon_caption_dir = target / "icon_caption"
    icon_caption_florence_dir = target / "icon_caption_florence"

    if icon_caption_dir.exists() and not icon_caption_florence_dir.exists():
        print(f"\nRenaming {icon_caption_dir} -> {icon_caption_florence_dir}")
        icon_caption_dir.rename(icon_caption_florence_dir)

    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"Weights saved to: {target.absolute()}")
    print()
    print("Directory structure:")
    for item in sorted(target.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(target)
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {rel_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Download OmniParser model weights from HuggingFace"
    )
    parser.add_argument(
        "--target-dir",
        default="weights",
        help="Directory to save weights (default: weights)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if weights exist, don't download",
    )

    args = parser.parse_args()

    # Check for existing weights
    target = Path(args.target_dir)
    detector_path = target / "icon_detect" / "model.pt"
    captioner_path = target / "icon_caption_florence" / "model.safetensors"

    if args.check_only:
        detector_exists = detector_path.exists()
        captioner_exists = captioner_path.exists()

        print(f"Icon detector: {'Found' if detector_exists else 'Missing'}")
        print(f"Icon captioner: {'Found' if captioner_exists else 'Missing'}")

        if detector_exists and captioner_exists:
            print("\nAll weights are present!")
            sys.exit(0)
        else:
            print("\nSome weights are missing. Run without --check-only to download.")
            sys.exit(1)

    # Check huggingface-cli
    if not check_huggingface_cli():
        print("huggingface-cli not found. Installing...")
        install_huggingface_hub()

    # Download weights
    download_omniparser_weights(args.target_dir)


if __name__ == "__main__":
    main()
