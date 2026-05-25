"""
Download the RAID benchmark dataset (train.csv and test.csv) from raid-bench.xyz.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output-dir data/raw
    python scripts/download_data.py --hf    # use Hugging Face datasets instead
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://dataset.raid-bench.xyz"
FILES = {
    "train.csv": f"{BASE_URL}/train.csv",
    "test.csv": f"{BASE_URL}/test.csv",
}

# Known SHA-256 checksums (update if the dataset is re-released)
CHECKSUMS: dict[str, str] = {
    # "train.csv": "<sha256>",
    # "test.csv":  "<sha256>",
}


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} already exists. Use --force to re-download.")
        return

    print(f"  Downloading {dest.name} from {url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
        ncols=80,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))

    if dest.name in CHECKSUMS:
        digest = sha256(dest)
        expected = CHECKSUMS[dest.name]
        if digest != expected:
            dest.unlink()
            raise ValueError(
                f"Checksum mismatch for {dest.name}!\n"
                f"  Expected: {expected}\n"
                f"  Got:      {digest}"
            )
        print(f"  [ok] checksum verified for {dest.name}")

    size_mb = dest.stat().st_size / (1 << 20)
    print(f"  Saved {dest.name} ({size_mb:.1f} MB)")


def download_via_hf(output_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading RAID via Hugging Face datasets library ...")
    raid = load_dataset("liamdugan/raid")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in raid.items():
        out_path = output_dir / f"{split_name}.csv"
        split_data.to_csv(str(out_path), index=False)
        size_mb = out_path.stat().st_size / (1 << 20)
        print(f"  Saved {out_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the RAID benchmark dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save the downloaded files (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Download via Hugging Face datasets library instead of direct HTTP",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}\n")

    if args.hf:
        download_via_hf(output_dir)
    else:
        for filename, url in FILES.items():
            dest = output_dir / filename
            try:
                download_file(url, dest, force=args.force)
            except requests.HTTPError as e:
                print(f"ERROR: HTTP error downloading {filename}: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"ERROR: {e}")
                sys.exit(1)

    print("\nAll files downloaded successfully.")
    print("Next step: open notebooks/01_eda.ipynb to explore the data.")


if __name__ == "__main__":
    main()
