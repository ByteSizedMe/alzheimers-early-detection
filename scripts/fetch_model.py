"""Download the trained checkpoint from the GitHub release.

    python scripts/fetch_model.py

The checkpoint is 126 MB, over GitHub's 100 MB per-file limit, so it ships as a
release asset rather than living in the repository. This pulls it into
``artifacts/`` where the API and dashboard expect to find it.

Prefer training your own if you have changed anything about the data or the
model — ``make all`` takes a few minutes on CPU and produces a checkpoint that
actually matches your code. This exists so someone cloning the repo can see the
interface working immediately.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://github.com/Harit117/alzheimers-early-detection"
    "/releases/download/v0.1.0/best_model.pt"
)
EXPECTED_MB = 126


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def progress(count: int, block: int, total: int) -> None:
        if total <= 0:
            return
        done = min(count * block, total)
        pct = done / total
        bar = "#" * int(pct * 34)
        sys.stdout.write(
            f"\r  [{bar:<34}] {pct:5.1%}  {done / 1e6:5.1f} / {total / 1e6:.1f} MB"
        )
        sys.stdout.flush()

    print(f"Downloading checkpoint (~{EXPECTED_MB} MB)")
    print(f"  from {url}")
    try:
        urllib.request.urlretrieve(url, destination, reporthook=progress)
    except Exception as exc:  # noqa: BLE001 - the message is the whole point
        print()
        raise SystemExit(
            f"\nDownload failed: {exc}\n\n"
            "If the release does not exist yet, train your own instead:\n"
            "  python scripts/make_synthetic_data.py --n-per-class 150\n"
            "  python -m alzheimers.data.prepare\n"
            "  python -m alzheimers.training.train"
        ) from exc

    print(f"\n\nSaved to {destination}  ({destination.stat().st_size / 1e6:.1f} MB)")
    print("Now run:  uvicorn alzheimers.api.main:app --port 8000")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the trained checkpoint")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="artifacts/best_model.pt")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()

    destination = Path(args.out)
    if not destination.is_absolute():
        destination = REPO_ROOT / destination

    if destination.exists() and not args.force:
        size = destination.stat().st_size / 1e6
        print(f"Checkpoint already present: {destination} ({size:.1f} MB)")
        print("Pass --force to download it again.")
        return

    download(args.url, destination)


if __name__ == "__main__":
    main()
