from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "https://github.com/PointCloudLibrary/documentation"
DEFAULT_OUTPUT = Path("data/raw/pcl")


def git_available() -> bool:
    result = subprocess.run(["git", "--version"], capture_output=True)
    return result.returncode == 0


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, **kwargs)
    if result.returncode != 0:
        print(f"[ERROR] Command exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def download_pcl_docs(repo_url: str, output_dir: Path) -> None:
    if not git_available():
        print("[ERROR] git is not installed or not on PATH.", file=sys.stderr)
        sys.exit(1)
    git_dir = output_dir / ".git"
    if git_dir.is_dir():
        print(f"Repository found at {output_dir}. Pulling latest changes …")
        run(["git", "-C", str(output_dir), "pull", "--ff-only"])
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {repo_url} → {output_dir} (shallow, depth=1) …")
        run(["git", "clone", "--depth=1", repo_url, str(output_dir)])
    html_files = list(output_dir.rglob("*.html"))
    print(f"\nDone – {len(html_files)} HTML files available in {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clone/update the PCL pre-built Doxygen HTML documentation."
    )
    p.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Git repository URL (default: {DEFAULT_REPO})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Local destination directory (default: {DEFAULT_OUTPUT})",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"Repo   : {args.repo}")
    print(f"Output : {args.output}")
    download_pcl_docs(repo_url=args.repo, output_dir=args.output)


if __name__ == "__main__":
    main()
