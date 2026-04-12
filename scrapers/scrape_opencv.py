from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

GITHUB_API_LATEST = "https://api.github.com/repos/opencv/opencv/releases/latest"
DEFAULT_OUTPUT = Path("data/raw/opencv")
DEFAULT_TIMEOUT: int = 60
DOWNLOAD_CHUNK = 65536


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "RAG-corpus-builder/1.0",
            "Accept": "application/vnd.github+json",
        }
    )
    return s


def fetch_latest_release(session: requests.Session, timeout: int) -> dict:
    resp = session.get(GITHUB_API_LATEST, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def find_docs_asset(release: dict) -> tuple[str, str]:
    tag = release.get("tag_name", "?")
    for asset in release.get("assets", []):
        name: str = asset.get("name", "")
        if name.endswith("-docs.zip") and name.startswith("opencv"):
            return (asset["browser_download_url"], name)
    raise ValueError(
        f"No *-docs.zip asset found in OpenCV release {tag}.\nAssets present: {[a['name'] for a in release.get('assets', [])]}"
    )


def download_file(
    url: str, dest: Path, session: requests.Session, timeout: int
) -> None:
    resp = session.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        dest.open("wb") as fh,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK):
            fh.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: Path, output_dir: Path) -> int:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, output_dir)
    return len(members)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch the latest stable OpenCV release from GitHub, download the pre-built Doxygen docs zip, and extract it locally."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory for extracted docs (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request HTTP timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    p.add_argument(
        "--keep-zip",
        action="store_true",
        default=False,
        help="Keep the downloaded zip file after extraction",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    print("Querying GitHub API for latest OpenCV release …")
    release = fetch_latest_release(session, args.timeout)
    tag = release["tag_name"]
    print(f"Latest stable release : {tag}")
    download_url, asset_name = find_docs_asset(release)
    print(f"Docs asset            : {asset_name}")
    print(f"Download URL          : {download_url}")
    zip_path = output_dir / asset_name
    if zip_path.exists():
        print(f"Zip already present   : {zip_path}  (skipping download)")
    else:
        print(f"\nDownloading {asset_name} …")
        download_file(download_url, zip_path, session, args.timeout)
        print(f"Saved to {zip_path}")
    marker = output_dir / ".extracted_tag"
    if marker.exists() and marker.read_text().strip() == tag:
        print(f"Already extracted for tag {tag}  (skipping extraction)")
    else:
        print(f"\nExtracting {zip_path} → {output_dir} …")
        n = extract_zip(zip_path, output_dir)
        marker.write_text(tag)
        print(f"Extracted {n} files")
    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"Removed zip file {zip_path.name}")
    html_count = len(list(output_dir.rglob("*.html")))
    print(f"\nDone – {html_count} HTML files available in {output_dir}")


if __name__ == "__main__":
    main()
