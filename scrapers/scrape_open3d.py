from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

GITHUB_API_RELEASES = "https://api.github.com/repos/isl-org/Open3D/releases"
DEFAULT_OUTPUT = Path("data/raw/open3d")
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


def fetch_latest_stable_release(session: requests.Session, timeout: int) -> dict:
    for page in range(1, 4):
        resp = session.get(
            GITHUB_API_RELEASES, params={"per_page": 10, "page": page}, timeout=timeout
        )
        resp.raise_for_status()
        releases = resp.json()
        if not releases:
            break
        for rel in releases:
            if not rel.get("prerelease") and (not rel.get("draft")):
                return rel
    raise RuntimeError("Could not find a stable Open3D release on GitHub.")


def find_docs_asset(release: dict) -> tuple[str, str]:
    tag = release.get("tag_name", "?")
    for asset in release.get("assets", []):
        name: str = asset.get("name", "")
        if name.endswith("-docs.tar.gz") and name.startswith("open3d"):
            return (asset["browser_download_url"], name)
    raise ValueError(
        f"No *-docs.tar.gz asset found in Open3D release {tag}.\nAssets present: {[a['name'] for a in release.get('assets', [])]}"
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


def extract_tarball(tarball: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    output_resolved = output_dir.resolve()
    with tarfile.open(tarball, "r:gz") as tf:
        members = tf.getmembers()
        top_dirs = {m.name.split("/")[0] for m in members if "/" in m.name}
        top = next(iter(top_dirs)) + "/" if len(top_dirs) == 1 else ""
        for member in tqdm(members, desc="Extracting", unit="file"):
            if not member.isfile():
                continue
            rel = (
                member.name[len(top) :]
                if top and member.name.startswith(top)
                else member.name
            )
            if not rel:
                continue
            dest = output_dir / rel
            try:
                dest.resolve().relative_to(output_resolved)
            except ValueError:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src:
                dest.write_bytes(src.read())
            extracted += 1
    return extracted


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch the latest stable Open3D release from GitHub, download the pre-built docs tarball, and extract it locally."
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
        "--keep-tarball",
        action="store_true",
        default=False,
        help="Keep the downloaded tarball after extraction",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    print("Querying GitHub API for latest stable Open3D release …")
    release = fetch_latest_stable_release(session, args.timeout)
    tag = release["tag_name"]
    print(f"Latest stable release : {tag}")
    download_url, asset_name = find_docs_asset(release)
    print(f"Docs asset            : {asset_name}")
    print(f"Download URL          : {download_url}")
    tarball_path = output_dir / asset_name
    if tarball_path.exists():
        print(f"Tarball already present : {tarball_path}  (skipping download)")
    else:
        print(f"\nDownloading {asset_name} …")
        download_file(download_url, tarball_path, session, args.timeout)
        print(f"Saved to {tarball_path}")
    marker = output_dir / ".extracted_tag"
    if marker.exists() and marker.read_text().strip() == tag:
        print(f"Already extracted for tag {tag}  (skipping extraction)")
    else:
        print(f"\nExtracting {tarball_path.name} → {output_dir} …")
        n = extract_tarball(tarball_path, output_dir)
        marker.write_text(tag)
        print(f"Extracted {n} files")
    if not args.keep_tarball and tarball_path.exists():
        tarball_path.unlink()
        print(f"Removed tarball {tarball_path.name}")
    html_count = len(list(output_dir.rglob("*.html")))
    print(f"\nDone – {html_count} HTML files available in {output_dir}")

if __name__ == "__main__":
    main()
