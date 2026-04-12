from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

DEFAULT_INPUT = Path("data/raw/open3d")
DEFAULT_OUTPUT = Path("data/parsed/open3d_chunks.json")
MIN_TEXT_LENGTH = 40
_API_PATTERN = re.compile("open3d[.\\-]", re.IGNORECASE)


def _dl_entry_type(dl: Tag) -> str:
    classes = dl.get("class") or []
    if len(classes) >= 2 and classes[0] == "py":
        return classes[1]
    return ""


def _extract_signature(dt: Tag) -> str:
    return " ".join(dt.get_text(" ", strip=True).split())


def _extract_description(dd: Tag) -> str:
    parts: list[str] = []
    for child in dd.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "dl":
            break
        if child.name in {"p", "div", "ul", "ol", "pre", "blockquote"}:
            text = child.get_text(" ", strip=True)
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def parse_html_file(html_path: Path, input_dir: Path) -> list[dict]:
    try:
        soup = BeautifulSoup(
            html_path.read_text(encoding="utf-8", errors="replace"), "lxml"
        )
    except Exception:
        return []
    rel_file = str(html_path.relative_to(input_dir))
    chunks: list[dict] = []
    for dl in soup.find_all("dl"):
        entry_type = _dl_entry_type(dl)
        if not entry_type:
            continue
        dt = dl.find("dt", recursive=False)
        dd = dl.find("dd", recursive=False)
        if dt is None or dd is None:
            continue
        qualified_name = (dt.get("id") or "").strip()
        if not qualified_name:
            qualified_name = dt.get_text(" ", strip=True).split("(")[0].strip()
        signature = _extract_signature(dt)
        description = _extract_description(dd)
        dense_parts = [f"[Open3D] Name: {qualified_name}"]
        if signature:
            dense_parts.append(f"Signature: {signature}")
        if description:
            dense_parts.append(description)
        dense_embedding_text = "\n\n".join(dense_parts).strip()
        if len(dense_embedding_text) < MIN_TEXT_LENGTH:
            continue
        chunks.append(
            {
                "entity_name": qualified_name,
                "signature": signature,
                "description": description,
                "dense_embedding_text": dense_embedding_text,
                "metadata": {
                    "source": "open3d",
                    "entry_type": entry_type,
                    "file": rel_file,
                },
            }
        )
    return chunks


def collect_html_files(input_dir: Path) -> list[Path]:
    all_html = sorted(input_dir.rglob("*.html"))
    api_files = [
        p for p in all_html if "python_api" in p.parts or _API_PATTERN.search(p.name)
    ]
    return api_files if api_files else all_html


def parse_all(input_dir: Path, output_file: Path) -> int:
    html_files = collect_html_files(input_dir)
    if not html_files:
        print(f"[WARN] No HTML files found in {input_dir}")
        print("       Run scrapers/scrape_open3d.py first.")
        return 0
    print(f"Found {len(html_files)} HTML files to parse in {input_dir}")
    all_chunks: list[dict] = []
    for html_path in tqdm(html_files, desc="Parsing Open3D HTML", unit="file"):
        all_chunks.extend(parse_html_file(html_path, input_dir))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_chunks)} chunks to {output_file}")
    return len(all_chunks)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse Open3D Sphinx HTML docs into JSON chunks."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory containing extracted HTML docs (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    n = parse_all(input_dir=args.input, output_file=args.output)
    print(f"\nDone – {n} chunks saved to {args.output}")


if __name__ == "__main__":
    main()
