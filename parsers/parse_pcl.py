from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

DEFAULT_INPUT = Path("data/raw/pcl")
DEFAULT_OUTPUT = Path("data/parsed/pcl_chunks.json")
MIN_TEXT_LENGTH = 30
_API_FILENAME_RE = re.compile(
    "^(modules|group__\\w+|classpcl\\w*|structpcl\\w*|namespacepcl\\w*|class\\w+|struct\\w+|namespace\\w+)\\.html$",
    re.IGNORECASE,
)
_SKIP_FILENAME_RE = re.compile(
    "^(index|annotated|functions|hierarchy|inherits|namespacemembers|globals|files|dirs|pages|todo|deprecated|search|navtree|resize|splitbar)\\b",
    re.IGNORECASE,
)


def clean_text(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines()]
    result: list[str] = []
    prev_blank = False
    for line in lines:
        if line == "":
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    return "\n".join(result).strip()


def tag_text(tag: Tag, separator: str = " ") -> str:
    return clean_text(tag.get_text(separator=separator))


def extract_page_title(soup: BeautifulSoup) -> str:
    title_div = soup.find("div", class_="title")
    if title_div:
        return tag_text(title_div)
    h1 = soup.find("h1")
    return tag_text(h1) if h1 else ""


def extract_breadcrumb(soup: BeautifulSoup) -> str:
    nav = soup.find("div", id="nav-path")
    if not nav:
        return ""
    parts = [tag_text(a) for a in nav.find_all("a")]
    return " > ".join((p for p in parts if p))


def extract_signature(memproto: Tag) -> str:
    if not memproto:
        return ""
    for ttc in memproto.find_all("div", class_="ttc"):
        ttc.decompose()
    return clean_text(memproto.get_text(separator=" "))


def extract_parameters(memdoc: Tag) -> str:
    param_table = memdoc.find("table", class_="params")
    if not param_table:
        return ""
    lines: list[str] = []
    for row in param_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            name = clean_text(cells[0].get_text())
            desc = clean_text(cells[-1].get_text())
            if name or desc:
                lines.append(f"  {name}: {desc}")
    return "Parameters:\n" + "\n".join(lines) if lines else ""


def extract_return_value(memdoc: Tag) -> str:
    retval = memdoc.find("table", class_="retval")
    if not retval:
        return ""
    return "Returns: " + clean_text(retval.get_text(separator=" "))


def extract_template_params(memdoc: Tag) -> str:
    tmpl_table = memdoc.find("table", class_="tparams")
    if not tmpl_table:
        return ""
    lines: list[str] = []
    for row in tmpl_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            name = clean_text(cells[0].get_text())
            desc = clean_text(cells[-1].get_text())
            if name or desc:
                lines.append(f"  {name}: {desc}")
    return "Template parameters:\n" + "\n".join(lines) if lines else ""


def extract_entity_name(memitem: Tag) -> str:
    memname_td = memitem.find("td", class_="memname")
    if memname_td:
        return tag_text(memname_td)
    memtitle_h2 = memitem.find("h2", class_="memtitle")
    if memtitle_h2:
        return tag_text(memtitle_h2).replace("◆", "").strip()
    return ""


def parse_memitem(
    memitem: Tag, page_title: str, breadcrumb: str, source_file: str
) -> dict | None:
    memproto = memitem.find("div", class_="memproto")
    memdoc = memitem.find("div", class_="memdoc")
    if not memdoc:
        return None
    entity_name = extract_entity_name(memitem)
    signature = extract_signature(memproto) if memproto else ""
    description_parts: list[str] = []
    for p in memdoc.find_all("p", recursive=False):
        txt = tag_text(p)
        if txt:
            description_parts.append(txt)
    if not description_parts:
        raw = tag_text(memdoc)
        if raw:
            description_parts.append(raw)
    description = "\n\n".join(description_parts)
    tparams = extract_template_params(memdoc)
    params = extract_parameters(memdoc)
    returns = extract_return_value(memdoc)
    dense_parts: list[str] = []
    if page_title:
        dense_parts.append(f"[PCL] {page_title}")
    if breadcrumb:
        dense_parts.append(f"Module: {breadcrumb}")
    if entity_name:
        dense_parts.append(f"Name: {entity_name}")
    if signature:
        dense_parts.append(f"Signature: {signature}")
    if description:
        dense_parts.append(description)
    if tparams:
        dense_parts.append(tparams)
    if params:
        dense_parts.append(params)
    if returns:
        dense_parts.append(returns)
    dense_embedding_text = "\n\n".join(dense_parts).strip()
    if len(dense_embedding_text) < MIN_TEXT_LENGTH:
        return None
    return {
        "entity_name": entity_name,
        "signature": signature,
        "description": description,
        "template_parameters": tparams,
        "parameters": params,
        "returns": returns,
        "dense_embedding_text": dense_embedding_text,
        "metadata": {
            "source": "pcl",
            "page_title": page_title,
            "breadcrumb": breadcrumb,
            "file": source_file,
        },
    }


def parse_page_as_single_chunk(
    soup: BeautifulSoup, page_title: str, breadcrumb: str, source_file: str
) -> dict | None:
    content = soup.find("div", class_="textblock")
    if not content:
        content = soup.find("div", class_="contents")
    if not content:
        return None
    text = clean_text(content.get_text(separator="\n"))
    if len(text) < MIN_TEXT_LENGTH:
        return None
    header = f"[PCL] {page_title}" if page_title else "[PCL]"
    return {
        "text": f"{header}\n\n{text}",
        "metadata": {
            "source": "pcl",
            "page_title": page_title,
            "breadcrumb": breadcrumb,
            "file": source_file,
            "signature": "",
        },
    }


def parse_html_file(html_path: Path) -> list[dict]:
    try:
        content = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    soup = BeautifulSoup(content, "lxml")
    source_file = html_path.name
    page_title = extract_page_title(soup)
    breadcrumb = extract_breadcrumb(soup)
    chunks: list[dict] = []
    memitems = soup.find_all("div", class_="memitem")
    for memitem in memitems:
        chunk = parse_memitem(memitem, page_title, breadcrumb, source_file)
        if chunk:
            chunks.append(chunk)
    if not chunks:
        chunk = parse_page_as_single_chunk(soup, page_title, breadcrumb, source_file)
        if chunk:
            chunks.append(chunk)
    return chunks


def collect_html_files(input_dir: Path) -> list[Path]:
    result: list[Path] = []
    for p in input_dir.rglob("*.html"):
        if _SKIP_FILENAME_RE.match(p.name):
            continue
        if _API_FILENAME_RE.match(p.name):
            result.append(p)
    return result


def parse_all(input_dir: Path, output_file: Path) -> int:
    html_files = collect_html_files(input_dir)
    if not html_files:
        print(f"[WARN] No API HTML files found in {input_dir}")
        return 0
    print(f"Found {len(html_files)} HTML files in {input_dir}")
    all_chunks: list[dict] = []
    for html_path in tqdm(html_files, desc="Parsing PCL pages", unit="file"):
        all_chunks.extend(parse_html_file(html_path))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_chunks)} chunks to {output_file}")
    return len(all_chunks)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse PCL Doxygen HTML files into JSON chunks."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory containing PCL Doxygen HTML files (default: {DEFAULT_INPUT})",
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
