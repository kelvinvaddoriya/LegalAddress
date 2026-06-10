"""Download and parse German statutes from gesetze-im-internet.de.

Each law is published as a zip of XML. We keep every <norm> that has a real
paragraph label (<enbez> like "§ 573c"), flatten its body text, and emit one
"section" record. Long sections are split later in chunk.py.

Run directly to (re)build data/chunks.json:
    python -m app.ingest
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from xml.etree import ElementTree as ET

import httpx

from . import config
from .chunk import split_section

XML_URL = "https://www.gesetze-im-internet.de/{slug}/xml.zip"
HTML_URL = "https://www.gesetze-im-internet.de/{slug}/__{num}.html"

# "§ 573c" / "§ 312g" / "§§ 305" -> (number, letter-suffix)
_ENBEZ_RE = re.compile(r"(\d+)\s*([a-z]*)", re.IGNORECASE)


def _parse_section_no(enbez: str) -> tuple[int, str] | None:
    m = _ENBEZ_RE.search(enbez or "")
    if not m:
        return None
    return int(m.group(1)), m.group(2).lower()


def _in_ranges(no: tuple[int, str], ranges: list[str]) -> bool:
    """True if a section number falls inside any 'start-end' range.

    Empty range list means "include everything".
    """
    if not ranges:
        return True
    for r in ranges:
        if "-" in r:
            lo_s, hi_s = r.split("-", 1)
            lo, hi = _parse_section_no(lo_s), _parse_section_no(hi_s)
            if lo and hi and lo <= no <= hi:
                return True
        else:
            single = _parse_section_no(r)
            if single and single == no:
                return True
    return False


def _flatten_text(elem: ET.Element) -> str:
    """Collect human-readable text from a <Content> body, dropping markup."""
    parts: list[str] = []
    for node in elem.iter():
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        if node.tail and node.tail.strip():
            parts.append(node.tail.strip())
    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def download_law(slug: str) -> bytes:
    """Fetch (and cache) the XML zip for a law slug."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = config.RAW_DIR / f"{slug}.zip"
    if cached.exists():
        return cached.read_bytes()
    url = XML_URL.format(slug=slug)
    resp = httpx.get(url, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    return resp.content


def parse_law(zip_bytes: bytes, entry: dict) -> list[dict]:
    """Turn one law's XML into section records filtered by entry['include']."""
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    xml_name = next(n for n in zf.namelist() if n.endswith(".xml"))
    root = ET.fromstring(zf.read(xml_name))

    sections: list[dict] = []
    for norm in root.findall("norm"):
        enbez_el = norm.find("metadaten/enbez")
        if enbez_el is None or not (enbez_el.text or "").startswith("§"):
            continue  # structural heading or preamble, not a real section
        no = _parse_section_no(enbez_el.text)
        if no is None or not _in_ranges(no, entry["include"]):
            continue

        body = norm.find("textdaten/text/Content")
        text = _flatten_text(body) if body is not None else ""
        if not text:
            continue

        title_el = norm.find("metadaten/titel")
        enbez = enbez_el.text.strip()
        num_str = f"{no[0]}{no[1]}"
        sections.append(
            {
                "law": entry["law"],
                "law_name_de": entry["name_de"],
                "law_name_en": entry["name_en"],
                "domain": entry["domain"],
                "section": enbez,                       # "§ 573c"
                "title": (title_el.text or "").strip() if title_el is not None else "",
                "citation": f"{enbez} {entry['law']}",  # "§ 573c BGB"
                "text": text,
                "lang": "de",
                "source_url": HTML_URL.format(slug=entry["slug"], num=num_str),
            }
        )
    return sections


def build_chunks() -> list[dict]:
    sources = json.loads(config.SOURCES_FILE.read_text(encoding="utf-8"))
    zip_cache: dict[str, bytes] = {}
    chunks: list[dict] = []
    seen_ids: set[str] = set()

    for entry in sources["laws"]:
        slug = entry["slug"]
        if slug not in zip_cache:
            print(f"  downloading {entry['law']} ({slug}) ...")
            zip_cache[slug] = download_law(slug)
        sections = parse_law(zip_cache[slug], entry)
        print(f"  {entry['law']} [{entry['domain']}]: {len(sections)} sections")
        for sec in sections:
            for chunk in split_section(sec):
                if chunk["id"] in seen_ids:
                    continue  # same § may appear under two domains; keep first
                seen_ids.add(chunk["id"])
                chunks.append(chunk)
    return chunks


def main() -> None:
    chunks = build_chunks()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(chunks)} chunks -> {config.CHUNKS_FILE}")


if __name__ == "__main__":
    main()
