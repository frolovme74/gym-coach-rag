"""Загрузка и очистка оригиналов корпуса. Без поискового индекса и LLM API."""

import argparse
import hashlib
import html
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ("training", "nutrition", "supplements", "recovery")
ALLOWED = {"CC BY", "CC BY-NC", "CC BY-SA"}
WORD = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*", re.UNICODE)
NOISE = re.compile(
    r"^(references|bibliography|acknowledg|funding|author|competing|conflicts? of interest|disclosure|ethics|consent|data availability|availability of data|supplementary|supporting information|abbreviations|publisher)",
    re.I,
)
OUT_OF_SCOPE = re.compile(
    r"\b(steroid\w*|phytoecdysteroid\w*|ecdysteron\w*|ecdysteroid\w*|turkesteron\w*|prohormone\w*|DHEA|androstenedione|doping|anabolic androgenic|prescription|pharmacolog\w*|pregnan\w*|cancer\w*|diabet\w*|patients?|pediatric\w*|children|adolescen\w*|renal disease|kidney disease|heart failure|clinical treatment|rehabilitat\w*|eating disorders?|anorexia|bulimia|weight cutting|rapid weight loss|starvation|ketogenic|intermittent fasting|VLED|VLCD|very.low.(?:energy|calorie).diet|testosterone|growth hormone)\b",
    re.I,
)
EXCLUDED_SECTIONS = re.compile(
    r"\b(legality|anti-doping|nasal sprays|inspired powders|clinical applications|therapeutic|medical uses|disease|disorders?|pharmacokinetics)\b",
    re.I,
)
SEARCH_METHOD = re.compile(
    r"^(?:\d+[. ]*)?(?:search strategy|literature search|database search|data extraction|data collection(?: process)?|statistical analys(?:is|es)|protocol and registration|registration and protocol|study selection|information sources|search and selection)(?:\s|$)",
    re.I,
)


def guard_branch():
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "2-rag-benchmark-Murad":
        raise SystemExit("Изменения разрешены только в 2-rag-benchmark-Murad")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    guard_branch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def sources():
    return read_json(ROOT / "data/sources/registry.json")["sources"]


def word_count(text):
    return len(WORD.findall(text))


def download_bytes(url, destination, expected_md5=None):
    destination = Path(destination)
    if destination.exists():
        data = destination.read_bytes()
        if not expected_md5 or hashlib.md5(data).hexdigest() == expected_md5:
            return data
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "GymCoachRAG-university-corpus/1.0"}
            )
            with urllib.request.urlopen(req, timeout=45) as response:
                data = response.read()
            if expected_md5 and hashlib.md5(data).hexdigest() != expected_md5:
                raise ValueError("Несовпадение MD5 оригинала")
            guard_branch()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fetch_source(source):
    sid = source["source_id"]
    folder = ROOT / "data/raw" / sid
    try:
        url = source["pmc_metadata"]["metadata_url"]
        meta = json.loads(download_bytes(url, folder / "metadata.json"))
        if meta.get("is_retracted") or meta.get("license_code") not in ALLOWED:
            raise ValueError("Материал отозван или лицензия не разрешает адаптацию")
        if meta["pmcid"] != source["pmcid"]:
            raise ValueError("Несовпадение идентификатора источника")
        files = {}
        for kind in ("xml", "pdf"):
            s3 = urllib.parse.urlparse(meta[kind + "_url"])
            url = "https://" + s3.netloc + ".s3.amazonaws.com" + s3.path
            md5 = urllib.parse.parse_qs(s3.query).get("md5", [None])[0]
            data = download_bytes(url, folder / ("article." + kind), md5)
            if kind == "pdf" and not data.startswith(b"%PDF-"):
                raise ValueError("Ответ не является PDF")
            if kind == "xml":
                ET.fromstring(data)
            files[kind] = {
                "url": url,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        return {"source_id": sid, "status": "downloaded", "files": files}
    except Exception as exc:
        return {"source_id": sid, "status": "failed", "error": str(exc)}


def inline_text(node):
    """Сохраняет tail после ссылок, но не библиографические номера и рисунки."""
    if node.tag in {"xref", "sup"} and (
        node.get("ref-type") == "bibr"
        or node.tag == "sup"
        and not "".join(node.itertext()).strip()
    ):
        return ""
    if node.tag in {
        "table-wrap",
        "fig",
        "ref-list",
        "fn-group",
        "supplementary-material",
    }:
        return ""
    parts = [node.text or ""]
    for child in node:
        value = inline_text(child)
        if child.tag in {"p", "list", "list-item", "title", "break", "label"}:
            value = " " + value + " "
        parts.extend([value, child.tail or ""])
    return "".join(parts)


def clean_text(text):
    text = re.sub(r"\[\s*[,;–—\-\s]*\]", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_article(xml_bytes, source_id):
    root = ET.fromstring(xml_bytes)
    body = root.find("body")
    if body is None:
        raise ValueError("JATS body отсутствует")
    parents = {child: parent for parent in root.iter() for child in parent}
    paragraphs, excluded = [], []
    seen = set()
    for index, paragraph in enumerate(body.iter("p"), 1):
        ancestors = []
        node = parents.get(paragraph)
        while node is not None and node is not body:
            ancestors.append(node)
            node = parents.get(node)
        if any(
            a.tag
            in {"p", "table-wrap", "fig", "fn", "ref-list", "supplementary-material"}
            for a in ancestors
        ):
            continue
        titles = [
            clean_text(inline_text(a.find("title")))
            for a in reversed(ancestors)
            if a.tag == "sec" and a.find("title") is not None
        ]
        section = " / ".join(titles) or "Main text"
        text = clean_text(inline_text(paragraph))
        pid = f"{source_id}:p{index:04}"
        reason = None
        if any(NOISE.search(t) for t in titles):
            reason = "служебный раздел"
        elif any(SEARCH_METHOD.search(t) for t in titles):
            reason = "процедура поиска и статистической обработки обзора; не практическое содержание"
        elif any(
            "math" in n.tag.lower() or n.tag in {"disp-formula", "inline-formula"}
            for n in paragraph.iter()
        ):
            reason = "формула требует отдельного представления; не уплощать математическую разметку"
        elif (
            EXCLUDED_SECTIONS.search(section)
            or OUT_OF_SCOPE.search(section)
            or OUT_OF_SCOPE.search(text)
        ):
            reason = "требует проверки тематических границ"
        elif word_count(text) < 5:
            reason = "короткая служебная строка"
        elif text in seen:
            reason = "точный повтор абзаца"
        if reason:
            excluded.append(
                {
                    "paragraph_id": pid,
                    "section_en": section,
                    "text_en": text,
                    "reason": reason,
                }
            )
            continue
        seen.add(text)
        paragraphs.append({"paragraph_id": pid, "section_en": section, "text_en": text})
    permissions = root.find("front/article-meta/permissions")
    notice = clean_text(inline_text(permissions)) if permissions is not None else ""
    links = (
        [
            n.get("{http://www.w3.org/1999/xlink}href")
            for n in permissions.iter()
            if n.get("{http://www.w3.org/1999/xlink}href")
        ]
        if permissions is not None
        else []
    )
    links.extend(
        re.findall(
            r"https?://creativecommons\.org/(?:licenses|publicdomain)/[a-z0-9/.-]+",
            notice,
        )
    )
    return {
        "source_id": source_id,
        "paragraphs": paragraphs,
        "excluded": excluded,
        "copyright_notice": notice,
        "license_links": list(dict.fromkeys(links)),
        "word_count_en": sum(word_count(p["text_en"]) for p in paragraphs),
    }


def extract_all():
    report = []
    registry = read_json(ROOT / "data/sources/registry.json")
    for source in registry["sources"]:
        sid = source["source_id"]
        path = ROOT / "data/raw" / sid / "article.xml"
        if not path.exists():
            report.append({"source_id": sid, "status": "missing_original"})
            continue
        data = extract_article(path.read_bytes(), sid)
        source["raw_status"] = "downloaded"
        source["original_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        source["license_links"] = data["license_links"]
        source["copyright_notice"] = data["copyright_notice"]
        source["cleaned_words_en"] = data["word_count_en"]
        write_json(ROOT / "data/processed/en" / (sid + ".json"), data)
        blocks = [
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>'
            + html.escape(source["title"])
            + "</title><body>",
            "<h1>" + html.escape(source["title"]) + "</h1>",
            "<p>Cleaned extraction from JATS; see original PDF/XML and source registry.</p>",
        ]
        previous = None
        for p in data["paragraphs"]:
            if p["section_en"] != previous:
                blocks.append("<h2>" + html.escape(p["section_en"]) + "</h2>")
                previous = p["section_en"]
            blocks.append(
                '<p id="'
                + p["paragraph_id"]
                + '">'
                + html.escape(p["text_en"])
                + "</p>"
            )
        blocks.append("</body></html>")
        guard_branch()
        (path.parent / "cleaned.html").write_text("\n".join(blocks), encoding="utf-8")
        report.append(
            {
                "source_id": sid,
                "topic": source["topic"],
                "status": "extracted",
                "paragraphs": len(data["paragraphs"]),
                "excluded_paragraphs": len(data["excluded"]),
                "words_en": data["word_count_en"],
            }
        )
    write_json(ROOT / "data/review/extraction_report.json", report)
    write_json(ROOT / "data/sources/registry.json", registry)
    print(
        json.dumps(
            {
                "sources": len(report),
                "words_en": sum(r.get("words_en", 0) for r in report),
                "paragraphs": sum(r.get("paragraphs", 0) for r in report),
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "extract"])
    args = parser.parse_args()
    guard_branch()
    if args.command == "download":
        result = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for item in pool.map(fetch_source, sources()):
                result.append(item)
                print(
                    item["source_id"], item["status"], item.get("error", ""), flush=True
                )
        write_json(ROOT / "data/review/download_report.json", result)
        if any(r["status"] != "downloaded" for r in result):
            raise SystemExit(1)
    else:
        extract_all()


if __name__ == "__main__":
    main()
