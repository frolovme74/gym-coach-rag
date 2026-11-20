"""Нарезка рабочего корпуса с явным различением глубины проверки."""

import hashlib
import json
import re
from collections import Counter

from corpus_prepare import (
    ROOT,
    guard_branch,
    read_json,
    sources,
    word_count,
    write_json,
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sentence_units(paragraph):
    text = paragraph["text_ru"]
    sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9(])", text)
    if any(word_count(s) > 600 for s in sentences):
        return None
    return [
        {
            "text": s,
            "paragraph_id": paragraph["paragraph_id"],
            "section_en": paragraph["section_en"],
            "section_ru": paragraph["section_ru"],
        }
        for s in sentences
        if s.strip()
    ]


def partition(units, minimum=300, maximum=600):
    """Ищет границы предложений без потерь, перекрытий и короткого хвоста."""
    prefix = [0]
    whitespace_prefix = [0]
    for u in units:
        prefix.append(prefix[-1] + word_count(u["text"]))
        whitespace_prefix.append(whitespace_prefix[-1] + len(u["text"].split()))
    n = len(units)
    paths = {0: []}
    scores = {0: 0}
    for end in range(1, n + 1):
        for start in range(end - 1, -1, -1):
            length = prefix[end] - prefix[start]
            whitespace_length = whitespace_prefix[end] - whitespace_prefix[start]
            if length > maximum or whitespace_length > maximum:
                break
            if length < minimum or whitespace_length < minimum or start not in paths:
                continue
            score = scores[start] + (length - 450) ** 2
            if end not in scores or score < scores[end]:
                paths[end] = paths[start] + [(start, end)]
                scores[end] = score
    return [[*units[a:b]] for a, b in paths[n]] if n in paths else None


def merge_short_sections(groups):
    """Соседние разделы одной статьи; не смешивает источники и не теряет хвост.

    Самостоятельные разделы остаются самостоятельными. Короткое введение
    соединяется со следующим разделом, заключение — с предыдущим.
    """
    groups = [dict(g, paragraphs=list(g["paragraphs"])) for g in groups]
    merges = []
    index = 0
    while index < len(groups):
        group = groups[index]
        units = []
        for p in group["paragraphs"]:
            part = sentence_units(p)
            if part is None:
                raise ValueError(f"Нельзя разрезать предложение: {p['paragraph_id']}")
            units.extend(part)
        if partition(units) is not None or len(groups) == 1:
            index += 1
            continue
        # Внутренний короткий блок связан с непосредственно предшествующим
        # контекстом; начальный — с последующим. Названия обоих сохраняются.
        neighbor = index - 1 if index else 1
        first, second = sorted((index, neighbor))
        left, right = groups[first], groups[second]
        merges.append(
            {
                "sections": [left["section_en"], right["section_en"]],
                "reason": "Короткий/неделимый хвост объединён с соседним контекстом той же публикации.",
            }
        )
        groups[first] = {
            "section_en": left["section_en"] + " / " + right["section_en"],
            "section_ru": left["section_ru"] + " / " + right["section_ru"],
            "paragraphs": left["paragraphs"] + right["paragraphs"],
        }
        del groups[second]
        index = first
    return groups, merges


def build():
    guard_branch()
    reviews_path = ROOT / "data/review/chunk_reviews.json"
    reviews = read_json(reviews_path) if reviews_path.exists() else {}
    drafts, manifest, ready, unchunked, missing = [], [], [], [], []
    merged, excluded = [], []
    seen = set()
    for source in sources():
        sid = source["source_id"]
        path = ROOT / "data/processed/ru" / (sid + ".json")
        if not path.exists():
            missing.append(sid)
            continue
        translated = read_json(path)
        groups = []
        for p in translated["paragraphs"]:
            if p.get("excluded_from_corpus") or p.get("flags"):
                excluded.append(
                    {
                        "paragraph_id": p["paragraph_id"],
                        "source_id": sid,
                        "reason": p.get("scope_exclusion_reason")
                        or "Неразобранные замечания",
                        "flags": p.get("flags", []),
                    }
                )
                continue
            parts = p["section_en"].split(" / ")
            # Некоторые JATS помещают предметные главы внутрь Methods.
            key = parts[1] if parts[0] == "Methods" and len(parts) > 1 else parts[0]
            if not groups or groups[-1]["section_en"] != key:
                groups.append(
                    {
                        "section_en": key,
                        "section_ru": p["section_ru"].split(" / ")[0],
                        "paragraphs": [],
                    }
                )
            groups[-1]["paragraphs"].append(p)
        groups, merge_log = merge_short_sections(groups)
        merged.extend({"source_id": sid, **m} for m in merge_log)
        source_index = 0
        for group in groups:
            units = []
            oversized = False
            for p in group["paragraphs"]:
                part = sentence_units(p)
                if part is None:
                    oversized = True
                    break
                units.extend(part)
            chunks = None if oversized else partition(units)
            if not chunks:
                unchunked.append(
                    {
                        "source_id": sid,
                        "section": group["section_en"],
                        "paragraph_ids": [
                            p["paragraph_id"] for p in group["paragraphs"]
                        ],
                        "words_ru": sum(
                            word_count(p["text_ru"]) for p in group["paragraphs"]
                        ),
                        "reason": "Нельзя разделить по предложениям на 300–600 слов без потерь; не склеивать несвязанные разделы",
                    }
                )
                continue
            by_id = {p["paragraph_id"]: p for p in group["paragraphs"]}
            for chunk in chunks:
                source_index += 1
                text = " ".join(u["text"] for u in chunk)
                pids = list(dict.fromkeys(u["paragraph_id"] for u in chunk))
                doc_id = f"doc_{sid.lower().replace('-', '_')}_{source_index:03}"
                sha = digest(text)
                if sha in seen:
                    continue
                seen.add(sha)
                entry = {
                    "doc_id": doc_id,
                    "topic": source["topic"],
                    "source": source["title"],
                    "section": " / ".join(
                        dict.fromkeys(u["section_ru"] for u in chunk)
                    ),
                    "text": text,
                }
                flags = sorted({f for pid in pids for f in by_id[pid]["flags"]})
                review = reviews.get(doc_id, {})
                approved = (
                    review.get("status") == "approved"
                    and review.get("text_sha256") == sha
                    and review.get("source_id") == sid
                    and review.get("original_sha256")
                    == digest("\n".join(by_id[p]["text_en"] for p in pids))
                )
                fully_reviewed = all(
                    by_id[p]["review_status"] == "reviewed" for p in pids
                )
                checked = not flags and all(
                    by_id[p]["review_status"] in ("reviewed", "targeted_checks_passed")
                    for p in pids
                )
                row = {
                    "doc_id": doc_id,
                    "source_id": sid,
                    "source_url": source["url"],
                    "pmcid": source["pmcid"],
                    "doi": source["doi"],
                    "license": source["license"],
                    "paragraph_ids": pids,
                    "sections_en": list(dict.fromkeys(u["section_en"] for u in chunk)),
                    "units": [
                        {
                            "paragraph_id": u["paragraph_id"],
                            "text_sha256": digest(u["text"]),
                        }
                        for u in chunk
                    ],
                    "word_count": word_count(text),
                    "whitespace_word_count": len(text.split()),
                    "text_sha256": sha,
                    "original_sha256": digest(
                        "\n".join(by_id[p]["text_en"] for p in pids)
                    ),
                    "flags": flags,
                    "status": "reviewed"
                    if fully_reviewed or approved
                    else "targeted_checks_passed"
                    if checked
                    else "needs_review",
                    "full_semantic_review": fully_reviewed or approved,
                    "translation": translated["model"],
                    "model_revision": translated["model_revision"],
                }
                drafts.append(entry)
                manifest.append(row)
                if checked or approved:
                    ready.append(entry)
    write_json(ROOT / "data/processed/chunks.draft.json", drafts)
    write_json(ROOT / "data/processed/chunks.manifest.json", manifest)
    write_json(ROOT / "data/review/unchunked.json", unchunked)
    # Пустой проверенный корпус не маскируется под завершённый результат.
    write_json(ROOT / "data/corpus/documents.json", ready)
    write_json(
        ROOT / "data/corpus/manifest.json",
        [m for m in manifest if m["status"] != "needs_review"],
    )
    write_json(ROOT / "data/review/section_merges.json", merged)
    write_json(ROOT / "data/review/build_exclusions.json", excluded)
    absent_sources = sorted(
        {s["source_id"] for s in sources()} - {m["source_id"] for m in manifest}
    )
    indexing_ready = (
        bool(ready)
        and len(ready) == len(drafts)
        and not missing
        and not unchunked
        and not absent_sources
        and not any(e["reason"] == "Неразобранные замечания" for e in excluded)
    )
    summary = {
        "sources_without_chunks": absent_sources,
        "sources_translated": len(sources()) - len(missing),
        "missing_translations": missing,
        "draft_chunks": len(drafts),
        "prepared_chunks": len(ready),
        "reviewed_chunks": sum(m["full_semantic_review"] for m in manifest),
        "draft_by_topic": dict(Counter(d["topic"] for d in drafts)),
        "flag_counts": dict(Counter(f for m in manifest for f in m["flags"])),
        "unchunked_sections": len(unchunked),
        "unchunked_words": sum(x["words_ru"] for x in unchunked),
        "target": {
            "sources_per_topic": [10, 20],
            "chunks": [500, 1500],
            "words_per_chunk": [300, 600],
        },
        "ready_for_indexing": indexing_ready,
        "ready_for_publication": indexing_ready
        and all(m["full_semantic_review"] for m in manifest),
        "excluded_paragraphs": len(excluded),
    }
    write_json(ROOT / "data/review/build_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    build()
