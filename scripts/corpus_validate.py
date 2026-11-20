"""Проверка корпуса и предобработки; это не benchmark ранжирования."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag.preprocess import preprocess
from corpus_build import digest, sentence_units
from corpus_prepare import ROOT, read_json, sources, word_count, write_json


def validate():
    docs = read_json(ROOT / "data/corpus/documents.json")
    manifest = read_json(ROOT / "data/corpus/manifest.json")
    source_map = {s["source_id"]: s for s in sources()}
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    check(len(docs) == len(manifest), "Размеры корпуса и manifest различаются")
    check(len({d["doc_id"] for d in docs}) == len(docs), "Повтор doc_id")
    check(len({digest(d["text"]) for d in docs}) == len(docs), "Дублирующиеся чанки")
    by_id = {m["doc_id"]: m for m in manifest}
    expected_units, actual_units = Counter(), Counter()
    paragraphs = {}
    included, excluded = [], []
    for sid in source_map:
        data = read_json(ROOT / "data/processed/ru" / f"{sid}.json")
        original = read_json(ROOT / "data/processed/en" / f"{sid}.json")
        original_map = {p["paragraph_id"]: p for p in original["paragraphs"]}
        check(
            list(original_map) == [p["paragraph_id"] for p in data["paragraphs"]],
            f"Порядок абзацев {sid}",
        )
        for p in data["paragraphs"]:
            pid = p["paragraph_id"]
            paragraphs[pid] = p
            check(
                p["text_en"] == original_map[pid]["text_en"], f"Изменён оригинал {pid}"
            )
            check(
                " ".join(s["en"] for s in p["segments"]).split()
                == p["text_en"].split(),
                f"Потеря входа перевода {pid}",
            )
            if p.get("excluded_from_corpus"):
                excluded.append(pid)
                continue
            check(not p["flags"], f"Остались замечания {pid}")
            included.append(pid)
            units = sentence_units(p)
            check(units is not None, f"Слишком длинное предложение {pid}")
            for unit in units or []:
                expected_units[(sid, pid, digest(unit["text"]))] += 1
    token_counts = []
    for doc in docs:
        did = doc["doc_id"]
        m = by_id.get(did)
        check(
            set(doc) == {"doc_id", "topic", "source", "section", "text"}, f"Схема {did}"
        )
        if m is None:
            errors.append(f"Нет manifest {did}")
            continue
        sid = m["source_id"]
        source = source_map[sid]
        check(doc["topic"] == source["topic"], f"Область {did}")
        check(doc["source"] == source["title"], f"Название источника {did}")
        check(
            m["license"] == source["license"] and m["source_url"] == source["url"],
            f"Атрибуция {did}",
        )
        check(300 <= word_count(doc["text"]) <= 600, f"Лексические слова {did}")
        check(300 <= len(doc["text"].split()) <= 600, f"Слова по пробелам {did}")
        check(m["text_sha256"] == digest(doc["text"]), f"Хеш чанка {did}")
        check(
            m["original_sha256"]
            == digest("\n".join(paragraphs[p]["text_en"] for p in m["paragraph_ids"])),
            f"Хеш оригиналов {did}",
        )
        check(
            all(p.startswith(sid + ":") for p in m["paragraph_ids"]),
            f"Смешаны статьи {did}",
        )
        check(
            not set(m["paragraph_ids"]).intersection(excluded),
            f"Исключённый абзац попал в чанк {did}",
        )
        check(
            not m["flags"] and m["status"] in ("reviewed", "targeted_checks_passed"),
            f"Статус {did}",
        )
        for unit in m["units"]:
            actual_units[(sid, unit["paragraph_id"], unit["text_sha256"])] += 1
        # Независимое сравнение текста с последовательностью исходных единиц.
        pools = {pid: sentence_units(paragraphs[pid]) for pid in m["paragraph_ids"]}
        rebuilt = []
        for unit in m["units"]:
            candidates = [
                u["text"]
                for u in pools[unit["paragraph_id"]]
                if digest(u["text"]) == unit["text_sha256"]
            ]
            check(bool(candidates), f"Нет исходного предложения {did}")
            if candidates:
                rebuilt.append(candidates[0])
        check(" ".join(rebuilt) == doc["text"], f"Изменён текст при нарезке {did}")
        tokens = preprocess(doc["text"])
        check(bool(tokens), f"Пустой результат предобработки {did}")
        check(
            tokens == preprocess(doc["text"]),
            f"Недетерминированная предобработка {did}",
        )
        token_counts.append(len(tokens))
    check(expected_units == actual_units, "Потеря или повтор предложений при нарезке")
    check(
        set(m["source_id"] for m in manifest) == set(source_map),
        "Не все 56 источников представлены",
    )
    check(
        not read_json(ROOT / "data/review/unchunked.json"),
        "Остались неразрезанные разделы",
    )
    check(
        not read_json(ROOT / "data/review/paragraph_review_queue.json"),
        "Очередь не закрыта",
    )
    initial = read_json(ROOT / "data/review/unchunked.initial.json")
    short_pids = {p for g in initial for p in g["paragraph_ids"]}
    represented = {u[1] for u in actual_units}
    check(short_pids <= represented | set(excluded), "Потеря коротких разделов")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "sources": len(source_map),
        "chunks": len(docs),
        "by_topic": dict(Counter(d["topic"] for d in docs)),
        "source_counts_by_topic": dict(Counter(s["topic"] for s in sources())),
        "included_paragraphs": len(included),
        "excluded_paragraphs": len(excluded),
        "previously_unchunked_sections": len(initial),
        "previously_unchunked_paragraphs_in_corpus": len(short_pids & represented),
        "previously_unchunked_paragraphs_excluded_with_reason": len(
            short_pids & set(excluded)
        ),
        "unaccounted_short_paragraphs": sorted(
            short_pids - represented - set(excluded)
        ),
        "sentence_units": sum(actual_units.values()),
        "words_min": min(word_count(d["text"]) for d in docs),
        "words_max": max(word_count(d["text"]) for d in docs),
        "preprocessed_tokens": sum(token_counts),
        "corpus_sha256": digest(json.dumps(docs, ensure_ascii=False, sort_keys=True)),
        "full_semantic_review": False,
        "retrieval_benchmark": "not_run: BM25 and relevance judgments are not implemented yet",
    }
    write_json(ROOT / "data/review/corpus_validation.json", report)
    print(json.dumps(report, ensure_ascii=False))
    if errors:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    validate()
