"""Применение редакторских решений без запуска модели перевода.

Оригинал и машинный перевод неизменны. Полная ручная сверка абзаца,
сверка отдельных предложений и детерминированная правка различаются.
"""

from collections import Counter
from corpus_prepare import ROOT, guard_branch, read_json, sources, write_json
from corpus_build import digest
from corpus_quality import quality_flags
from corpus_review import terminology_flags
from corpus_editorial import edit_terms, method_section


def checked_record(record, en, machine=None):
    expected = record.get("original_sha256", record.get("en_sha256"))
    if expected != digest(en):
        raise ValueError("Редакторская правка относится к другому английскому тексту")
    if machine is not None and record.get("machine_ru_sha256") != digest(machine):
        raise ValueError("Машинный перевод изменился после редакторской проверки")


def curate():
    guard_branch()
    whole = read_json(ROOT / "data/review/editorial_edits.json")
    segments = read_json(ROOT / "data/review/segment_edits.json")
    resolved = read_json(ROOT / "data/review/flag_resolutions.json")
    scope = read_json(ROOT / "data/review/editorial_scope_exclusions.json")
    headings = read_json(ROOT / "data/review/section_edits.json")
    decisions, queue, coverage, exclusions = [], [], [], []
    for source in sources():
        sid = source["source_id"]
        path = ROOT / "data/processed/ru" / f"{sid}.json"
        data = read_json(path)
        originals = {
            p["paragraph_id"]: p
            for p in read_json(ROOT / "data/processed/en" / f"{sid}.json")["paragraphs"]
        }
        if set(originals) != {p["paragraph_id"] for p in data["paragraphs"]}:
            raise ValueError(f"Нарушено соответствие абзацев {sid}")
        for p in data["paragraphs"]:
            pid = p["paragraph_id"]
            if originals[pid]["text_en"] != p["text_en"]:
                raise ValueError(f"Изменён оригинал {pid}")
            if " ".join(s["en"] for s in p["segments"]).split() != p["text_en"].split():
                raise ValueError(f"Пропущен английский текст {pid}")
            full = whole.get(pid)
            issues, texts, manual_count, rule_count = [], [], 0, 0
            for i, s in enumerate(p["segments"]):
                key = f"{pid}:{i}"
                edit = segments.get(key)
                if edit:
                    checked_record(edit, s["en"], s["ru"])
                    if edit["status"] != "reviewed" or edit["action"] not in (
                        "replace",
                        "exclude",
                    ):
                        raise ValueError(f"Непроверенная правка {key}")
                    value = edit["text_ru"]
                    manual_count += 1
                    status = "reviewed" if value else "excluded"
                    flags = []
                    decisions.append({"key": key, **edit})
                else:
                    value, changes = edit_terms(s["en"], s["ru"])
                    rule_count += len(changes)
                    flags = quality_flags(s["en"], value) + terminology_flags(
                        s["en"], value
                    )
                    resolution = resolved.get(key)
                    if resolution:
                        checked_record(resolution, s["en"], s["ru"])
                        flags = [
                            f for f in flags if f not in resolution["resolved_flags"]
                        ]
                        decisions.append({"key": key, **resolution})
                    if s.get("possible_output_truncation"):
                        flags.append("possible_output_truncation")
                    status = "rule_edited" if changes else "automatic_checks_only"
                    if changes:
                        decisions.append(
                            {
                                "key": key,
                                "en_sha256": digest(s["en"]),
                                "machine_ru_sha256": digest(s["ru"]),
                                "changes": changes,
                                "text_sha256": digest(value),
                                "review": "deterministic_rules_not_full_semantic_review",
                            }
                        )
                s["text_ru_edited"] = value
                s["editorial_status"] = status
                s["flags_after_edit"] = sorted(set(flags))
                if value:
                    texts.append(value)
                if flags:
                    issues.append({"segment_index": i, "flags": sorted(set(flags))})
            if full:
                checked_record(full, p["text_en"])
                if full["status"] != "approved":
                    raise ValueError(f"Непроверенный абзац {pid}")
                p["text_ru"] = full["text_ru"]
                p["section_ru"] = full["section_ru"]
                p["review_status"] = "reviewed"
                p["editorial_review"] = {
                    k: full[k]
                    for k in ("reviewer", "reason", "original_sha256", "review_date")
                }
                p["editorial_review"]["text_sha256"] = digest(p["text_ru"])
                issues = []
            else:
                p["text_ru"] = " ".join(texts)
                p["section_ru"], _ = edit_terms(
                    p["section_en"], p["section_ru_machine"]
                )
                p["review_status"] = (
                    "needs_review" if issues else "targeted_checks_passed"
                )
                p.pop("editorial_review", None)
            reason = scope.get(pid, {}).get("reason")
            if p["section_en"] in headings:
                p["section_ru"] = headings[p["section_en"]]["text_ru"]
            if method_section(p["section_en"]):
                reason = "Методика поиска/статистического анализа обзора: не содержит тренировочного протокола или ответа предметной области. Перевод сохранён отдельно."
            if not p["text_ru"].strip():
                reason = (
                    reason
                    or "Все предложения исключены отдельными редакторскими решениями."
                )
            p["excluded_from_corpus"] = bool(reason)
            if reason:
                p["scope_exclusion_reason"] = reason
                exclusions.append(
                    {
                        "paragraph_id": pid,
                        "source_id": sid,
                        "reason": reason,
                        "text_sha256": digest(p["text_ru"]),
                    }
                )
            else:
                p.pop("scope_exclusion_reason", None)
            p["flags"] = sorted({f for x in issues for f in x["flags"]})
            p["editorial_coverage"] = {
                "whole_paragraph_reviewed": bool(full),
                "manually_edited_segments": manual_count,
                "rule_applications": rule_count,
            }
            if issues and not reason:
                queue.append(
                    {
                        "paragraph_id": pid,
                        "source_id": sid,
                        "topic": source["topic"],
                        "section": p["section_en"],
                        "flags": p["flags"],
                        "segments": issues,
                    }
                )
            coverage.append(
                {
                    "paragraph_id": pid,
                    "source_id": sid,
                    "status": p["review_status"],
                    "excluded": bool(reason),
                    **p["editorial_coverage"],
                }
            )
        write_json(path, data)
    write_json(ROOT / "data/review/editorial_log.json", decisions)
    write_json(ROOT / "data/review/paragraph_review_queue.json", queue)
    write_json(ROOT / "data/review/review_coverage.json", coverage)
    write_json(ROOT / "data/review/corpus_exclusions.json", exclusions)
    summary = {
        "paragraphs": len(coverage),
        "fully_reviewed_paragraphs": sum(
            x["whole_paragraph_reviewed"] for x in coverage
        ),
        "manual_segment_decisions": len(segments),
        "excluded_paragraphs": len(exclusions),
        "flagged_paragraphs": len(queue),
        "remaining_flags": dict(Counter(f for p in queue for f in p["flags"])),
        "semantic_review_of_entire_corpus": False,
        "translation_models_called": False,
    }
    write_json(ROOT / "data/review/editorial_report.json", summary)
    print(summary)


if __name__ == "__main__":
    curate()
