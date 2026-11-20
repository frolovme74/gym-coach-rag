"""Составление программ из карточек той же базы, что используется для вопросов."""

from app.llm import LLMError

SCHEDULES = {
    2: [("Понедельник", "full"), ("Четверг", "full")],
    3: [("Понедельник", "full"), ("Среда", "full"), ("Пятница", "full")],
    4: [
        ("Понедельник", "upper"),
        ("Вторник", "lower"),
        ("Четверг", "upper"),
        ("Пятница", "lower"),
    ],
    5: [
        ("Понедельник", "upper"),
        ("Вторник", "lower"),
        ("Четверг", "upper"),
        ("Пятница", "lower"),
        ("Воскресенье", "light"),
    ],
}
PATTERNS = {
    "full": ["knee", "posterior", "push", "pull", "core"],
    "light": ["knee", "posterior", "push", "pull", "core"],
    "upper": ["push", "pull", "shoulders", "pull"],
    "lower": ["knee", "posterior", "calf", "core"],
}
LABELS = {
    "full": "всё тело",
    "upper": "верх тела",
    "lower": "низ тела",
    "light": "лёгкая техника",
}


def prepare(pipeline, params):
    catalog, prescription, rule_doc, schedule_doc = {}, None, None, None
    for did, metadata in pipeline.manifest.items():
        program = metadata.get("program", {})
        for ex in program.get("exercises", []):
            if params["equipment"] in ex["equipment"]:
                catalog[ex["id"]] = {**ex, "doc_id": did}
        p = program.get("prescription", {})
        if p.get("goal") == params["goal"] and p.get("level") == params["level"]:
            prescription, rule_doc = p, did
        if program.get("schedule"):
            schedule_doc = did
    if not catalog or not prescription or not schedule_doc:
        return None
    # Prefer machines in gym mode, without removing dumbbell alternatives.
    ordered = sorted(
        catalog,
        key=lambda eid: (
            not eid.startswith("gym_") if params["equipment"] == "gym" else False,
            eid,
        ),
    )
    drafts = []
    for day_index, (_, kind) in enumerate(SCHEDULES[params["days"]]):
        selected = []
        for pattern in PATTERNS[kind]:
            candidates = [
                eid
                for eid in ordered
                if catalog[eid]["pattern"] == pattern and eid not in selected
            ]
            if not candidates:
                # Dumbbells have one row; an arm accessory completes upper day.
                candidates = [
                    eid
                    for eid in ordered
                    if catalog[eid]["pattern"] == "arms" and eid not in selected
                ]
            if not candidates:
                return None
            selected.append(candidates[day_index % len(candidates)])
        drafts.append({"exercise_ids": selected})
    ids = list(
        dict.fromkeys(
            [rule_doc, schedule_doc] + [ex["doc_id"] for ex in catalog.values()]
        )
    )
    sources = [pipeline.source(did, None) for did in ids]
    return catalog, prescription, rule_doc, schedule_doc, drafts, sources


def valid_days(value, catalog, params):
    days = value.get("days")
    if not isinstance(days, list) or len(days) != params["days"]:
        return False
    for day, (_, kind) in zip(days, SCHEDULES[params["days"]]):
        ids = day.get("exercise_ids") if isinstance(day, dict) else None
        if (
            not isinstance(ids, list)
            or not 4 <= len(ids) <= 6
            or any(not isinstance(e, str) or e not in catalog for e in ids)
        ):
            return False
        if len(set(ids)) != len(ids):
            return False
        patterns = {catalog[e]["pattern"] for e in ids}
        if not set(PATTERNS[kind]) <= patterns:
            return False
        if kind == "upper" and patterns & {"knee", "posterior", "calf"}:
            return False
        if kind == "lower" and patterns & {"push", "pull", "shoulders", "arms"}:
            return False
    return True


def build_program(pipeline, params):
    prepared = prepare(pipeline, params)
    if prepared is None:
        return dict(
            mode="rag",
            found=False,
            title="Программа не составлена",
            days=[],
            note="В базе нет подходящих карточек программы для этих параметров.",
            sources=[],
        )
    catalog, rule, rule_doc, schedule_doc, draft, candidates = prepared
    request = {
        **params,
        "schedule": [
            {"day": day, "kind": kind} for day, kind in SCHEDULES[params["days"]]
        ],
        "catalog": [
            {k: ex[k] for k in ("id", "name", "pattern")} for ex in catalog.values()
        ],
        "draft_days": draft,
    }
    try:
        value = pipeline._generate("program", request, candidates)
    except LLMError as error:
        if error.code not in {"grounding", "truncated"}:
            raise
        value = {"found": False}
    chosen = (
        value
        if value.get("found") and valid_days(value, catalog, params)
        else {"days": draft}
    )
    if not valid_days(chosen, catalog, params):
        raise LLMError(
            "config", "Карточки не позволяют построить согласованную программу."
        )
    used = {}

    def cite(did):
        if did not in used:
            source = {
                k: v
                for k, v in pipeline.source(did, None).items()
                if not k.startswith("_")
            }
            from app.rag.prompts import passages

            parts = passages(source["fragment"])
            selected_ids = {
                eid for day in chosen["days"] for eid in day["exercise_ids"]
            }
            source["quotes"] = [
                p
                for p in parts
                if any(
                    catalog[eid]["name"] in p
                    for eid in selected_ids
                    if catalog[eid]["doc_id"] == did
                )
            ]
            if not source["quotes"]:
                source["quotes"] = parts[:3]
            if did == rule_doc:
                source["quotes"] = [
                    p
                    for p in parts
                    if any(
                        term in p
                        for term in (
                            "рабочих подхода",
                            "повторений",
                            "Отдых между",
                            "облегчённый день",
                        )
                    )
                ]
            if did == schedule_doc:
                source["quotes"] = [
                    p
                    for p in parts
                    if f"Для {params['days']} занятий" in p
                    or any(
                        term in p
                        for term in (
                            "Новичку",
                            "промежутки",
                            "облегчённая",
                            "Тренировка всего тела включает",
                            "Верх тела включает",
                            "Низ тела включает",
                        )
                    )
                ]
            used[did] = source
        return list(used).index(did) + 1

    rule_ref, schedule_ref = cite(rule_doc), cite(schedule_doc)
    days = []
    for day, (weekday, kind) in zip(chosen["days"], SCHEDULES[params["days"]]):
        exercises = []
        for eid in day["exercise_ids"]:
            ex = catalog[eid]
            exercises.append(
                dict(
                    name=ex["name"],
                    sets=1 if kind == "light" else rule["sets"],
                    reps=rule["reps"],
                    citations=[cite(ex["doc_id"]), rule_ref, schedule_ref],
                )
            )
        days.append(dict(name=f"{weekday} — {LABELS[kind]}", exercises=exercises))
    note = (
        f"Базовая программа для здорового взрослого. Отдых между рабочими подходами: {rule['rest']}. "
        f"Начинайте с нагрузки, при которой сохраняется контроль движений; разминочные подходы не включены в таблицу. [{rule_ref}] "
        f"Сохраняйте промежутки между тренировками одной группы мышц. [{schedule_ref}]"
    )
    if params["days"] >= 4 and params["level"] == "beginner":
        note += f" Новичку обычно проще начать с 2–3 дней; дополнительные посещения не должны увеличивать нагрузку через усталость. [{schedule_ref}]"
    if params["days"] == 5:
        note += f" Воскресенье — лёгкая техника, а не ещё одна тяжёлая тренировка. [{schedule_ref}]"
    return dict(
        mode="rag",
        found=True,
        title=f"Программа на неделю: {params['days']} тренировочных дня"
        if params["days"] < 5
        else "Программа на неделю: 5 тренировочных дней",
        days=days,
        note=note,
        sources=list(used.values()),
    )
