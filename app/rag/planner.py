"""Составление программ из карточек корпуса и каталога конструктора."""

from itertools import chain

from app.llm import LLMError
from app.rag.prompts import passages

GOALS = {
    "strength": "Сила",
    "muscle": "Мышечная масса",
    "endurance": "Выносливость",
    "fitness": "Общая физподготовка",
    "body_composition": "Снижение жировой массы",
}
LEVELS = {
    "beginner": "Новичок",
    "intermediate": "Средний уровень",
    "advanced": "Продвинутый уровень",
}
EQUIPMENT = {
    "gym": "Тренажёрный зал",
    "dumbbells": "Только гантели",
    "bodyweight": "Без оборудования",
    "bands": "Резиновые петли",
    "kettlebell": "Одна гиря",
}
FORMATS = {"full_body": "Фуллбади", "split": "Сплит"}
LEVEL_PATTERNS = {"knee", "push"}
ACCESSORY_PATTERNS = {"arms", "calf", "core"}

WEEKDAYS = (
    "Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"
)
FULL_BODY = {
    2: [(0, "full"), (3, "full")],
    3: [(0, "full"), (2, "full"), (4, "full")],
    4: [(0, "full"), (1, "light_full"), (3, "full"), (5, "full")],
    5: [(0, "full"), (1, "light_full"), (2, "full"), (4, "full"), (5, "light_full")],
}
SPLIT = {
    2: [(0, "upper"), (3, "lower")],
    3: [(0, "push"), (2, "pull"), (4, "lower")],
    4: [(0, "upper"), (1, "lower"), (3, "upper"), (4, "lower")],
    5: [(0, "upper"), (1, "lower"), (3, "upper"), (4, "lower"), (6, "light_lower")],
}
LEGACY = {
    2: FULL_BODY[2],
    3: FULL_BODY[3],
    4: SPLIT[4],
    5: SPLIT[4] + [(6, "light_full")],
}
PATTERNS = {
    "full": ["knee", "posterior", "push", "pull", "core"],
    "upper": ["push", "pull", "shoulders", "pull"],
    "lower": ["knee", "posterior", "calf", "core"],
    "push": ["push", "shoulders", "core"],
    "pull": ["pull", "pull", "core"],
}
EXTRA_PATTERNS = {
    "upper": {"arms", "core"},
    "lower": set(),
    "push": {"arms"},
    "pull": {"arms"},
}
LABELS = {
    "full": "всё тело",
    "upper": "верх тела",
    "lower": "низ тела",
    "push": "жимы и плечи",
    "pull": "тяги и спина",
}


def schedule_for(params):
    plan = {"full_body": FULL_BODY, "split": SPLIT}.get(params.get("format"), LEGACY)
    return [
        (WEEKDAYS[day], kind.removeprefix("light_"), kind.startswith("light_"))
        for day, kind in plan[params["days"]]
    ]


def prepare(pipeline, params):
    catalog, prescription, rule_doc, schedule_doc = {}, None, None, None
    level_exercises, rest_doc = None, None
    accessory_reps, accessory_doc = None, None
    entries = chain(pipeline.manifest.items(), pipeline.program_cards.items())
    for did, metadata in entries:
        program = metadata.get("program", {})
        equipment_levels = program.get("level_exercises", {})
        level_exercises = equipment_levels.get(params["equipment"], level_exercises)
        for ex in program.get("exercises", []):
            if params["equipment"] in ex["equipment"]:
                catalog[ex["id"]] = {**ex, "doc_id": did}
        p = program.get("prescription", {})
        if p.get("goal") == params["goal"] and p.get("level") == params["level"]:
            prescription, rule_doc = p, did
        if program.get("schedule") == params.get("format", True):
            schedule_doc = did
        if program.get("rest_guidance"):
            rest_doc = did
        reps = program.get("accessory_reps", {}).get(params["goal"])
        if reps:
            accessory_reps, accessory_doc = reps, did
    if not catalog or not prescription or not schedule_doc:
        return None
    if level_exercises is not None:
        variants = set(level_exercises.get(params["level"], []))
        if (
            not variants <= catalog.keys()
            or {catalog[eid]["pattern"] for eid in variants} != LEVEL_PATTERNS
        ):
            raise LLMError("config", "В каталоге неполный набор упражнений для уровня.")
        catalog = {
            eid: ex
            for eid, ex in catalog.items()
            if ex["pattern"] not in LEVEL_PATTERNS or eid in variants
        }
    # Prefer machines in gym mode, without removing dumbbell alternatives.
    ordered = sorted(
        catalog,
        key=lambda eid: (
            not eid.startswith("gym_") if params["equipment"] == "gym" else False,
            eid,
        ),
    )
    drafts = []
    for day_index, (_, kind, _) in enumerate(schedule_for(params)):
        selected = []
        for pattern in PATTERNS[kind]:
            candidates = [
                eid
                for eid in ordered
                if catalog[eid]["pattern"] == pattern and eid not in selected
            ]
            if (
                not candidates
                and pattern == "pull"
                and PATTERNS[kind].count("pull") > 1
            ):
                # When the catalog has one row, an arm exercise completes the day.
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
    if accessory_doc:
        prescription = {
            **prescription,
            "accessory_reps": accessory_reps,
            "accessory_doc": accessory_doc,
        }
    if rest_doc:
        prescription = {**prescription, "rest_doc": rest_doc}
    return catalog, prescription, rule_doc, schedule_doc, drafts, sources


def valid_days(value, catalog, params):
    days = value.get("days")
    if not isinstance(days, list) or len(days) != params["days"]:
        return False
    for day, (_, kind, _) in zip(days, schedule_for(params)):
        ids = day.get("exercise_ids") if isinstance(day, dict) else None
        if (
            not isinstance(ids, list)
            or not min(4, len(PATTERNS[kind])) <= len(ids) <= 6
            or any(not isinstance(e, str) or e not in catalog for e in ids)
        ):
            return False
        if len(set(ids)) != len(ids):
            return False
        day_patterns = [catalog[e]["pattern"] for e in ids]
        patterns = set(day_patterns)
        required = set(PATTERNS[kind])
        if not required <= patterns:
            return False
        if kind in EXTRA_PATTERNS and not patterns <= required | EXTRA_PATTERNS[kind]:
            return False
        if PATTERNS[kind].count("pull") > 1 and sum(
            p in {"pull", "arms"} for p in day_patterns
        ) < 2:
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
    schedule = schedule_for(params)
    request = {
        **params,
        "schedule": [
            {"day": day, "kind": kind, "light": light} for day, kind, light in schedule
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
                    if p == rule.get("guidance") or any(
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
                            "Жимы включают",
                            "Тяги включают",
                        )
                    )
                ]
            if did == rule.get("accessory_doc"):
                source["quotes"] = [
                    p for p in parts if p.startswith(f"{GOALS[params['goal']]} — ")
                ]
                if not source["quotes"]:
                    raise LLMError("config", "В карточке нет нагрузки для выбранной цели.")
            used[did] = source
        return list(used).index(did) + 1

    rule_ref, schedule_ref = cite(rule_doc), cite(schedule_doc)
    accessory_ref = cite(rule["accessory_doc"]) if "accessory_doc" in rule else None
    days = []
    for day, (weekday, kind, light) in zip(chosen["days"], schedule):
        exercises = []
        for eid in day["exercise_ids"]:
            ex = catalog[eid]
            reps = rule["reps"]
            citations = [cite(ex["doc_id"]), rule_ref, schedule_ref]
            if accessory_ref is not None and ex["pattern"] in ACCESSORY_PATTERNS:
                reps = rule["accessory_reps"]
                citations.append(accessory_ref)
            exercises.append(
                dict(
                    name=ex["name"],
                    sets=1 if light else rule["sets"],
                    reps=reps,
                    citations=citations,
                )
            )
        label = f"{LABELS[kind]}, лёгкая техника" if light else LABELS[kind]
        days.append(dict(name=f"{weekday} — {label}", exercises=exercises))
    details = [LEVELS[params["level"]], GOALS[params["goal"]], EQUIPMENT[params["equipment"]]]
    if format_name := params.get("format"):
        details.append(FORMATS[format_name])
    if rest_doc := rule.get("rest_doc"):
        rest_ref = cite(rest_doc)
        rest = f"{used[rest_doc]['fragment']} [{rest_ref}]"
    else:
        rest = f"Отдых между рабочими подходами — {rule['rest']}. [{rule_ref}]"
    note = (
        f"{' · '.join(details)}. Основные упражнения — {rule['sets']} × {rule['reps']}. [{rule_ref}] "
        f"{rest} "
        "Начинайте с нагрузки, при которой сохраняется контроль движений. "
        "В таблице указаны рабочие подходы. "
        f"Сохраняйте промежутки между тренировками одной группы мышц. [{schedule_ref}]"
    )
    if accessory_ref is not None:
        note += f" Для вспомогательных упражнений — {rule['accessory_reps']} повторений. [{accessory_ref}]"
    if guidance := rule.get("guidance"):
        note += f" {guidance} [{rule_ref}]"
    if params["days"] >= 4 and params["level"] == "beginner":
        note += f" Новичку обычно проще начать с 2–3 дней; дополнительные посещения не должны увеличивать нагрузку через усталость. [{schedule_ref}]"
    if any(light for _, _, light in schedule):
        note += f" На занятиях с пометкой «лёгкая техника» — один рабочий подход на упражнение. [{schedule_ref}]"
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
