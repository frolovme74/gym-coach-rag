"""Поиск v3.1 → GigaChat → проверенные цитаты → контракт API."""

import json
import math
import re
import threading
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from app.llm import GigaChatClient, LLMError
from app.rag.prompts import messages, passages
from app.rag.retrieval import RULESET, ConceptBM25, _digest, _fingerprint

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "benchmark/retrieval-v3.1/selected_config.json"
NO_ANSWER = "В найденных материалах недостаточно сведений для обоснованного ответа. Попробуйте уточнить вопрос."
NO_PROGRAM = "В найденных материалах недостаточно сведений для полной программы с указанными параметрами. Попробуйте другие параметры."
OUT_OF_SCOPE = "Я отвечаю на общие вопросы о тренировках, питании, спортивных добавках и восстановлении. Лечение, запрещённые средства и экстремальные диеты не входят в задачи ассистента."
GOALS = {
    "strength": "сила",
    "muscle": "мышечная масса гипертрофия",
    "endurance": "мышечная выносливость",
}
LEVELS = {
    "beginner": "новичок начинающий",
    "intermediate": "средний уровень тренированности",
}
EQUIPMENT = {"gym": "тренажерный зал штанга", "dumbbells": "только гантели"}
FORBIDDEN = re.compile(
    r"\b(?:стероид\w*|анаболик\w*|тестостерон\w*|гормон\w*|допинг\w*|"
    r"рецептурн\w*|тренболон\w*|станозолол\w*|инсулин\w*|"
    r"травм\w*|болезн\w*|заболеван\w*|беремен\w*|лечени\w*|лечить|"
    r"грыж\w*|диабет\w*|гипертони\w*|перелом\w*|анорекс\w*|булими\w*|"
    r"голодан\w*|сгонк\w*|рпп|steroids?|doping|pregnan\w*|injur\w*)\b|"
    r"\bболит\b|\bболь\w*\s+в\s+(?:колен\w*|спин\w*|сустав\w*)|"
    r"\b(?:экстремальн\w*|жестк\w*)\s+диет\w*|"
    r"\b(?:состав\w*|напиш\w*|сдела\w*)\b.{0,50}\bменю\b",
    re.IGNORECASE,
)


def invalid_output(reason="Неверный JSON или структура ответа"):
    error = LLMError(
        "grounding",
        "Не удалось подтвердить ответ цитатами из источников. Попробуйте уточнить запрос.",
    )
    error.validation_reason = reason
    return error


def numbers(text):
    return {Decimal(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", text)}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повторный ключ JSON")
        result[key] = value
    return result


def text_field(value, limit=1800):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise invalid_output()
    if re.search(r"\[\s*\d+\s*\]", value):
        raise invalid_output()
    return value.strip()


class Evidence:
    """Нумерует только реально использованные источники в порядке цитирования."""

    def __init__(self, candidates):
        self.candidates = candidates
        self.used = {}

    def validate(self, text, entries):
        if not isinstance(entries, list) or not 1 <= len(entries) <= 5:
            raise invalid_output()
        quotes, refs = [], []
        for entry in entries:
            if not isinstance(entry, dict):
                raise invalid_output()
            n, ids = entry.get("source"), entry.get("passages")
            if (
                type(n) is not int
                or not 1 <= n <= len(self.candidates)
                or set(entry) != {"source", "passages"}
                or not isinstance(ids, list)
                or not 1 <= len(ids) <= 5
            ):
                raise invalid_output()
            parts = passages(self.candidates[n - 1]["fragment"])
            if any(type(i) is not int or not 1 <= i <= len(parts) for i in ids):
                raise invalid_output()
            selected = [parts[i - 1] for i in ids]
            if n not in self.used:
                self.used[n] = {**self.candidates[n - 1], "quotes": []}
            for quote in selected:
                if quote not in self.used[n]["quotes"]:
                    self.used[n]["quotes"].append(quote)
            ref = list(self.used).index(n) + 1
            if ref not in refs:
                refs.append(ref)
            quotes.extend(selected)
        if not numbers(text) <= numbers(" ".join(quotes)):
            raise invalid_output(
                "Числа утверждения отсутствуют в указанных предложениях. Удали неподтверждённые числа, не придумывай другие."
            )
        return refs, quotes

    def claims(self, claims):
        if not isinstance(claims, list) or not 1 <= len(claims) <= 6:
            raise invalid_output()
        paragraphs = []
        for claim in claims:
            if not isinstance(claim, dict):
                raise invalid_output()
            text = text_field(claim.get("text"))
            refs, quotes = self.validate(text, claim.get("evidence"))
            if re.search(
                r"другие находят ухудшение|результаты были смешанными|результаты неоднозначны",
                " ".join(quotes),
                re.I,
            ) and not re.search(
                r"неоднознач|смешан|противореч|ухудш|не во всех|плацебо", text, re.I
            ):
                raise invalid_output(
                    "В выбранных цитатах есть противоречащие результаты или ухудшение. Обязательно передай эту неоднозначность; нельзя оставить только положительный эффект."
                )
            if re.search(
                r"предполага|предполож|гипотез", " ".join(quotes), re.I
            ) and not re.search(
                r"предполага|предполож|гипотез|может|могут|возмож|вероят", text, re.I
            ):
                raise invalid_output(
                    "Цитаты содержат предположение или гипотезу. Нельзя излагать её как доказанный факт: сохрани неопределённость или выбери другое подтверждённое утверждение."
                )
            if (
                numbers(text)
                and re.search(
                    r"при отсутствии дефицита калорий", " ".join(quotes), re.I
                )
                and not re.search(r"дефицит|калорийност", text, re.I)
            ):
                raise invalid_output(
                    "Числовая норма в цитате относится к отсутствию дефицита калорий. Обязательно сохрани это условие в ответе."
                )
            if re.search(
                r"\d\s*(?:мг|г)\s*/\s*(?:день|сут)", text, re.I
            ) and not re.search(r"исследован|в обзоре|протокол", text, re.I):
                raise invalid_output(
                    "Дозировка из исследовательского протокола изложена как общее правило. Если вопрос не о дозировке, убери её; иначе явно укажи исследовательский контекст, сохрани единицы и сроки."
                )
            paragraphs.append(text + " " + " ".join(f"[{n}]" for n in refs))
        return "\n\n".join(paragraphs)


class RAGPipeline:
    def __init__(self, documents, index, manifest, threshold, client=None):
        self.documents = {d["doc_id"]: d for d in documents}
        self.manifest = {d["doc_id"]: d for d in manifest}
        self.index, self.threshold, self._client = index, threshold, client
        self._client_lock = threading.Lock()

    @classmethod
    def from_project(cls, root=ROOT, client=None):
        root = Path(root)
        try:
            profile = json.loads((root / PROFILE).read_text())
            raw = (root / profile["corpus"]).read_bytes()
            documents = json.loads(raw)
            if _fingerprint(documents) != profile["corpus_sha256"]:
                raise ValueError("Корпус отличается от откалиброванного")
            if _digest(RULESET) != profile["rules_sha256"]:
                raise ValueError("Правила отличаются от откалиброванных")
            index_path = root / profile["index"]
            index = (
                ConceptBM25.load(index_path)
                if index_path.exists()
                else ConceptBM25.from_documents(documents, **profile["retrieval"])
            )
            if index.config != profile["retrieval"] or not index.matches_documents(
                documents
            ):
                raise ValueError("Индекс не соответствует профилю/корпусу")
            threshold = profile["refusal_threshold"]
            if (
                type(threshold) not in (int, float)
                or not math.isfinite(threshold)
                or threshold < 0
            ):
                raise ValueError("Неверный порог")
            manifest = json.loads(
                (root / profile["corpus"]).with_name("manifest.json").read_text()
            )
            if {d["doc_id"] for d in documents} != {d["doc_id"] for d in manifest}:
                raise ValueError("Manifest не соответствует корпусу")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise LLMError(
                "config",
                "Не удалось загрузить согласованный корпус и индекс v3. Проверьте файлы проекта.",
            ) from exc
        return cls(documents, index, manifest, threshold, client)

    def retrieve(self, query):
        hits = self.index.search(query, k=5)
        if not hits or hits[0].score <= self.threshold:
            return []
        sources = []
        for hit in hits:
            d = self.documents[hit.doc_id]
            url = self.manifest[hit.doc_id].get("source_url", "")
            parsed = urlsplit(url)
            if parsed.scheme not in ("https", "http") or not parsed.netloc:
                url = ""
            sources.append(
                dict(
                    doc_id=d["doc_id"],
                    topic=d["topic"],
                    title=d["source"],
                    section=d["section"],
                    fragment=d["text"],
                    score=hit.score,
                    url=url,
                )
            )
        return sources

    def _generate(self, kind, request, sources, feedback=None):
        with self._client_lock:
            if self._client is None:
                self._client = GigaChatClient.from_env()
        prompt = messages(kind, request, sources)
        if feedback:
            prompt.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "validation_error": feedback,
                            "instruction": 'Предыдущий ответ отклонён. Сформируй заново корректный JSON, строго по источникам. Если обоснование невозможно, верни только {"found":false}. Не повторяй неподтверждённые упражнения и утверждения.',
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        response = self._client.generate(
            prompt,
            max_tokens=3000 if kind == "program" else None,
        )
        try:
            # Fences are tolerated; unrelated prefix/suffix prose is not.
            text = response.text.strip()
            if text.startswith("```json\n") and text.endswith("```"):
                text = text[8:-3].strip()
            value = json.loads(text, object_pairs_hook=unique_object)
        except (ValueError, TypeError):
            raise invalid_output() from None
        if not isinstance(value, dict) or type(value.get("found")) is not bool:
            raise invalid_output()
        return value

    def _run(self, kind, request, sources, render, refusal):
        feedback = None
        for attempt in range(2):
            value = None
            try:
                value = self._generate(kind, request, sources, feedback)
                return render(value) if value["found"] else refusal
            except LLMError as error:
                if error.code != "grounding":
                    raise
                if attempt:
                    if kind == "ask" and value:
                        try:
                            return self._extractive(value, sources)
                        except LLMError:
                            pass
                    raise
                feedback = getattr(
                    error,
                    "validation_reason",
                    "Неверный JSON или неподтверждённые источниками утверждения",
                )

    @staticmethod
    def _answer(value, sources):
        evidence = Evidence(sources)
        answer = evidence.claims(value.get("claims"))
        return dict(
            mode="rag",
            found=True,
            answer_type="generated",
            answer=answer,
            sources=list(evidence.used.values()),
        )

    @staticmethod
    def _extractive(value, sources):
        claims = value.get("claims")
        if not isinstance(claims, list) or not 1 <= len(claims) <= 6:
            raise invalid_output()
        # Validate ALL selected references before exposing even a subset.
        checked = Evidence(sources)
        for claim in claims:
            if not isinstance(claim, dict):
                raise invalid_output()
            checked.validate("", claim.get("evidence"))
        selected, paragraphs, size = [], [], 0
        for source in checked.used.values():
            kept = []
            for quote in source["quotes"]:
                if len(paragraphs) >= 4 or size + len(quote) > 3200:
                    continue
                kept.append(quote)
                size += len(quote)
                # Brackets in corpus prose must not become UI citation buttons.
                display = quote.replace("[", "［").replace("]", "］")
                paragraphs.append(f"«{display}» [{len(selected) + 1}]")
            if kept:
                selected.append({**source, "quotes": kept})
        if not paragraphs:
            raise invalid_output()
        return dict(
            mode="rag",
            found=True,
            answer_type="extractive",
            answer="Не удалось составить надёжный краткий ответ. Привожу отрывки из найденных источников для проверки:\n\n"
            + "\n\n".join(paragraphs),
            sources=selected,
        )

    def ask(self, payload):
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            raise ValueError("Введите непустой вопрос длиной до 500 символов")
        question = question.strip()
        refusal = dict(mode="rag", found=False, answer=NO_ANSWER, sources=[])
        if FORBIDDEN.search(question.replace("ё", "е")):
            return {**refusal, "answer": OUT_OF_SCOPE}
        sources = self.retrieve(question)
        if not sources:
            return refusal
        return self._run(
            "ask",
            question,
            sources,
            lambda value: self._answer(value, sources),
            refusal,
        )

    def program(self, payload):
        for field, options in (
            ("goal", GOALS),
            ("level", LEVELS),
            ("equipment", EQUIPMENT),
        ):
            if not isinstance(payload.get(field), str) or payload[field] not in options:
                raise ValueError(f"Некорректное поле {field}")
        if type(payload.get("days")) is not int or not 2 <= payload["days"] <= 5:
            raise ValueError("Число дней должно быть целым от 2 до 5")
        parameters = {k: payload[k] for k in ("goal", "level", "days", "equipment")}
        query = (
            f"Программа силовых тренировок {GOALS[payload['goal']]} "
            f"{LEVELS[payload['level']]} {payload['days']} дня в неделю "
            f"{EQUIPMENT[payload['equipment']]} упражнения подходы повторения"
        )
        refusal = dict(
            mode="rag",
            found=False,
            title="Программа не составлена",
            days=[],
            note=NO_PROGRAM,
            sources=[],
        )
        sources = self.retrieve(query)
        if not sources:
            return refusal
        return self._run(
            "program",
            parameters,
            sources,
            lambda value: self._program(value, sources, parameters),
            refusal,
        )

    @staticmethod
    def _program(value, sources, payload):
        days = value.get("days")
        if not isinstance(days, list) or len(days) != payload["days"]:
            raise invalid_output(
                f"days должен содержать ровно {payload['days']} дней, а claims должен быть на верхнем уровне JSON."
            )
        evidence = Evidence(sources)
        clean_days = []
        for n, day in enumerate(days, 1):
            if (
                not isinstance(day, dict)
                or not isinstance(day.get("exercises"), list)
                or not 2 <= len(day["exercises"]) <= 5
            ):
                raise invalid_output()
            exercises = []
            for ex in day["exercises"]:
                if not isinstance(ex, dict):
                    raise invalid_output()
                name, reps = (
                    text_field(ex.get("name"), 100),
                    text_field(ex.get("reps"), 30),
                )
                sets = ex.get("sets")
                if (
                    type(sets) is not int
                    or not 1 <= sets <= 10
                    or not re.fullmatch(r"\d{1,2}(?:\s*[-–—]\s*\d{1,2})?", reps)
                ):
                    raise invalid_output()
                endpoints = list(map(int, re.findall(r"\d+", reps)))
                if (
                    min(endpoints) < 1
                    or max(endpoints) > 50
                    or endpoints != sorted(endpoints)
                ):
                    raise invalid_output()
                refs, quotes = evidence.validate(
                    f"{name} {sets} {reps}", ex.get("evidence")
                )
                if name.casefold() not in " ".join(quotes).casefold():
                    raise invalid_output(
                        f"Название упражнения «{name}» отсутствует в выбранных предложениях. Нельзя придумывать упражнения по общим сведениям о свободных весах. Если нет понятных названий упражнений с подходящим оборудованием, верни found:false."
                    )
                if payload["equipment"] == "dumbbells" and re.search(
                    r"штанг|тренаж|блок|турник", name, re.I
                ):
                    raise invalid_output()
                exercises.append(dict(name=name, sets=sets, reps=reps, citations=refs))
            clean_days.append(dict(name=f"День {n}", exercises=exercises))
        note = evidence.claims(value.get("claims"))
        return dict(
            mode="rag",
            found=True,
            title=f"Программа на {payload['days']} дня в неделю",
            days=clean_days,
            note=note,
            sources=list(evidence.used.values()),
        )
