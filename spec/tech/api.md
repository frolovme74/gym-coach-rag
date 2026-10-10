# API

Все ответы — JSON в UTF-8. Поле `mode` показывает, что сформировало ответ: `stub` (заглушка) или `rag` (поиск и LLM).

## GET /api/health

```json
{"status": "ok", "mode": "stub"}
```

## POST /api/ask

Запрос:

```json
{"question": "Сколько отдыхать между подходами?"}
```

`question` — непустая строка до 500 символов.

Ответ 200:

```json
{
  "mode": "stub",
  "found": true,
  "answer": "Текст ответа со ссылками [1].",
  "sources": [
    {"doc_id": "demo_rest", "topic": "training", "title": "Название документа", "fragment": "Текст фрагмента", "score": 7.42}
  ]
}
```

`topic` — область документа: `training`, `nutrition`, `supplements` или `recovery`.

Номер в квадратных скобках в `answer` — порядковый номер элемента в `sources`, начиная с 1.

Если в корпусе нет ответа: `found` равно `false`, `sources` пуст, `answer` содержит сообщение об этом.

## POST /api/program

Запрос:

```json
{"goal": "muscle", "level": "beginner", "days": 3, "equipment": "gym"}
```

| Поле | Значения |
|---|---|
| `goal` | `strength`, `muscle`, `endurance` |
| `level` | `beginner`, `intermediate` |
| `days` | целое от 2 до 5 |
| `equipment` | `gym`, `dumbbells` |

Ответ 200:

```json
{
  "mode": "stub",
  "title": "Программа на 3 дня в неделю",
  "days": [
    {
      "name": "День 1",
      "exercises": [
        {"name": "Приседания со штангой", "sets": 3, "reps": "8–12"}
      ]
    }
  ],
  "note": "Пояснение к программе [1].",
  "sources": [
    {"doc_id": "demo_frequency", "topic": "training", "title": "Название документа", "fragment": "Текст фрагмента", "score": 5.1}
  ]
}
```

## Ошибки

| Код | Когда | Тело |
|---|---|---|
| 400 | некорректный JSON или значения полей | `{"error": "описание"}` |
| 404 | неизвестный путь | `{"error": "Не найдено"}` |

## Статика

`GET /` и остальные пути отдают файлы из `app/static/`.
