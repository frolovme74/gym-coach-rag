# Живая проверка

Единственный текущий вспомогательный скрипт — `rag_smoke.py`:

```bash
python scripts/rag_smoke.py --live --suite current --output /tmp/gym-live-check.json
```

Он поднимает временный HTTP-сервер, вызывает настоящий GigaChat и сохраняет
ответы, цитаты и результаты без ключа. Расходует квоту. Текущий набор включает
13 вопросов, две программы и 12 отказных запросов. Дополнительные наборы —
`basic` и `refusals`; список опций доступен через `--help`.

Сборка индекса — `python -m app.rag.build_index`, автономный benchmark —
`python -m benchmark.evaluate --output PATH`. Подготовительные скрипты старых
версий перенесены в архив вне проекта; см. `../spec/tech/maintenance.md`.
