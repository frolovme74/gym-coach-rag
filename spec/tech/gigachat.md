# Подключение GigaChat Freemium

Подготовлен транспортный клиент `app/llm.py`, без новых зависимостей.
Сервер использует поиск и продуктовые промпты: [контракт интеграции](integration.md).
Настройка проверена по официальной документации 20 ноября 2026.

## Получение ключа

1. Откройте [Studio](https://developers.sber.ru/studio/workspaces/my-space/get/gigachat-api) и войдите через Сбер ID.
2. В личном пространстве физлица создайте проект GigaChat API. Новый пользователь получает Freemium; проверьте доступные модели, остатки и срок квоты в проекте. Для этого шага покупать токены не нужно.
3. В настройках API получите **Authorization key**. В `.env` нужен целиком этот ключ, без префикса `Basic`; Client ID, Client Secret и временный Access token по отдельности не подходят.
4. Убедитесь, что Scope — `GIGACHAT_API_PERS`.

Источники: [регистрация](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-create-project),
[Freemium для личного пространства](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-tokens-purchase),
[ключ и access token](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-token).
Ключ не отправляйте в чат, Git, скриншоты и отчёты.

## Локальная настройка

Рабочая папка: `/Users/murad/projects/GymRAG/gym-coach-rag`.
`.env` настроен локально, с правами 0600 и путём к сертификату; ключ не публикуется.
Откройте его в редакторе и заполните только значение GIGACHAT_CREDENTIALS:

```dotenv
LLM_PROVIDER=gigachat
GIGACHAT_CREDENTIALS=ВАШ_AUTHORIZATION_KEY
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat-2-Pro
GIGACHAT_CA_BUNDLE_FILE=.certs/russian_trusted_root_ca_pem.crt
GIGACHAT_TIMEOUT=30
GIGACHAT_MAX_TOKENS=1024
GIGACHAT_TEMPERATURE=0.2
```

Шаблон для других участников — `.env.example`. При новом клонировании скопируйте
его в `.env`, установите `chmod 600 .env` и скачайте сертификат по инструкции ниже.
Переменные процесса имеют приоритет над файлом. Относительный путь CA считается
от папки `.env`. Поддерживаются строки KEY=value, одинарные/двойные кавычки,
отдельные строки комментариев и необязательный префикс export. Подстановки
`${...}`, команды shell, многострочные значения и комментарии после значения
не поддерживаются. Не нужно выполнять `source .env`: файл читает клиент.

## Сертификат

Публичный корневой сертификат уже скачан по официальной ссылке в
`.certs/russian_trusted_root_ca_pem.crt`; глобальное хранилище ОС не менялось.
Проверка TLS остаётся включённой. `.certs/` исключена из Git.
Для другого компьютера из корня проекта:

```bash
mkdir -p .certs
curl --fail --show-error --location --proto '=https' \
  https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt \
  --output .certs/russian_trusted_root_ca_pem.crt
```

Источник ссылки и назначения CA: [документация GigaChat](https://developers.sber.ru/docs/ru/gigachat/certificates).
Скачивание выполняется с проверкой HTTPS. Сертификат добавляется только в
SSLContext этого клиента, вместе с обычными доверенными сертификатами.
Отпечаток скачанного 20.11.2026 файла SHA256:
`D26D2D0231B7C39F92CC738512BA54103519E4405D68B5BD703E9788CA8ECF31`.
Провайдер может обновить сертификат; актуальный источник указан выше.

## Проверка после заполнения ключа

Из корня проекта:

```bash
venv/rag/bin/python -m app.llm --check
venv/rag/bin/python -m app.llm --smoke
```

`--check` получает токен и список моделей, проверяет наличие выбранной модели,
не генерирует текст. `--smoke` делает небольшой запрос и расходует квоту;
это проверка подключения, не готовый RAG и не оценка качества ответов.
Можно указать другой файл: `--env-file /абсолютный/путь/.env`.
Без ключа команда завершится понятной ошибкой до обращения к сети.

## Контракт клиента

```python
from app.llm import GigaChatClient, LLMError

client = GigaChatClient.from_env()
try:
    response = client.generate([
        {"role": "system", "content": "Отвечай только по переданному контексту."},
        {"role": "user", "content": "Контекст [1]: это учебный пример. Что это?"},
    ])
    print(response.text)
    print(response.usage)
except LLMError as error:
    print(error.code, str(error))
```

Один экземпляр клиента переиспользуется: он хранит access token в памяти и
последовательно выполняет обращения. Токен обновляется за 60 секунд до истечения
и один раз после 401 от API модели. OAuth `expires_at` — миллисекунды Unix.
Запросы идут на `https://api.giga.chat/v1`; OAuth — на
`https://ngw.devices.sberbank.ru:9443/api/v2/oauth`.
[Официальные адреса](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api).

HTTP 429/503 допускают один повтор с задержкой до 5 секунд; длинный Retry-After
возвращается как ошибка без преждевременного повтора. Таймаут/разрыв соединения
не повторяется автоматически, поскольку POST уже мог быть обработан.
Таймаут задан на сетевую операцию, не на весь вызов с авторизацией и повторами.
Отдельные коды ошибок: config, auth, quota, forbidden, rate_limit, tls, timeout,
network, http, response, input, truncated, blocked. Ошибки не содержат тело
ответа провайдера, ключ или access token. `length` и `blacklist` не считаются
успешным полным ответом. Для физлиц действует один поток запросов:
[ограничения](https://developers.sber.ru/docs/ru/gigachat/limitations).

## Проверено

18 тестов клиента используют подменённый HTTP-транспорт, не расходуют квоту.
После интеграции проходят 173 теста проекта; исторические файлы интерфейса
для прежних снимков benchmark сохранены в `benchmark/history/pre-gigachat/`.
Настоящие авторизация и генерация проверены после заполнения ключа пользователем.
Пайплайн проверяет структуру ответа, существование источников, номера предложений
и числовые значения. Смысловая проверка остаётся отдельной задачей. Живые прогоны
API сохраняются в `benchmark/results/2026-11-21-gigachat-smoke-*.json`.
Метод `generate` принимает необязательный `max_tokens` для конкретного вызова;
программа использует лимит 3000, вопрос — настройку из `.env`.
