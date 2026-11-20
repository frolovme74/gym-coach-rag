const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function plural(count, one, few, many) {
  const tens = count % 100;
  const units = count % 10;
  if (tens >= 11 && tens <= 14) return many;
  if (units === 1) return one;
  if (units >= 2 && units <= 4) return few;
  return many;
}

const TOPIC_LABELS = {
  training: "Тренировки",
  nutrition: "Питание",
  supplements: "Добавки",
  recovery: "Восстановление",
};

// Режимы

const tabs = { ask: $("tab-ask"), program: $("tab-program") };
const panels = { ask: $("panel-ask"), program: $("panel-program") };
const results = {};
let activeTab = "ask";

function selectTab(name) {
  if (activeTab === name) return;
  activeTab = name;
  for (const key of Object.keys(tabs)) {
    tabs[key].setAttribute("aria-selected", String(key === name));
    panels[key].hidden = key !== name;
  }
  $("result").hidden = true;
  showActiveResult();
}

tabs.ask.addEventListener("click", () => selectTab("ask"));
tabs.program.addEventListener("click", () => selectTab("program"));

// Запросы

const INVALID_RESPONSE = "Не удалось обработать ответ сервера. Попробуйте ещё раз.";
const HTTP_ERRORS = {
  502: "Не удалось сформировать корректный ответ. Попробуйте ещё раз.",
  503: "Сервис ответов сейчас недоступен. Попробуйте позже.",
  504: "Сервис ответов не ответил вовремя. Попробуйте ещё раз.",
};
const PROVIDER_ERRORS = {
  quota: "Квота сервиса ответов исчерпана. Попробуйте позже.",
  rate_limit: "Слишком много запросов к сервису ответов. Попробуйте позже.",
};

async function post(path, payload) {
  let response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error("Не удалось соединиться с сервером. Проверьте подключение и попробуйте ещё раз.");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 400 && typeof data?.error === "string") {
      throw new Error(data.error);
    }
    const providerMessage = response.status === 503 && Object.hasOwn(PROVIDER_ERRORS, data?.code)
      ? PROVIDER_ERRORS[data.code]
      : null;
    throw new Error(providerMessage || HTTP_ERRORS[response.status] || INVALID_RESPONSE);
  }
  if (!Array.isArray(data?.sources)) throw new Error(INVALID_RESPONSE);
  return data;
}

async function submit(name, form, payload, render) {
  if (form.querySelector("button[type=submit]").disabled) return;
  const controls = form.querySelectorAll("button:enabled, textarea:enabled, input:enabled");
  const status = $(name + "-status");
  const hint = $(name + "-hint");

  delete results[name];
  if (activeTab === name) $("result").hidden = true;
  hint.textContent = "";
  status.textContent = status.dataset.loading;
  for (const control of controls) control.disabled = true;

  try {
    const data = await post("/api/" + name, payload);
    results[name] = { data, render };
  } catch (error) {
    hint.textContent = error.message;
  } finally {
    status.textContent = "";
    for (const control of controls) control.disabled = false;
  }
  if (activeTab === name) showActiveResult(true);
}

function showActiveResult(scroll = false) {
  const result = results[activeTab];
  if (!result) return;
  try {
    result.render(result.data, scroll);
  } catch {
    delete results[activeTab];
    $(activeTab + "-hint").textContent = INVALID_RESPONSE;
  }
}

function scrollToElement(element) {
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  element.scrollIntoView({ behavior: reduceMotion ? "instant" : "smooth", block: "nearest" });
}

// Связь чипов в тексте с карточками источников

function sourceCard(number) {
  return document.querySelector(`.source[data-n="${number}"]`);
}

function setActive(number, active) {
  sourceCard(number)?.classList.toggle("active", active);
  for (const cite of document.querySelectorAll(`.cite[data-n="${number}"]`)) {
    cite.classList.toggle("active", active);
  }
}

function openSource(number) {
  const card = sourceCard(number);
  if (!card) return;
  card.querySelector(".source-toggle").setAttribute("aria-expanded", "true");
  scrollToElement(card.querySelector("mark") ?? card);
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 900);
}

// Превращает «[1]» в тексте в чипы, связанные с источником под тем же номером.
function appendWithCitations(parent, text, sourceCount) {
  for (const part of text.split(/(\[\d+\])/)) {
    const number = Number(part.match(/^\[(\d+)\]$/)?.[1]);
    if (number >= 1 && number <= sourceCount) {
      const cite = el("button", "cite", String(number));
      cite.type = "button";
      cite.dataset.n = number;
      cite.setAttribute("aria-label", `Источник ${number}`);
      cite.addEventListener("mouseenter", () => setActive(number, true));
      cite.addEventListener("mouseleave", () => setActive(number, false));
      cite.addEventListener("focus", () => setActive(number, true));
      cite.addEventListener("blur", () => setActive(number, false));
      cite.addEventListener("click", () => openSource(number));
      // Склеиваем чип с предыдущим словом, чтобы он не уезжал на новую строку один.
      const before = parent.lastChild;
      const word = before?.nodeType === Node.TEXT_NODE && before.data.match(/(\S+)\s?$/);
      if (word) {
        before.data = before.data.slice(0, word.index);
        const glue = el("span", "nowrap", `${word[1]} `);
        glue.append(cite);
        parent.append(glue);
      } else {
        (before?.classList?.contains("nowrap") ? before : parent).append(cite);
      }
    } else if (part) {
      parent.append(part);
    }
  }
}

function highlightedFragment(source) {
  const fragment = el("span", "source-fragment");
  const ranges = [];
  for (const quote of source.quotes ?? []) {
    if (typeof quote !== "string" || !quote) continue;
    let start = source.fragment.indexOf(quote);
    while (start !== -1) {
      ranges.push([start, start + quote.length]);
      start = source.fragment.indexOf(quote, start + quote.length);
    }
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const [start, end] of ranges) {
    const previous = merged.at(-1);
    if (previous && start <= previous[1]) previous[1] = Math.max(previous[1], end);
    else merged.push([start, end]);
  }
  let position = 0;
  for (const [start, end] of merged) {
    fragment.append(source.fragment.slice(position, start));
    fragment.append(el("mark", "", source.fragment.slice(start, end)));
    position = end;
  }
  fragment.append(source.fragment.slice(position));
  return fragment;
}

function renderSource(source, index, maxScore) {
  const number = index + 1;
  const item = el("li", "source");
  item.dataset.n = number;

  const toggle = el("button", "source-toggle");
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");

  const score = el("span", "source-score");
  if (source.selection === "parameters" || !Number.isFinite(source.score)) {
    score.textContent = "Подобрано под программу";
  } else {
    const bar = el("span", "source-bar");
    const fill = el("i");
    fill.style.width = `${maxScore > 0 ? Math.round((source.score / maxScore) * 100) : 0}%`;
    bar.append(fill);
    score.append("BM25", bar, source.score.toFixed(2));
    score.title = "Оценка совпадения с запросом, не достоверность источника";
  }

  toggle.append(
    el("span", "source-num", String(number)),
    el("span", "source-topic", TOPIC_LABELS[source.topic] ?? source.topic),
    el("span", "source-title", source.title),
    score,
    highlightedFragment(source),
  );
  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!expanded));
  });
  item.addEventListener("mouseenter", () => setActive(number, true));
  item.addEventListener("mouseleave", () => setActive(number, false));

  item.append(toggle);
  for (const url of source.references || []) {
    if (!/^https?:\/\//i.test(url)) continue;
    const link = el("a", "source-original", "Материал к справке ↗");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    item.append(link);
  }
  if (/^https?:\/\//i.test(source.url || "")) {
    const link = el("a", "source-original", "Открыть оригинал ↗");
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    item.append(link);
  }
  return item;
}

function trustLine(sources) {
  if (sources.length === 0) return "";
  const fragments = sources.length;
  const documents = new Set(sources.map((source) => source.title)).size;
  return (
    `${fragments} ${plural(fragments, "фрагмент", "фрагмента", "фрагментов")} из ` +
    `${documents} ${plural(documents, "документа", "документов", "документов")}`
  );
}

function showResult(title, data, body, scroll) {
  const maxScore = Math.max(0, ...data.sources.map((source) => source.score));
  const sources = data.sources.map((source, index) => renderSource(source, index, maxScore));
  $("result-title").textContent = title;
  $("result-badge").hidden = data.mode !== "stub";
  $("trust").textContent = trustLine(data.sources);
  $("trust").hidden = data.sources.length === 0;
  $("result-body").replaceChildren(...body);
  $("caution").hidden = !data.sources.some((source) => source.topic === "supplements");
  $("sources").replaceChildren(...sources);
  $("sources-block").hidden = data.sources.length === 0;
  $("result").classList.toggle("no-sources", data.sources.length === 0);
  $("result").hidden = false;
  if (scroll) scrollToElement($("result"));
}

function renderAnswer(data, scroll) {
  if (typeof data.found !== "boolean" || typeof data.answer !== "string") {
    throw new Error(INVALID_RESPONSE);
  }
  if (!data.found) {
    const empty = el("div", "empty");
    empty.append(el("strong", "", "В материалах не найдено"), data.answer);
    showResult("Ответ", data, [empty], scroll);
    return;
  }
  const answer = el("p", "answer");
  appendWithCitations(answer, data.answer, data.sources.length);
  showResult("Ответ", data, [answer], scroll);
}

function renderProgram(data, scroll) {
  if (typeof data.title !== "string" || typeof data.note !== "string") {
    throw new Error(INVALID_RESPONSE);
  }
  if (data.found === false) {
    showResult(data.title, data, [el("p", "answer", data.note)], scroll);
    return;
  }
  const days = el("div", "days");
  for (const day of data.days) {
    const block = el("div", "day");
    const table = el("table");
    for (const exercise of day.exercises) {
      const row = table.insertRow();
      const name = row.insertCell();
      const citations = (exercise.citations ?? []).map((n) => `[${n}]`).join(" ");
      appendWithCitations(name, exercise.name + " " + citations, data.sources.length);
      row.insertCell().textContent = `${exercise.sets} × ${exercise.reps}`;
    }
    block.append(el("h4", "", day.name), table);
    days.append(block);
  }
  const note = el("p", "answer note");
  appendWithCitations(note, data.note, data.sources.length);
  showResult(data.title, data, [days, note], scroll);
}

// Формы

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) {
    $("ask-hint").textContent = "Введите вопрос";
    return;
  }
  submit("ask", event.currentTarget, { question }, renderAnswer);
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("ask-form").requestSubmit();
  }
});

$("examples").addEventListener("click", (event) => {
  if (event.target.classList.contains("chip")) {
    $("question").value = event.target.textContent;
    $("ask-form").requestSubmit();
  }
});

$("program-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = {
    goal: form.get("goal"),
    level: form.get("level"),
    days: Number(form.get("days")),
    equipment: form.get("equipment"),
  };
  submit("program", event.currentTarget, payload, renderProgram);
});
