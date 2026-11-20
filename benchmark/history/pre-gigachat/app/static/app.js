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

function selectTab(name) {
  for (const key of Object.keys(tabs)) {
    tabs[key].setAttribute("aria-selected", String(key === name));
    panels[key].hidden = key !== name;
  }
  $("result").hidden = true;
}

tabs.ask.addEventListener("click", () => selectTab("ask"));
tabs.program.addEventListener("click", () => selectTab("program"));

// Запросы

async function post(path, payload) {
  let response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error("Не удалось получить ответ. Проверьте, что сервер запущен.");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || "Не удалось получить ответ.");
  }
  return data;
}

async function submit(form, hint, path, payload, render) {
  const button = form.querySelector("button[type=submit]");
  const label = button.textContent;
  hint.textContent = "";
  button.disabled = true;
  button.textContent = button.dataset.busy;
  try {
    render(await post(path, payload));
  } catch (error) {
    hint.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
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
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
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

function renderSource(source, index, maxScore) {
  const number = index + 1;
  const item = el("li", "source");
  item.dataset.n = number;

  const toggle = el("button", "source-toggle");
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");

  const score = el("span", "source-score");
  const bar = el("span", "source-bar");
  const fill = el("i");
  fill.style.width = `${maxScore > 0 ? Math.round((source.score / maxScore) * 100) : 0}%`;
  bar.append(fill);
  score.append("BM25", bar, source.score.toFixed(2));

  toggle.append(
    el("span", "source-num", String(number)),
    el("span", "source-topic", TOPIC_LABELS[source.topic] ?? source.topic),
    el("span", "source-title", source.title),
    score,
    el("span", "source-fragment", source.fragment),
  );
  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!expanded));
  });
  item.addEventListener("mouseenter", () => setActive(number, true));
  item.addEventListener("mouseleave", () => setActive(number, false));

  item.append(toggle);
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

function showResult(title, data, body) {
  const maxScore = Math.max(0, ...data.sources.map((source) => source.score));
  $("result-title").textContent = title;
  $("result-badge").hidden = data.mode !== "stub";
  $("trust").textContent = trustLine(data.sources);
  $("trust").hidden = data.sources.length === 0;
  $("result-body").replaceChildren(...body);
  $("caution").hidden = !data.sources.some((source) => source.topic === "supplements");
  $("sources").replaceChildren(
    ...data.sources.map((source, index) => renderSource(source, index, maxScore)),
  );
  $("sources-block").hidden = data.sources.length === 0;
  $("result").classList.toggle("no-sources", data.sources.length === 0);
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderAnswer(data) {
  if (!data.found) {
    const empty = el("div", "empty");
    empty.append(el("strong", "", "В материалах не найдено"), data.answer);
    showResult("Ответ", data, [empty]);
    return;
  }
  const answer = el("p", "answer");
  appendWithCitations(answer, data.answer, data.sources.length);
  showResult("Ответ", data, [answer]);
}

function renderProgram(data) {
  const days = el("div", "days");
  for (const day of data.days) {
    const block = el("div", "day");
    const table = el("table");
    for (const exercise of day.exercises) {
      const row = table.insertRow();
      row.insertCell().textContent = exercise.name;
      row.insertCell().textContent = `${exercise.sets} × ${exercise.reps}`;
    }
    block.append(el("h4", "", day.name), table);
    days.append(block);
  }
  const note = el("p", "answer note");
  appendWithCitations(note, data.note, data.sources.length);
  showResult(data.title, data, [days, note]);
}

// Формы

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) {
    $("ask-hint").textContent = "Введите вопрос";
    return;
  }
  submit(event.target, $("ask-hint"), "/api/ask", { question }, renderAnswer);
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
  const form = new FormData(event.target);
  const payload = {
    goal: form.get("goal"),
    level: form.get("level"),
    days: Number(form.get("days")),
    equipment: form.get("equipment"),
  };
  submit(event.target, $("program-hint"), "/api/program", payload, renderProgram);
});