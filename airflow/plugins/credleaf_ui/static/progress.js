// The DOM nodes survive each poll: only their text, values and status change.
const rows = new Map();
const number = value => new Intl.NumberFormat().format(value);

export function duration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "ETA unavailable";
  if (seconds < 60) return "<1 min remaining";
  if (seconds < 3600) return `~${Math.ceil(seconds / 60)} min remaining`;
  return `~${(seconds / 3600).toFixed(1)} h remaining`;
}

function progress(element, value, active = false) {
  if (value == null && active) element.removeAttribute("value");
  else element.value = value ?? 0;
}

function createRow(step) {
  const node = document.createElement("article");
  node.className = "task";
  const heading = document.createElement("div");
  heading.className = "task-heading";
  const title = document.createElement("a");
  title.className = "task-title";
  title.target = "_top";
  const status = document.createElement("span");
  status.className = "task-status";
  const bar = document.createElement("progress");
  bar.max = 100;
  const detail = document.createElement("p");
  detail.className = "task-detail";
  heading.append(title, status);
  node.append(heading, bar, detail);
  const row = { node, title, status, bar, detail };
  rows.set(step.key, row);
  return row;
}

export function render(data) {
  document.getElementById("run-state").textContent = data.state;
  document.getElementById("stage-count").textContent = `${data.stages.completed} / ${data.stages.total}`;
  progress(document.getElementById("stage-progress"), data.stages.total ? data.stages.completed / data.stages.total * 100 : 0);
  const normalization = document.getElementById("normalization");
  normalization.hidden = !data.normalization?.total;
  if (!normalization.hidden) normalization.textContent = `Normalization: ${number(data.normalization.completed)} / ${number(data.normalization.total)} captures processed`;
  const steps = document.getElementById("steps");
  const current = new Set(data.steps.map(step => step.key));
  for (const [key, row] of rows) {
    if (!current.has(key)) { row.node.remove(); rows.delete(key); }
  }
  data.steps.forEach((step, index) => {
    const row = rows.get(step.key) ?? createRow(step);
    row.title.textContent = step.label + (step.collection_id == null ? "" : ` · ${step.collection_id}`);
    row.title.href = step.task_url;
    row.status.textContent = step.status_text + (step.stale ? " · stale telemetry" : "");
    Object.assign(row.node.dataset, { status: step.status, paused: String(step.paused), stale: String(step.stale) });
    row.bar.setAttribute("aria-label", row.title.textContent);
    progress(row.bar, step.percent, step.indeterminate);
    row.bar.hidden = ["off", "skipped", "removed"].includes(step.status);
    const counts = step.completed == null ? "" : `${number(step.completed)} / ${step.total == null ? "?" : number(step.total)} ${step.unit}`;
    const percent = step.percent == null ? "" : `${Math.floor(step.percent)}%`;
    const eta = step.paused ? "Paused" : step.stale ? "Waiting for fresh telemetry" : duration(step.remaining_seconds);
    const finished = ["success", "cached", "reused", "off", "skipped", "removed"].includes(step.status);
    row.detail.textContent = [counts, percent, finished ? "" : eta].filter(Boolean).join(" · ") || step.status_text;
    if (steps.children[index] !== row.node) steps.insertBefore(row.node, steps.children[index] ?? null);
  });
}

export function start() {
  const endpoint = location.pathname.replace("/progress/", "/api/progress/");
  const notice = document.getElementById("notice");
  const connection = document.getElementById("connection");
  let timer, controller, stopped = false;
  async function refresh() {
    clearTimeout(timer);
    if (document.hidden || stopped || controller) return;
    controller = new AbortController();
    const timeout = setTimeout(() => controller?.abort(), 10000);
    try {
      const response = await fetch(endpoint, { cache: "no-store", credentials: "same-origin", signal: controller.signal });
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) throw new Error("Sign in to Airflow and check access to this run.");
        if (response.status === 404) throw new Error("Waiting for this run’s first progress report…");
        throw new Error("Progress is temporarily unavailable. Retrying…");
      }
      const data = await response.json();
      render(data);
      notice.hidden = true;
      connection.textContent = `Updated ${new Date(data.updated_at).toLocaleTimeString()}`;
      document.getElementById("steps").removeAttribute("data-disconnected");
    } catch (error) {
      if (!stopped && !document.hidden) {
        notice.textContent = error.name === "AbortError" ? "Progress request timed out. Retrying…" : error.message;
        notice.hidden = false;
        connection.textContent = "Reconnecting · displayed values may be outdated";
        document.getElementById("steps").setAttribute("data-disconnected", "true");
        // Freeze unknown-total animation when the connection stops supplying data.
        for (const row of rows.values()) if (!row.bar.hasAttribute("value")) row.bar.value = 0;
      }
    } finally {
      clearTimeout(timeout);
      controller = null;
      if (!document.hidden && !stopped) timer = setTimeout(refresh, 5000);
    }
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { clearTimeout(timer); controller?.abort(); }
    else refresh();
  });
  window.addEventListener("pagehide", () => { stopped = true; clearTimeout(timer); controller?.abort(); }, { once: true });
  refresh();
}

if (typeof document !== "undefined") start();
