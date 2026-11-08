import assert from "node:assert/strict";
import { test } from "node:test";
import { render, start, duration } from "../airflow/plugins/credleaf_ui/static/progress.js";

// Minimal DOM contract for checking node reuse and polling without a browser.
class Element {
  children = [];
  dataset = {};
  attributes = new Map();
  textContent = "";
  set value(value) { this.attributes.set("value", value); }
  get value() { return this.attributes.get("value"); }
  setAttribute(key, value) { this.attributes.set(key, value); }
  removeAttribute(key) { this.attributes.delete(key); }
  hasAttribute(key) { return this.attributes.has(key); }
  append(...elements) { elements.forEach(element => this.insertBefore(element, null)); }
  insertBefore(element, before) {
    element.remove();
    const index = before == null ? this.children.length : this.children.indexOf(before);
    this.children.splice(index, 0, element);
    element.parent = this;
  }
  remove() {
    if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1);
    this.parent = null;
  }
}

const elements = Object.fromEntries(["run-state", "stage-count", "stage-progress", "normalization", "steps", "notice", "connection"].map(id => [id, new Element()]));
const events = {};
globalThis.document = {
  hidden: false,
  createElement: () => new Element(),
  getElementById: id => elements[id],
  addEventListener: (name, callback) => { events[name] = callback; },
};
globalThis.window = { addEventListener: (name, callback) => { events[name] = callback; } };
const step = { key: "normalize__13559", label: "Normalize", collection_id: 13559, status: "running", status_text: "running", completed: 3, total: 10, percent: 30, unit: "captures", remaining_seconds: 90, paused: false, stale: false, task_url: "/dags/build/runs/test/tasks/normalize" };
const snapshot = { state: "running", stages: { completed: 1, total: 7 }, normalization: { completed: 3, total: 10 }, steps: [step], updated_at: "2026-11-07T12:00:00Z" };

test("progress polls update the same row and preserve explicit paused/unknown state", () => {
  render(snapshot);
  const row = elements.steps.children[0];
  const bar = row.children[1];
  assert.equal(bar.value, 30);
  render({ ...snapshot, steps: [{ ...step, completed: 8, percent: 80 }] });
  assert.equal(elements.steps.children.length, 1);
  assert.equal(elements.steps.children[0], row);
  assert.equal(bar.value, 80);
  assert.match(row.children[2].textContent, /8 \/ 10 captures/);
  render({ ...snapshot, steps: [{ ...step, percent: null, indeterminate: true, total: null }] });
  assert.equal(bar.hasAttribute("value"), false);
  render({ ...snapshot, steps: [{ ...step, paused: true, remaining_seconds: null }] });
  assert.equal(row.dataset.paused, "true");
  assert.match(row.children[2].textContent, /Paused/);
  assert.equal(duration(null), "ETA unavailable");
});

test("polling uses the run endpoint, recovers without replacing rows, and stops on page exit", async () => {
  const originalSetTimeout = globalThis.setTimeout;
  const originalClearTimeout = globalThis.clearTimeout;
  const timers = new Map();
  let id = 0;
  globalThis.setTimeout = (callback, delay) => { timers.set(++id, { callback, delay }); return id; };
  globalThis.clearTimeout = key => timers.delete(key);
  globalThis.location = { pathname: "/credleaf/progress/build/manual__2026-11-07T12:00:00+00:00" };
  let request = 0;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/credleaf/api/progress/build/manual__2026-11-07T12:00:00+00:00");
    assert.equal(options.credentials, "same-origin");
    request++;
    if (request === 2) throw new Error("Offline");
    return { ok: true, json: async () => snapshot };
  };
  const flush = () => new Promise(resolve => setImmediate(resolve));
  try {
    start();
    await flush();
    const row = elements.steps.children[0];
    assert.equal(elements.notice.hidden, true);
    await [...timers.values()].find(timer => timer.delay === 5000).callback();
    assert.match(elements.connection.textContent, /Reconnecting/);
    assert.equal(elements.steps.children[0], row);
    await [...timers.values()].find(timer => timer.delay === 5000).callback();
    assert.equal(elements.notice.hidden, true);
    assert.equal(elements.steps.children[0], row);
    events.pagehide();
    assert.equal(timers.size, 0);
  } finally {
    globalThis.setTimeout = originalSetTimeout;
    globalThis.clearTimeout = originalClearTimeout;
  }
});
