// Runs the panel page's inline <script> without a browser, against a small hand-written DOM and a fetch stub that
// serves JSON recorded from the real server (host/tests/test_web_live.py writes it). Node only, no packages.
//
//   node page_harness.js <index.html> <fixtures.json>
//
// Prints one line per check and exits 1 if any check failed. What it cannot see: layout, CSS, colours, what a
// real browser does with focus, user activation, real Web Audio, real timing. See test_web_live.py.
"use strict";
const fs = require("fs");
const vm = require("vm");

const htmlPath = process.argv[2], fixPath = process.argv[3];
const HTML = fs.readFileSync(htmlPath, "utf8");
const FIX = JSON.parse(fs.readFileSync(fixPath, "utf8"));

let failures = 0, passes = 0;
function check(name, ok, detail) {
  if (ok) { passes++; console.log("ok   " + name); }
  else { failures++; console.log("FAIL " + name + (detail !== undefined ? "  -- " + JSON.stringify(detail) : "")); }
}

// ---------------------------------------------------------------- the HTML: elements with an id, the script
const scripts = [...HTML.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
check("exactly one inline script", scripts.length === 1, scripts.length);
const SCRIPT = scripts[0];
const BODY = HTML.slice(0, HTML.indexOf("<script>"));

function parseTags() {
  const out = [];
  for (const m of BODY.matchAll(/<([a-zA-Z][a-zA-Z0-9]*)(\s[^<>]*)?>/g)) {
    const attrs = {};
    for (const a of (m[2] || "").matchAll(/([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:\s*=\s*"([^"]*)")?/g)) attrs[a[1].toLowerCase()] = a[2] === undefined ? "" : a[2];
    out.push({ tag: m[1].toUpperCase(), attrs });
  }
  return out;
}
const TAGS = parseTags();
const IDS = {};
TAGS.forEach(t => { if (t.attrs.id !== undefined) { if (IDS[t.attrs.id]) check("id is unique: " + t.attrs.id, false); IDS[t.attrs.id] = t; } });

// ---------------------------------------------------------------- static checks
(function staticChecks() {
  const used = new Set();
  for (const m of SCRIPT.matchAll(/\$\("([^"]+)"\)/g)) used.add(m[1]);
  for (const m of SCRIPT.matchAll(/setButton\("([^"]+)", "([^"]+)"/g)) { used.add(m[1]); used.add(m[2]); }
  for (const m of SCRIPT.matchAll(/setMsg\("([^"]+)"/g)) used.add(m[1]);
  for (const m of SCRIPT.matchAll(/act\("([^"]+)", "([^"]+)"/g)) { used.add(m[1]); used.add(m[2]); }
  // ids built by concatenation in the script
  ["board", "sim", "sim-live", "off", "ping", "notes", "rec", "ba-rec", "ba-play", "live-start", "live-stop", "live-scale",
   "live-rec"].forEach(k => { used.add("btn-" + k); used.add("why-" + k); });
  ["live-level", "live-conf", "live-cents"].forEach(k => { used.add(k); used.add(k + "-bar"); });
  const missing = [...used].filter(id => !IDS[id]);
  check("every id the script looks up exists in the HTML (" + used.size + " ids)", missing.length === 0, missing);
  const a = id => (IDS[id] || { attrs: {} }).attrs;
  check("initial source = fake microphone", "checked" in a("live-src-fake") && !("checked" in a("live-src-mic")));
  check("both source radios share one name", a("live-src-fake").name === a("live-src-mic").name && !!a("live-src-fake").name);
  check("initial board speaker = off", a("live-sing").type === "checkbox" && !("checked" in a("live-sing")));
  check("initial volume = 30, range 0..100", a("live-vol").value === "30" && a("live-vol").min === "0" && a("live-vol").max === "100");
  check("initial volume number shows 30", /id="live-vol-num">30</.test(BODY));
  check("this computer's sound starts unticked", a("live-mac-sound").type === "checkbox" && !("checked" in a("live-mac-sound")));
  check("AudioContext is created only inside the tick handler",
        (SCRIPT.match(/new AC\(/g) || []).length === 1 &&
        SCRIPT.indexOf("new AC(") > SCRIPT.indexOf('$("live-mac-sound").addEventListener("change"'));
  check("the live section has the 马上 chip", /<section id="sec-live">[\s\S]*?class="chip lane-now">马上</.test(BODY));
  check("sec-live comes right after sec-link", BODY.indexOf('id="sec-live"') > BODY.indexOf('id="sec-link"') &&
        BODY.indexOf('id="sec-live"') < BODY.indexOf('id="sec-notes"'));
  const ALLOWED = ["#F6F3EC", "#1E1C19", "#6B665D", "#D6D0C3", "#FBF9F4", "#A8261B", "#2456C9", "#D9600F", "#1C6F5A",
                   "#1B1A17", "#ECE7DC", "#A39D91", "#3D3A34", "#232220", "#F08A7E", "#7C9DF2", "#F29552", "#4DBA98"];
  const extraColours = [...HTML.matchAll(/#[0-9A-Fa-f]{6}\b/g)].map(m => m[0].toUpperCase()).filter(c => !ALLOWED.includes(c));
  check("no new colour in the page (only the page's own palette)", extraColours.length === 0, extraColours);
})();

// ---------------------------------------------------------------- a strict, small DOM
class TextNode {
  constructor(t) { this.nodeType = 3; this._t = String(t); this.parentNode = null; }
  get textContent() { return this._t; }
  set textContent(v) { this._t = String(v); }
}

class Element {
  constructor(tag, attrs, doc) {
    this.nodeType = 1;
    this.tagName = tag;
    this.doc = doc;
    this.children = [];
    this.parentNode = null;
    this.listeners = {};
    this.style = {};
    this.className = (attrs && attrs.class) || "";
    this.id = (attrs && attrs.id) || "";
    this.hidden = !!(attrs && "hidden" in attrs);
    this.disabled = !!(attrs && "disabled" in attrs);
    this.type = (attrs && attrs.type) || "";
    this.name = (attrs && attrs.name) || "";
    this.min = attrs && attrs.min; this.max = attrs && attrs.max;
    this._checked = !!(attrs && "checked" in attrs);
    this.title = "";
    if (tag === "SELECT") {
      this._sel = undefined;
      Object.defineProperty(this, "value", {
        get() {
          const opts = this.children.filter(c => c.tagName === "OPTION");
          if (this._sel === null) return "";
          const hit = opts.find(o => String(o.value) === this._sel);
          if (hit) return String(hit.value);
          return opts.length ? String(opts[0].value) : "";
        },
        set(v) {
          const opts = this.children.filter(c => c.tagName === "OPTION");
          this._sel = opts.some(o => String(o.value) === String(v)) ? String(v) : null;
        }
      });
    } else {
      this.value = (attrs && attrs.value !== undefined) ? attrs.value : "";
    }
    this.classList = {
      add: (...c) => { const s = new Set(this.className.split(/\s+/).filter(Boolean)); c.forEach(x => s.add(x)); this.className = [...s].join(" "); },
      remove: (...c) => { this.className = this.className.split(/\s+/).filter(x => x && !c.includes(x)).join(" "); },
      contains: c => this.className.split(/\s+/).includes(c),
      toggle: (c, on) => { const has = this.classList.contains(c); if (on === undefined) on = !has; on ? this.classList.add(c) : this.classList.remove(c); return on; }
    };
  }
  get checked() { return this._checked; }
  set checked(v) {
    this._checked = !!v;
    if (v && this.type === "radio" && this.name) {
      Object.values(this.doc.byId).forEach(e => { if (e !== this && e.type === "radio" && e.name === this.name) e._checked = false; });
    }
  }
  get textContent() { return (this._text || "") + this.children.map(c => c.textContent).join(""); }
  set textContent(v) { this.children.forEach(c => { c.parentNode = null; }); this.children = []; this._text = String(v); }
  get firstChild() { return this.children[0] || null; }
  get nextSibling() {
    const p = this.parentNode;
    if (!p) return null;
    const i = p.children.indexOf(this);
    return p.children[i + 1] || null;
  }
  _adopt(n) { if (n.parentNode) n.parentNode.removeChild(n); n.parentNode = this; }
  appendChild(n) { this._adopt(n); this.children.push(n); return n; }
  removeChild(n) { const i = this.children.indexOf(n); if (i < 0) throw new Error("removeChild: not a child"); this.children.splice(i, 1); n.parentNode = null; return n; }
  insertBefore(n, ref) {
    if (ref === null || ref === undefined) return this.appendChild(n);
    this._adopt(n);
    const i = this.children.indexOf(ref);
    if (i < 0) throw new Error("insertBefore: reference is not a child");
    this.children.splice(i, 0, n);
    return n;
  }
  replaceChild(n, old) {
    const i = this.children.indexOf(old);
    if (i < 0) throw new Error("replaceChild: not a child");
    this._adopt(n);
    const j = this.children.indexOf(old);
    this.children[j] = n;
    old.parentNode = null;
    return old;
  }
  replaceChildren(...nodes) { this.children.forEach(c => { c.parentNode = null; }); this.children = []; this._text = ""; nodes.forEach(n => this.appendChild(n)); }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatch(type) { (this.listeners[type] || []).forEach(fn => fn.call(this, { type, target: this })); }
  click() {
    if (this.disabled) return false;
    if (this.type === "checkbox") { this.checked = !this.checked; this.dispatch("click"); this.dispatch("change"); return true; }
    if (this.type === "radio") { if (!this.checked) { this.checked = true; this.dispatch("change"); } return true; }
    this.dispatch("click");
    return true;
  }
  walk(fn) { fn(this); this.children.forEach(c => { if (c.nodeType === 1) c.walk(fn); }); }
}

function makeDocument() {
  const doc = { byId: {} };
  TAGS.forEach(t => { if (t.attrs.id !== undefined) doc.byId[t.attrs.id] = new Element(t.tag, t.attrs, doc); });
  doc.getElementById = function (id) {
    if (doc.byId[id]) return doc.byId[id];
    let hit = null;                               // elements the script built and put into the page
    Object.values(doc.byId).forEach(e => e.walk(x => { if (!hit && x.id === id && x !== e) hit = x; }));
    return hit;
  };
  doc.createElement = tag => new Element(String(tag).toUpperCase(), {}, doc);
  doc.createTextNode = t => new TextNode(t);
  return doc;
}

// ---------------------------------------------------------------- one page instance with fake time and network
function makePage(routes) {
  const doc = makeDocument();
  const errors = [];
  const timers = [];
  let now = 1000000, tid = 1;
  const pending = [];
  const calls = [];
  const inflight = { poll: 0, maxPoll: 0 };
  const audio = { made: 0, started: 0, stopped: 0 };

  function AC() {
    audio.made++;
    this.state = "suspended"; this.currentTime = 0; this.destination = {};
    this.resume = () => { this.state = "running"; return Promise.resolve(); };
    this.createOscillator = () => ({ type: "", frequency: { value: 0 }, connect() {}, start() { audio.started++; }, stop() { audio.stopped++; } });
    this.createGain = () => ({ gain: { value: 0, setValueAtTime() {}, linearRampToValueAtTime() {}, cancelScheduledValues() {} }, connect() {} });
  }

  function fetchStub(path, opt) {
    const method = (opt && opt.method) || "GET";
    let body = null;
    try { body = opt && opt.body ? JSON.parse(opt.body) : null; } catch (e) { body = opt.body; }
    calls.push({ method, path, body });
    const isPoll = path.startsWith("/api/live/poll");
    if (isPoll) { inflight.poll++; inflight.maxPoll = Math.max(inflight.maxPoll, inflight.poll); }
    return new Promise(resolve => {
      pending.push(() => {
        if (isPoll) inflight.poll--;
        const key = method + " " + path.split("?")[0];
        let r = routes[key];
        if (typeof r === "function") r = r(body, path);
        if (r === undefined) r = { status: 404, json: { error: "not found" } };
        if (r.network) { resolve(Promise.reject(new TypeError("Failed to fetch"))); return; }
        const status = r.status || 200;
        const text = r.raw !== undefined ? r.raw : JSON.stringify(r.json);
        resolve({ ok: status >= 200 && status < 300, status, text: () => Promise.resolve(text) });
      });
    });
  }

  const win = {};
  const ctx = vm.createContext({
    document: doc, window: win, fetch: fetchStub, console,
    setTimeout: (fn, ms) => { const id = tid++; timers.push({ id, at: now + Math.max(0, Number(ms) || 0), fn }); return id; },
    clearTimeout: id => { const i = timers.findIndex(t => t.id === id); if (i >= 0) timers.splice(i, 1); }
  });
  ctx.__now = () => now;
  vm.runInContext("Date.now = function () { return __now(); };", ctx);
  win.AudioContext = undefined;

  function guard(fn) { try { fn(); } catch (e) { errors.push(String(e && e.stack || e)); } }

  async function settle() {
    for (let k = 0; k < 50; k++) {
      await new Promise(r => setImmediate(r));
      if (!pending.length) { await new Promise(r => setImmediate(r)); if (!pending.length) return; }
      const batch = pending.splice(0, pending.length);
      batch.forEach(f => guard(f));
    }
  }

  async function advance(ms) {
    const end = now + ms;
    let fired = 0;
    while (true) {
      await settle();
      timers.sort((a, b) => a.at - b.at || a.id - b.id);
      const t = timers[0];
      if (!t || t.at > end) break;
      // a real page fires at most a few timers per 100 ms; far more means runaway timers (e.g. polls doubling)
      if (++fired > 50 + ms / 5 || timers.length > 200) {
        errors.push(`runaway timers: ${fired} fired, ${timers.length} pending in ${ms} ms of page time`);
        timers.splice(0, timers.length);
        break;
      }
      timers.shift();
      now = Math.max(now, t.at);
      guard(t.fn);
    }
    now = end;
    await settle();
  }

  const page = {
    doc, errors, calls, inflight, audio, ctx, win, AC, settle, advance,
    $: id => doc.getElementById(id),
    text: id => { const e = doc.getElementById(id); return e ? e.textContent : null; },
    clickId: id => { let r; guard(() => { r = doc.getElementById(id).click(); }); return r; },
    start() { guard(() => vm.runInContext(SCRIPT, ctx, { filename: "index.html<script>" })); },
    pollCalls: () => calls.filter(c => c.path.startsWith("/api/live/poll")).length,
    renderFailures() {
      const out = [];
      Object.values(doc.byId).forEach(e => { if (/画不出来/.test(e.textContent)) out.push(e.id + ": " + e.textContent); });
      return out;
    }
  };
  return page;
}

process.on("unhandledRejection", e => { failures++; console.log("FAIL unhandled promise rejection: " + (e && e.stack || e)); });
process.on("uncaughtException", e => { failures++; console.log("FAIL uncaught exception: " + (e && e.stack || e)); });

function routesFor(sc) {
  const r = {
    "GET /api/state": sc.state_fail ? { status: 500, json: { error: "boom" } } : { json: sc.state },
    "GET /api/live/poll": sc.poll_fail ? { status: 500, json: { error: "服务器出错：boom" } } : { json: sc.poll }
  };
  Object.assign(r, sc.extra || {});
  return r;
}

function common(name, p, ticked) {
  check(name + ": nothing threw", p.errors.length === 0, p.errors.slice(0, 3));
  check(name + ": no section reports a drawing failure", p.renderFailures().length === 0, p.renderFailures());
  check(name + ": AudioContext made only if the box was ticked", p.audio.made === (ticked ? 1 : 0), p.audio);
  check(name + ": at most one live poll in flight", p.inflight.maxPoll <= 1, p.inflight);
  // every disabled button shows a reason
  const bad = [];
  Object.values(p.doc.byId).forEach(e => e.walk(x => {
    if (x.tagName === "BUTTON" && x.disabled && x.id) {
      const why = p.doc.getElementById(x.id.replace(/^btn-/, "why-"));
      if (!why || !why.textContent.trim()) bad.push(x.id);
    }
  }));
  check(name + ": every disabled button shows why", bad.length === 0, bad);
}

function noteText(n) { return n === "-" || n === undefined || n === null ? "—" : n; }

// ---------------------------------------------------------------- the situations
async function main() {
  const S = FIX;

  // 1. not connected (state and poll recorded from the real server)
  {
    const p = makePage(routesFor({ state: S.none.state, poll: S.none.poll }));
    p.start(); await p.advance(5000);
    common("not connected", p);
    check("not connected: state line says so and names the live simulator",
          /^未连接/.test(p.text("live-state")) && p.text("live-state").includes("连模拟器（现场版）"), p.text("live-state"));
    check("not connected: 开始听 disabled with a reason", p.$("btn-live-start").disabled && p.text("why-live-start").length > 0);
    check("not connected: 连模拟器（现场版） enabled (simulator built)", !p.$("btn-sim-live").disabled, p.text("why-sim-live"));
    check("not connected: fake-microphone notice shown", !p.$("live-standin").hidden && !p.$("live-standin-fake").hidden && p.$("live-standin-sim").hidden);
    check("not connected: readouts empty", p.text("live-her") === "—" && p.text("live-duet") === "—");
    check("not connected: no fast polling", p.pollCalls() === 0, p.pollCalls());
    check("not connected: board audio section says 未连接", p.text("ba-state") === "未连接", p.text("ba-state"));
    check("not connected: keyboard has 37 keys", p.$("live-kb").children.length === 37, p.$("live-kb").children.length);
    // clicking the disabled start button sends nothing
    p.clickId("btn-live-start"); await p.advance(200);
    check("not connected: a disabled 开始听 sends nothing", !p.calls.some(c => c.path === "/api/live/start"));
  }

  // 2. first firmware (sketch 1: no audio key, no live), recorded from the simulator with --no-audio
  {
    const p = makePage(routesFor({ state: S.sketch1.state, poll: S.sketch1.poll }));
    p.start(); await p.advance(3000);
    common("sketch 1", p);
    const t = p.text("live-state");
    check("sketch 1: state line says no live part, flash duet_live or connect the live simulator",
          t.includes("没有现场部分") && t.includes("duet_live") && t.includes("连模拟器（现场版）"), t);
    check("sketch 1: board audio section still says 第一版固件", p.text("ba-state").includes("第一版固件"), p.text("ba-state"));
    check("sketch 1: 开始听 disabled with a reason", p.$("btn-live-start").disabled && p.text("why-live-start").includes("没有现场部分"));
    check("sketch 1: no fast polling", p.pollCalls() === 0);
  }

  // 3. the plain simulator (sketch 2 stand-in audio): no live part either
  {
    const p = makePage(routesFor({ state: S.plain_sim.state, poll: S.plain_sim.poll }));
    p.start(); await p.advance(3000);
    common("plain sim", p);
    check("plain sim: state line names the firmware and says no live part",
          p.text("live-state").includes("board_sim") && p.text("live-state").includes("没有现场部分"), p.text("live-state"));
    check("plain sim: board audio section shows the stand-in", p.text("ba-state").includes("替身") && !p.$("ba-standin").hidden);
    check("plain sim: 喇叭测试 disabled with a reason", p.$("btn-live-scale").disabled && p.text("why-live-scale").length > 0);
  }

  // 4. the live firmware, idle; then the owner presses 开始听 with the controls untouched
  {
    let started = false;
    const extra = {
      "POST /api/live/start": body => { started = true; return { json: S.start_reply }; },
      "POST /api/live/set": body => {
        // the server now reports the speaker on in both endpoints, as the real one does
        const st = JSON.parse(JSON.stringify(S.live_idle.state));
        st.live.sing = true;
        routes["GET /api/state"] = { json: st };
        routes["GET /api/live/poll"] = { json: S.set_sing_reply };
        return { json: S.set_sing_reply };
      },
      "POST /api/live/stop": () => ({ json: S.stop_reply })
    };
    const routes = routesFor({ state: S.live_idle.state, poll: S.live_idle.poll, extra });
    const idleRoutes = Object.assign({}, routes);
    const p = makePage(routes);
    p.start(); await p.advance(3000);
    common("live idle", p);
    check("live idle: state line says ready", p.text("live-state").startsWith("准备好了") && p.text("live-state").includes("喇叭关着"), p.text("live-state"));
    check("live idle: board audio section points to 现场", p.text("ba-state").includes("现场"), p.text("ba-state"));
    check("live idle: board audio buttons disabled with that reason",
          p.$("btn-ba-rec").disabled && p.text("why-ba-rec").includes("现场") && p.$("btn-ba-play").disabled && p.text("why-ba-play").length > 0,
          [p.text("why-ba-rec"), p.text("why-ba-play")]);
    check("live idle: 开始听 and 停 enabled", !p.$("btn-live-start").disabled && !p.$("btn-live-stop").disabled);
    check("live idle: 喇叭测试 disabled because the speaker box is off",
          p.$("btn-live-scale").disabled && p.text("why-live-scale").includes("喇叭出声"), p.text("why-live-scale"));
    check("live idle: 录成片段 disabled until listening", p.$("btn-live-rec").disabled && p.text("why-live-rec").includes("开始听"));
    check("live idle: simulator note shown", !p.$("live-standin-sim").hidden && !p.$("live-standin-fake").hidden);
    check("live idle: no fast polling while idle", p.pollCalls() === 0, p.pollCalls());
    check("live idle: plug-in select filled, third_below chosen", p.$("live-plugin").value === "third_below", p.$("live-plugin").value);
    check("live idle: key select C", p.$("live-key").value === "0");

    // tick the speaker box while idle: /api/live/set {sing: true}, then the scale button is enabled
    p.clickId("live-sing"); await p.advance(100);
    const setCall = p.calls.find(c => c.path === "/api/live/set");
    check("live idle: ticking the speaker sends set {sing: true}", !!setCall && setCall.body && setCall.body.sing === true, setCall);
    check("live idle: 喇叭测试 enabled once the speaker is on", !p.$("btn-live-scale").disabled, p.text("why-live-scale"));
    await p.advance(3000);
    check("live idle: the box stays on across slow polls (server says sing on)", p.$("live-sing").checked && !p.$("btn-live-scale").disabled);
    check("live idle: state line now says the speaker is on, volume 30", p.text("live-state").includes("喇叭开着，音量 30"), p.text("live-state"));
    Object.assign(routes, idleRoutes);
    // a fresh page for the start click with untouched controls
    const q = makePage(routes);
    q.start(); await q.advance(2000);
    q.clickId("btn-live-start"); await q.advance(50);
    const st = q.calls.find(c => c.path === "/api/live/start");
    check("start with untouched controls sends fake / speaker off / volume 30 / third_below / C",
          !!st && JSON.stringify(st.body) === JSON.stringify({ fake: true, sing: false, volume: 30, plugin: "third_below", tonic: 0 }),
          st && st.body);
    // after the start reply the page polls fast with since = the start's next
    routes["GET /api/state"] = { json: S.listening.state };
    routes["GET /api/live/poll"] = { json: S.listening.poll };
    await q.advance(1500);
    const polls = q.calls.filter(c => c.path.startsWith("/api/live/poll"));
    check("after 开始听 the page polls fast (>= 8 polls in 1.5 s)", polls.length >= 8, polls.length);
    check("the first fast poll asks since = the start reply's next", polls.length > 0 && polls[0].path === "/api/live/poll?since=" + S.start_reply.next,
          polls.length ? polls[0].path : null);
    check("later polls ask since = the last poll's next", polls.length > 1 && polls[polls.length - 1].path === "/api/live/poll?since=" + S.listening.poll.next,
          polls.length ? polls[polls.length - 1].path : null);
    common("live idle -> start", q);
  }

  // 5. listening (recorded while the fake melody ran on the live simulator)
  {
    const routes = routesFor({ state: S.listening.state, poll: S.listening.poll });
    const p = makePage(routes);
    p.win.AudioContext = p.AC;
    p.start(); await p.advance(2000);
    const pl = S.listening.poll;
    check("listening: 你 shows her note", p.text("live-her") === noteText(pl.now.her_name), [p.text("live-her"), pl.now.her_name]);
    check("listening: Duet shows its note", p.text("live-duet") === noteText(pl.now.duet_name), [p.text("live-duet"), pl.now.duet_name]);
    const t = p.text("live-state");
    check("listening: state line says listening with the fake microphone, speaker off, seconds",
          t.startsWith("正在听") && t.includes("假麦克风") && t.includes("喇叭关着") && /已经听了 \d+\.\d 秒/.test(t), t);
    check("listening: the keys of her note and Duet's note are filled",
          p.$("live-kb").children.some(k => /\bher\b/.test(k.className)) && p.$("live-kb").children.some(k => /\bduet\b/.test(k.className)));
    const rows = p.$("live-rows").children;
    check("listening: the table has rows (at most 16)", rows.length > 0 && rows.length <= 16, rows.length);
    const heard = pl.events.filter(e => e.kind === 1);
    const lastHeard = heard[heard.length - 1];
    check("listening: newest row first, her note matches the newest heard event",
          rows.length > 0 && !!lastHeard && rows[0].children[1].textContent === lastHeard.name, rows.length ? rows[0].textContent : null);
    check("listening: Duet's cell is in Duet's colour class", rows.length > 0 && rows[0].children[2].className === "duet");
    check("listening: meters show numbers", /dB$/.test(p.text("live-level")) && /%$/.test(p.text("live-conf")),
          [p.text("live-level"), p.text("live-conf"), p.text("live-cents")]);
    check("listening: counters line filled", /^听到 \d+ 个音 · 唱了 \d+ 个音 · 丢了 0 个事件 · 听了 \d+\.\d 秒$/.test(p.text("live-counts")), p.text("live-counts"));
    check("listening: 录成片段 enabled", !p.$("btn-live-rec").disabled, p.text("why-live-rec"));
    check("listening: board audio buttons disabled with the 现场 reason", p.$("btn-ba-rec").disabled && p.text("why-ba-rec").includes("现场"));
    const n0 = p.pollCalls();
    check("listening: fast polling runs (>= 10 polls in 2 s)", n0 >= 10, n0);
    // this computer's sound: made only on the tick, tones follow; untick stops them
    check("listening: no AudioContext before the tick", p.audio.made === 0);
    p.clickId("live-mac-sound"); await p.advance(300);
    const soundingNow = (pl.now.her >= 0 ? 1 : 0) + (pl.now.duet >= 0 ? 1 : 0);
    check("listening: tick makes one AudioContext and starts the sounding notes",
          p.audio.made === 1 && p.audio.started === soundingNow, p.audio);
    p.clickId("live-mac-sound"); await p.advance(100);
    check("listening: untick stops them", p.audio.stopped === p.audio.started, p.audio);
    // every other control while listening: the request each one sends
    const sent = path => p.calls.filter(c => c.path === path);
    const startBodies = [];
    routes["POST /api/live/start"] = body => { startBodies.push(body); return { json: S.start_reply }; };
    routes["POST /api/live/set"] = () => ({ json: pl });
    routes["POST /api/live/scale"] = () => ({ json: { started: true, notes: ["C4"], seconds: 2.88 } });
    routes["POST /api/live/record"] = body => ({ json: { id: "c1", name: "录音", summary: "125 帧", seconds: body.seconds } });
    routes["POST /api/live/stop"] = () => ({ json: S.stop_reply });
    p.$("live-vol").value = "45";
    p.$("live-vol").dispatch("input");
    check("slider: the number follows while dragging", p.text("live-vol-num") === "45", p.text("live-vol-num"));
    p.$("live-vol").dispatch("change"); await p.advance(100);
    const vs = sent("/api/live/set");
    check("slider: release sends set {volume: 45}", vs.length === 1 && JSON.stringify(vs[0].body) === '{"volume":45}', vs);
    p.clickId("live-sing"); await p.advance(100);
    const ss = sent("/api/live/set");
    check("speaker box while listening sends set {sing: true}", ss.length === 2 && ss[1].body.sing === true, ss);
    check("喇叭测试 enabled with the box on", !p.$("btn-live-scale").disabled, p.text("why-live-scale"));
    p.clickId("btn-live-scale"); await p.advance(100);
    check("喇叭测试 sends /api/live/scale", sent("/api/live/scale").length === 1);
    check("喇叭测试 says what it plays", p.text("live-msg").includes("音阶"), p.text("live-msg"));
    p.clickId("btn-live-rec"); await p.advance(100);
    const rs = sent("/api/live/record");
    check("录成片段 sends record {seconds: 5}", rs.length === 1 && rs[0].body.seconds === 5, rs);
    check("录成片段 reports the clip", p.text("live-rec-msg").includes("录好了"), p.text("live-rec-msg"));
    p.clickId("live-src-mic"); await p.advance(100);
    check("真麦克风 while listening restarts with fake: false",
          startBodies.length === 1 && startBodies[0].fake === false && startBodies[0].volume === 45, startBodies);
    p.$("live-key").value = "7";
    p.$("live-key").dispatch("change"); await p.advance(100);
    check("a key change while listening restarts with that key", startBodies.length === 2 && startBodies[1].tonic === 7, startBodies);
    p.$("live-plugin").value = "echo";
    p.$("live-plugin").dispatch("change"); await p.advance(100);
    check("a model change while listening restarts with that model", startBodies.length === 3 && startBodies[2].plugin === "echo", startBodies);
    p.clickId("btn-live-stop"); await p.advance(100);
    check("停 sends /api/live/stop", sent("/api/live/stop").length === 1);
    // the server stops: fast polling ends
    routes["GET /api/live/poll"] = { json: S.stopped.poll };
    routes["GET /api/state"] = { json: S.stopped.state };
    await p.advance(2000);
    const n1 = p.pollCalls();
    await p.advance(3000);
    check("stopped: fast polling stops", p.pollCalls() === n1, [n1, p.pollCalls()]);
    check("stopped: state line back to ready", p.text("live-state").startsWith("准备好了"), p.text("live-state"));
    common("listening", p, true);
  }

  // 5b. a restart that fails puts the source radio back to what the board does
  {
    const routes = routesFor({ state: S.listening.state, poll: S.listening.poll });
    routes["POST /api/live/start"] = () => ({ status: 400, json: { error: "板子没有回应现场的指令。" } });
    const p = makePage(routes);
    p.start(); await p.advance(1000);
    p.clickId("live-src-mic"); await p.advance(200);
    check("failed restart: the error is shown", p.text("live-msg").includes("板子没有回应"), p.text("live-msg"));
    check("failed restart: the radio is back on the fake microphone", p.$("live-src-fake").checked && !p.$("live-src-mic").checked);
    common("failed restart", p);
  }

  // 5c. a box ticked while not connected does not survive connecting: the controls follow the server
  {
    const routes = routesFor({ state: S.none.state, poll: S.none.poll });
    const p = makePage(routes);
    p.start(); await p.advance(2000);
    p.clickId("live-sing"); await p.advance(100);
    check("not connected: ticking the speaker sends nothing", !p.calls.some(c => c.path === "/api/live/set"));
    routes["GET /api/state"] = { json: S.live_idle.state };
    routes["GET /api/live/poll"] = { json: S.live_idle.poll };
    await p.advance(2000);
    check("after connecting: the speaker box shows the board's state (off)", !p.$("live-sing").checked);
    check("after connecting: 喇叭测试 disabled again", p.$("btn-live-scale").disabled && p.text("why-live-scale").length > 0);
    common("tick then connect", p);
  }

  // 6. hand-made: empty objects from both endpoints (fields missing everywhere)
  {
    const p = makePage(routesFor({ state: {}, poll: {} }));
    p.start(); await p.advance(3000);
    common("empty replies", p);
    check("empty replies: state line says 未连接", /^未连接/.test(p.text("live-state")), p.text("live-state"));
  }

  // 7. hand-made: a linked live state whose live object lacks most fields, and a poll with junk events
  {
    const st = JSON.parse(JSON.stringify(S.listening.state));
    st.live = { available: true, running: true };
    const poll = { available: true, running: true, events: [null, { n: "x" }, { n: 999999, kind: 1 }], now: {}, state: null };
    const p = makePage(routesFor({ state: st, poll }));
    p.start(); await p.advance(2000);
    common("partial fields", p);
    check("partial fields: state line still says listening", p.text("live-state").startsWith("正在听"), p.text("live-state"));
  }

  // 8. the fast poll fails while listening: the message line says so and polling goes on
  {
    const p = makePage(routesFor({ state: S.listening.state, poll_fail: true }));
    p.start(); await p.advance(1000);
    const n0 = p.pollCalls();
    await p.advance(3000);
    common("poll failing", p);
    check("poll failing: the section's message line says so", p.text("live-msg").includes("读现场状态没成功"), p.text("live-msg"));
    check("poll failing: polling keeps going", p.pollCalls() > n0, [n0, p.pollCalls()]);
  }

  // 9. the whole server gone (network error on every request)
  {
    const p = makePage({ "GET /api/state": { network: true }, "GET /api/live/poll": { network: true } });
    p.start(); await p.advance(3000);
    common("server gone", p);
    check("server gone: the page says the server does not answer", p.text("server-msg").includes("面板服务没有回应"), p.text("server-msg"));
  }

  console.log(`${passes} passed, ${failures} failed`);
  process.exit(failures ? 1 : 0);
}

main().catch(e => { console.log("FAIL harness: " + (e && e.stack || e)); process.exit(1); });
