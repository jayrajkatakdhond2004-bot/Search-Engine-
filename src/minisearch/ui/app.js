// minisearch UI: plain JS, no build step. State lives in the URL (?q=&page=&ranker=&op=...),
// so results are linkable and the browser's back/forward buttons work.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const PAGE_SIZE = 10;
  const state = { q: "", page: 1, ranker: "bm25", op: "or", k1: "1.2", b: "0.75", engine: "auto" };
  let inflight = null;

  const fmt = new Intl.NumberFormat();
  const bytes = (n) => {
    if (n == null) return "–";
    const u = ["B", "KB", "MB", "GB"];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
  };
  // Only allow http(s) links; a stored "javascript:" URL must never become clickable.
  const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : null);
  const text = (s) => document.createTextNode(s);

  function readUrl() {
    const p = new URLSearchParams(location.search);
    state.q = p.get("q") || "";
    state.page = Math.max(1, parseInt(p.get("page") || "1", 10) || 1);
    for (const key of ["ranker", "op", "k1", "b", "engine"]) if (p.get(key)) state[key] = p.get(key);
  }

  function writeUrl(push) {
    const p = new URLSearchParams();
    if (state.q) p.set("q", state.q);
    if (state.page > 1) p.set("page", state.page);
    if (state.ranker !== "bm25") p.set("ranker", state.ranker);
    if (state.op !== "or") p.set("op", state.op);
    if (state.k1 !== "1.2") p.set("k1", state.k1);
    if (state.b !== "0.75") p.set("b", state.b);
    if (state.engine !== "auto") p.set("engine", state.engine);
    const url = p.toString() ? `?${p}` : location.pathname;
    (push ? history.pushState : history.replaceState).call(history, null, "", url);
  }

  function syncControls() {
    $("q").value = state.q;
    document.querySelectorAll("[data-ranker]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.ranker === state.ranker));
    document.querySelectorAll("[data-op]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.op === state.op));
    $("k1").value = state.k1; $("k1-out").value = state.k1;
    $("b").value = state.b; $("b-out").value = state.b;
    $("engine").value = state.engine;
    const bm25 = state.ranker === "bm25";
    $("k1").disabled = !bm25; $("b").disabled = !bm25;
  }

  async function run() {
    syncControls();
    const has = state.q.trim().length > 0;
    $("welcome").hidden = has;
    $("error").hidden = true;
    if (!has) { $("results").replaceChildren(); $("pager").replaceChildren(); $("meta").textContent = ""; return; }
    document.title = `${state.q} – minisearch`;

    const params = new URLSearchParams({ q: state.q, page: state.page, size: PAGE_SIZE, ranker: state.ranker, op: state.op, engine: state.engine });
    if (state.ranker === "bm25") { params.set("k1", state.k1); params.set("b", state.b); }
    if (inflight) inflight.abort();
    inflight = new AbortController();
    const t0 = performance.now();
    let data;
    try {
      const res = await fetch(`/search?${params}`, { signal: inflight.signal });
      data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "request failed");
    } catch (err) {
      if (err.name === "AbortError") return;
      $("results").replaceChildren(); $("pager").replaceChildren(); $("meta").textContent = "";
      $("error").textContent = err.message; $("error").hidden = false;
      return;
    }
    renderMeta(data, performance.now() - t0);
    renderResults(data);
    renderPager(data);
  }

  function renderMeta(d, roundTrip) {
    const meta = $("meta");
    const totalLabel = `${d.total_is_exact ? "" : "≥ "}${fmt.format(d.total)} result${d.total === 1 ? "" : "s"}`;
    const rk = d.ranker.name === "bm25" ? `BM25 (k1=${d.ranker.k1}, b=${d.ranker.b})` : "TF-IDF";
    meta.replaceChildren(
      text(`${totalLabel} · ${d.search_ms.toFixed(2)} ms search, ${d.took_ms.toFixed(1)} ms total, ${roundTrip.toFixed(0)} ms round trip · ${rk} · `),
      Object.assign(document.createElement("code"), { textContent: d.parsed, title: "Parsed and analysed query" })
    );
  }

  function renderResults(d) {
    const list = $("results");
    if (!d.results.length) {
      const li = document.createElement("li");
      li.className = "result";
      li.textContent = "No documents match this query.";
      list.replaceChildren(li);
      return;
    }
    list.start = d.results[0].rank;
    list.replaceChildren(...d.results.map((r) => {
      const li = document.createElement("li");
      li.className = "result";
      const href = safeUrl(r.url) || `/documents/${encodeURIComponent(r.id)}`;
      const url = document.createElement("div");
      url.className = "url";
      url.textContent = r.url || `doc:${r.id}`;
      const h2 = document.createElement("h2");
      const a = document.createElement("a");
      a.href = href;
      a.rel = "noopener";
      a.innerHTML = r.title_html; // server-escaped; only <mark> tags are added
      h2.append(a);
      const snip = document.createElement("p");
      snip.className = "snippet";
      snip.innerHTML = r.snippet; // server-escaped; only <mark> tags are added
      const score = document.createElement("div");
      score.className = "score";
      score.textContent = `#${r.rank} · score ${r.score.toFixed(4)}`;
      li.append(url, h2, snip, score);
      return li;
    }));
  }

  function renderPager(d) {
    const pager = $("pager");
    const pages = Math.min(Math.ceil(d.total / d.size), 1000);
    if (pages <= 1) { pager.replaceChildren(); return; }
    const link = (label, page, cls = "") => {
      if (page < 1 || page > pages) return Object.assign(document.createElement("span"), { className: "disabled", textContent: label });
      if (page === d.page) return Object.assign(document.createElement("span"), { className: "current", textContent: label, ariaCurrent: "page" });
      const a = Object.assign(document.createElement("a"), { href: "#", textContent: label, className: cls });
      a.addEventListener("click", (e) => { e.preventDefault(); state.page = page; writeUrl(true); run(); window.scrollTo({ top: 0 }); });
      return a;
    };
    const items = [link("‹ Prev", d.page - 1)];
    const from = Math.max(1, d.page - 4), to = Math.min(pages, from + 8);
    for (let p = from; p <= to; p++) items.push(link(String(p), p));
    items.push(link("Next ›", d.page + 1));
    pager.replaceChildren(...items);
  }

  async function loadStats() {
    try {
      const s = await (await fetch("/stats")).json();
      const ratio = s.compression_ratio ? ` · postings compressed ${s.compression_ratio}×` : "";
      $("stats").textContent = `${fmt.format(s.documents)} documents · ${fmt.format(s.terms)} terms · ` +
        `index ${bytes(s.index_bytes)} · ${s.segments} segment${s.segments === 1 ? "" : "s"}${ratio}`;
    } catch { $("stats").textContent = ""; }
  }

  // --- wiring ---
  $("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    state.q = $("q").value; state.page = 1; writeUrl(true); run();
  });
  document.querySelectorAll("[data-ranker]").forEach((b) => b.addEventListener("click", () => {
    state.ranker = b.dataset.ranker; state.page = 1; writeUrl(false); run();
  }));
  document.querySelectorAll("[data-op]").forEach((b) => b.addEventListener("click", () => {
    state.op = b.dataset.op; state.page = 1; writeUrl(false); run();
  }));
  for (const key of ["k1", "b"]) {
    $(key).addEventListener("input", () => { $(`${key}-out`).value = $(key).value; });
    $(key).addEventListener("change", () => { state[key] = $(key).value; state.page = 1; writeUrl(false); run(); });
  }
  $("engine").addEventListener("change", () => { state.engine = $("engine").value; writeUrl(false); run(); });
  $("help-toggle").addEventListener("click", () => {
    const help = $("help"); help.hidden = !help.hidden;
    $("help-toggle").setAttribute("aria-expanded", String(!help.hidden));
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== $("q")) { e.preventDefault(); $("q").focus(); $("q").select(); }
  });
  window.addEventListener("popstate", () => { readUrl(); run(); });

  readUrl();
  run();
  loadStats();
})();
