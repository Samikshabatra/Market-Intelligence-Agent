/* Market Intelligence Agent - interface.
   Hash routing, one fetch layer, and views that render from the run snapshot the
   server streams. No framework: the state is small and the shapes are stable. */

const SECTION_LABELS = {
  company_overview: "Company overview",
  positioning: "Positioning",
  pricing: "Pricing",
  recent_moves: "Recent moves",
  strengths: "Strengths",
  weaknesses: "Weaknesses",
  market_signals: "Market signals",
};

const STATUS_COPY = {
  grounded: "Grounded",
  unverified: "Unverified",
  conflicting: "Conflicting",
  insufficient_data: "No data",
};

const FOCUS_AREAS = ["Pricing", "Positioning", "Product", "Funding", "Reviews", "Hiring"];

const state = {
  config: null,
  depth: "standard",
  focus: new Set(),
  stream: null,
  tab: "summary",
  activeClaim: null,
  timer: null,
  startedAt: null,
};

/* ------------------------------------------------------------------ utils */

const el = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmtSeconds(ms) {
  if (ms == null) return "--";
  return `${(ms / 1000).toFixed(1)}s`;
}

function fmtClock(ms) {
  const total = Math.max(0, ms) / 1000;
  const mins = String(Math.floor(total / 60)).padStart(2, "0");
  const secs = (total % 60).toFixed(1).padStart(4, "0");
  return `${mins}:${secs}`;
}

function fmtWhen(iso) {
  if (!iso) return "";
  const then = new Date(iso);
  const mins = Math.round((Date.now() - then.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  if (mins < 1440) return `${Math.round(mins / 60)} h ago`;
  return then.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function meterClass(confidence, status) {
  if (status === "conflicting") return "meter__fill meter__fill--alert";
  if (status === "unverified" || confidence < 0.55) return "meter__fill meter__fill--warn";
  return "meter__fill";
}

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function toast(message) {
  const node = document.createElement("div");
  node.className = "toast";
  node.setAttribute("role", "status");
  node.textContent = message;
  document.body.appendChild(node);
  setTimeout(() => node.remove(), 3200);
}

function navigate(hash) {
  if (location.hash === hash) render();
  else location.hash = hash;
}

/* ------------------------------------------------------------------ views */

function viewHome(runs) {
  const recent = runs.filter((r) => r.status === "done").slice(0, 3);
  return `
    <section class="hero">
      <div>
        <h1 class="hero__title">Market<br>intelligence,<br>automated.</h1>
        <p class="hero__lede">
          Ask a market question. The agent plans its own search, reads across public
          sources, and returns a competitor brief where every claim points at the page
          it came from.
        </p>
        <form class="ask" id="askForm">
          <input id="askInput" placeholder="Compare Notion and Coda in enterprise SaaS"
                 aria-label="Market or competitor question" required minlength="3">
          <button class="btn btn--primary" type="submit">
            Start research
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
          </button>
        </form>
        <div class="statgrid">
          <div><div class="stat__value">5+</div><div class="stat__label">Distinct domains per brief</div></div>
          <div><div class="stat__value">&lt; 60s</div><div class="stat__label">Latency budget, enforced per stage</div></div>
          <div><div class="stat__value">100%</div><div class="stat__label">Claims tied to a retrieved passage</div></div>
          <div><div class="stat__value">1</div><div class="stat__label">Fallback round when evidence is thin</div></div>
        </div>
      </div>
      <div id="contourHost"></div>
    </section>

    <section style="margin-top:34px">
      <div class="row" style="margin-bottom:10px">
        <span class="eyebrow">Recent research</span>
        <button class="btn btn--ghost btn--sm" data-nav="history" style="margin-left:auto">View all</button>
      </div>
      <div class="panel">
        ${recent.length ? `<table class="rows"><tbody>${recent.map(rowFor).join("")}</tbody></table>`
          : `<div class="empty"><div class="empty__title">No research yet</div>
             <p>Ask a question above and the agent will build your first brief.</p></div>`}
      </div>
    </section>`;
}

function rowFor(run) {
  return `
    <tr data-run="${esc(run.id)}">
      <td>
        <div>${esc(run.query)}</div>
        <div class="muted" style="font-size:11.5px;margin-top:2px">${fmtWhen(run.created_at)}</div>
      </td>
      <td class="num muted">${run.sources ?? 0} sources</td>
      <td class="num">${run.confidence != null
        ? `<span style="color:var(--signal)">${Math.round(run.confidence * 100)}% confidence</span>`
        : `<span class="muted">--</span>`}</td>
      <td class="num muted">${fmtSeconds(run.latency_ms)}</td>
      <td class="num muted">${run.flags ? `${run.flags} flags` : ""}</td>
    </tr>`;
}

function viewNew() {
  const depths = state.config?.depths || {};
  const describe = (name) => {
    const preset = depths[name];
    return preset ? `~${preset.seconds}s budget &middot; ${preset.steps} search steps` : "";
  };
  return `
    <section class="compose">
      <h1 style="font-size:26px;letter-spacing:-0.03em">New research</h1>
      <p class="muted" style="margin:6px 0 20px">What do you want to know?</p>

      <form id="composeForm" class="stack">
        <textarea class="field" id="composeQuery" rows="4" required minlength="3"
          placeholder="Compare Linear and Jira on positioning, pricing, recent product moves and enterprise strategy."></textarea>

        <div>
          <span class="eyebrow">Research depth</span>
          <div class="depth" role="group" aria-label="Research depth">
            ${["quick", "standard", "deep"].map((name) => `
              <button type="button" class="depth__opt" data-depth="${name}"
                      aria-pressed="${state.depth === name}">
                <span class="depth__name">${name[0].toUpperCase() + name.slice(1)}</span>
                <span class="depth__meta">${describe(name)}</span>
              </button>`).join("")}
          </div>
        </div>

        <div>
          <span class="eyebrow">Focus areas <span class="muted" style="text-transform:none;letter-spacing:0">- optional</span></span>
          <div class="chips" role="group" aria-label="Focus areas">
            ${FOCUS_AREAS.map((area) => `
              <button type="button" class="chip" data-focus="${area}"
                      aria-pressed="${state.focus.has(area)}">${area}</button>`).join("")}
          </div>
        </div>

        <div class="row" style="margin-top:6px">
          <button class="btn btn--primary" type="submit">Begin research</button>
          <span class="muted" style="font-size:12px">
            The agent plans the steps itself and finds the sources.
          </span>
        </div>
      </form>
    </section>`;
}

function viewProgress(run) {
  const elapsed = state.startedAt ? Date.now() - state.startedAt : 0;
  return `
    <section class="progress">
      <div class="stack">
        <div class="panel">
          <div class="panel__head">
            <div>
              <div class="panel__title">Researching</div>
              <div class="muted" style="font-size:12px;margin-top:2px">${esc(run.query)}</div>
            </div>
            <div class="panel__tools" style="text-align:right">
              <div>
                <div class="eyebrow">Time elapsed</div>
                <div class="clock" id="clock">${fmtClock(elapsed)}</div>
              </div>
            </div>
          </div>
          <ul class="stagelist">
            ${run.stages.map((stage) => `
              <li data-status="${stage.status}">
                <span class="stage__dot"></span>
                <span>
                  <span class="stage__label">${esc(stage.label)}</span>
                  ${stageDetail(stage)}
                </span>
                <span class="stage__status">${
                  stage.status === "done" ? "Completed"
                  : stage.status === "active" ? "In progress"
                  : stage.status === "failed" ? "Failed" : "Pending"}</span>
              </li>`).join("")}
          </ul>
        </div>
        ${run.error ? `<div class="panel panel--pad" style="border-color:var(--alert)">
          <div class="eyebrow" style="color:var(--alert)">Run failed</div>
          <p style="margin-top:6px">${esc(run.error)}</p>
          <button class="btn" data-nav="new" style="margin-top:12px">Start over</button>
        </div>` : ""}
      </div>

      <div class="panel trace" aria-label="Research trace">
        <span class="eyebrow" style="margin-bottom:6px">Research trace</span>
        ${run.stages.map((stage, index) => `
          <div class="trace__node" data-status="${stage.status}">
            <span class="trace__label">${esc(stage.key)}</span>
          </div>
          ${index < run.stages.length - 1 ? '<div class="trace__link"></div>' : ""}
        `).join("")}
      </div>
    </section>`;
}

function stageDetail(stage) {
  const d = stage.detail || {};
  const bits = [];
  if (d.subject) bits.push(esc(d.subject));
  if (d.sub_questions) bits.push(`${d.sub_questions.length} steps planned`);
  if (d.domains) bits.push(`${d.domains} domains`);
  if (d.sources) bits.push(`${d.sources} sources`);
  if (d.fallback_rounds) bits.push(`${d.fallback_rounds} fallback round`);
  if (d.sections_asserted != null) bits.push(`${d.sections_asserted}/7 sections asserted`);
  return bits.length ? `<div class="stage__detail">${bits.join(" &middot; ")}</div>` : "";
}

function viewBrief(run) {
  const result = run.result;
  if (!result) return `<div class="empty"><div class="empty__title">This run has no brief.</div></div>`;
  const detail = result.detail || {};
  const sections = detail.sections || {};
  const sources = detail.sources || [];
  const entries = Object.entries(result.brief);
  const asserted = entries.filter(([name]) => (sections[name]?.status || "grounded") === "grounded"
    && result.brief[name].text);
  const confidences = asserted.map(([, s]) => s.confidence);
  const overall = confidences.length
    ? Math.round((confidences.reduce((a, b) => a + b, 0) / confidences.length) * 100) : 0;
  const domains = new Set(sources.map((s) => s.domain)).size;
  const cited = new Set(entries.flatMap(([, s]) => s.citations)).size;

  return `
    <section>
      <div class="brief__head">
        <div>
          <h1 class="brief__title">${esc(result.query)}</h1>
          <div class="brief__sub">
            Generated ${fmtWhen(run.created_at)} &middot; ${fmtSeconds(result.latency_ms)}
            ${detail.fallback_rounds ? ` &middot; ${detail.fallback_rounds} fallback round` : ""}
          </div>
        </div>
        <div class="brief__metrics">
          <div><div class="metric__value">${overall}%</div><div class="metric__label">Confidence</div></div>
          <div><div class="metric__value">${domains}</div><div class="metric__label">Domains</div></div>
          <div><div class="metric__value">${cited}/${sources.length}</div><div class="metric__label">Cited</div></div>
        </div>
      </div>

      <div class="tabs" role="tablist">
        ${[["summary", "Executive summary"], ["evidence", "Evidence"],
           ["sources", "Sources"], ["trace", "Search plan"]].map(([key, label]) => `
          <button class="tab" role="tab" data-tab="${key}"
                  aria-selected="${state.tab === key}">${label}</button>`).join("")}
      </div>

      <div id="tabPanel">${renderTab(run)}</div>
    </section>`;
}

function renderTab(run) {
  const result = run.result;
  const detail = result.detail || {};
  if (state.tab === "summary") return renderSummary(result, detail);
  if (state.tab === "evidence") return renderEvidence(result, detail);
  if (state.tab === "sources") return renderSources(detail.sources || []);
  return renderTrace(result, detail);
}

function renderSummary(result, detail) {
  const sections = detail.sections || {};
  const sourceById = Object.fromEntries((detail.sources || []).map((s) => [s.source_id, s]));
  const cards = Object.entries(result.brief).map(([name, section]) => {
    const status = sections[name]?.status || "grounded";
    const pct = Math.round((section.confidence || 0) * 100);
    const long = (section.text || "").length > 420;
    const body = section.text
      ? `<p class="section__text${long ? " section__text--clamped" : ""}">${esc(section.text)}</p>
         ${long ? '<button class="section__more" data-expand>Show full text</button>' : ""}`
      : `<p class="section__text muted">${
          status === "conflicting"
            ? "Sources disagree on this. Both figures are cited rather than one being chosen."
            : "No retrieved source supports this section."}</p>`;
    return `
      <article class="panel panel--pad">
        <div class="section__head">
          <span class="section__name">${SECTION_LABELS[name] || name}</span>
          <span class="badge badge--${status}" style="margin-left:auto">${STATUS_COPY[status]}</span>
        </div>
        ${body}
        <div class="section__foot">
          <div class="meter"><div class="${meterClass(section.confidence, status)}"
               style="width:${pct}%"></div></div>
          <span class="num">${pct}%</span>
          <span style="display:flex;gap:4px">
            ${section.citations.map((id) => `<button class="cite" data-src="${esc(id)}"
              title="${esc(sourceById[id]?.domain || id)}">${esc(id)}</button>`).join("")}
          </span>
        </div>
      </article>`;
  }).join("");

  const flags = result.unverified_flags || [];
  return `
    <div class="sections">${cards}</div>
    ${flags.length ? `
      <div class="panel" style="margin-top:14px">
        <div class="panel__head"><span class="panel__title">Flags</span>
          <span class="muted" style="margin-left:auto;font-size:12px">${flags.length}</span></div>
        <ul class="flaglist">${flags.map((f) => `<li>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--warn)" stroke-width="1.8" style="flex:none;margin-top:2px"><path d="M12 8v5"/><circle cx="12" cy="16.5" r=".7" fill="var(--warn)"/><circle cx="12" cy="12" r="9"/></svg>
          <span>${esc(f)}</span></li>`).join("")}</ul>
      </div>` : ""}`;
}

function renderEvidence(result, detail) {
  const sections = detail.sections || {};
  const sourceById = Object.fromEntries((detail.sources || []).map((s) => [s.source_id, s]));
  const claims = Object.entries(result.brief).filter(([, s]) => s.citations.length);
  if (!claims.length) {
    return `<div class="panel"><div class="empty">
      <div class="empty__title">Nothing was grounded</div>
      <p>No section retained a citation, so there is no evidence to inspect.</p></div></div>`;
  }
  const active = state.activeClaim && claims.some(([n]) => n === state.activeClaim)
    ? state.activeClaim : claims[0][0];
  const [, section] = claims.find(([n]) => n === active);
  const status = sections[active]?.status || "grounded";
  const breakdown = sections[active]?.breakdown;

  return `
    <div class="evidence">
      <div class="panel claimlist">
        ${claims.map(([name, s]) => `
          <button class="claim" data-claim="${name}" aria-current="${name === active}">
            <span class="claim__name">${SECTION_LABELS[name] || name}</span>
            <span class="claim__text">${esc(s.text || "Not asserted")}</span>
          </button>`).join("")}
      </div>

      <div class="stack">
        <div class="panel panel--pad">
          <div class="row">
            <span class="eyebrow">${SECTION_LABELS[active] || active}</span>
            <span class="badge badge--${status}" style="margin-left:auto">${STATUS_COPY[status]}</span>
          </div>
          <p class="quote">${section.text ? esc(section.text) : "Not asserted."}</p>
          <div class="row">
            <div class="meter"><div class="${meterClass(section.confidence, status)}"
                 style="width:${Math.round(section.confidence * 100)}%"></div></div>
            <span class="num">${Math.round(section.confidence * 100)}%</span>
          </div>
          ${breakdown ? `
            <div class="grid-2" style="margin-top:14px;gap:10px">
              ${[["Corroboration", breakdown.corroboration], ["Authority", breakdown.authority],
                 ["Recency", breakdown.recency], ["Ambiguity penalty", breakdown.ambiguity_penalty]]
                .map(([label, value]) => `
                  <div class="row" style="gap:8px">
                    <span class="muted" style="font-size:11.5px;min-width:112px">${label}</span>
                    <div class="meter"><div class="meter__fill" style="width:${Math.round(value * 100)}%"></div></div>
                    <span class="num" style="font-size:11.5px">${value.toFixed(2)}</span>
                  </div>`).join("")}
            </div>
            ${breakdown.reasons?.length ? `<p class="muted" style="font-size:12px;margin-top:12px">
              ${esc(breakdown.reasons.join("; "))}</p>` : ""}` : ""}
        </div>

        <div class="panel">
          <div class="panel__head"><span class="panel__title">Supporting passages</span>
            <span class="muted" style="margin-left:auto;font-size:12px">${section.citations.length}</span></div>
          <div style="padding:14px 16px;display:grid;gap:12px">
            ${section.citations.map((id) => {
              const source = sourceById[id];
              if (!source) return "";
              return `
                <div style="border-left:2px solid var(--hairline-strong);padding-left:12px">
                  <div class="row" style="gap:8px">
                    <span class="favicon">${esc(source.domain[0].toUpperCase())}</span>
                    <a href="${esc(source.url)}" target="_blank" rel="noopener noreferrer"
                       style="font-size:12.5px;color:var(--ink)">${esc(source.domain)}</a>
                    <span class="mono muted">${esc(id)}</span>
                  </div>
                  <p style="font-size:12.5px;color:var(--ink-soft);margin-top:7px;line-height:1.55">
                    ${esc(source.passage.slice(0, 420))}${source.passage.length > 420 ? "..." : ""}</p>
                </div>`;
            }).join("")}
          </div>
        </div>
      </div>
    </div>`;
}

function renderSources(sources) {
  if (!sources.length) {
    return `<div class="panel"><div class="empty">
      <div class="empty__title">No sources retrieved</div></div></div>`;
  }
  return `<div class="sourcecards">${sources.map((source) => `
    <article class="panel sourcecard">
      <div class="sourcecard__domain">
        <span class="favicon">${esc(source.domain[0].toUpperCase())}</span>
        <a href="${esc(source.url)}" target="_blank" rel="noopener noreferrer"
           style="color:var(--ink)">${esc(source.domain)}</a>
      </div>
      <div class="sourcecard__kind">${esc(source.kind.replace(/_/g, " "))}</div>
      <p class="sourcecard__passage">${esc(source.passage)}</p>
      <div class="sourcecard__foot">
        <span class="mono">${esc(source.source_id)}</span>
        <span>relevance ${(source.relevance || 0).toFixed(2)}</span>
        <span style="margin-left:auto">${source.published_at
          ? new Date(source.published_at).toLocaleDateString(undefined, { month: "short", year: "numeric" })
          : "undated"}</span>
      </div>
    </article>`).join("")}</div>`;
}

function renderTrace(result, detail) {
  const steps = result.search_plan_trace || [];
  const timings = detail.timings || {};
  return `
    <div class="grid-2">
      <div class="panel">
        <div class="panel__head"><span class="panel__title">Search plan</span>
          <span class="muted" style="margin-left:auto;font-size:12px">${steps.length} steps</span></div>
        <ol style="margin:0;padding:12px 18px 16px 34px;display:grid;gap:9px">
          ${steps.map((step) => `<li style="font-size:12.5px;color:var(--ink-soft)">${esc(step)}</li>`).join("")}
        </ol>
      </div>
      <div class="panel">
        <div class="panel__head"><span class="panel__title">Stage timings</span>
          <span class="muted" style="margin-left:auto;font-size:12px">${fmtSeconds(result.latency_ms)} total</span></div>
        <div style="padding:14px 18px;display:grid;gap:11px">
          ${[["Planning", timings.planning_ms, 5000], ["Search", timings.search_ms, 25000],
             ["Grounding", timings.grounding_ms, 10000], ["Fallback", timings.fallback_ms, 15000],
             ["Synthesis", timings.synthesis_ms, 5000]].map(([label, value, budget]) => `
            <div class="row" style="gap:10px">
              <span class="muted" style="font-size:11.5px;min-width:78px">${label}</span>
              <div class="meter"><div class="${(value || 0) > budget ? "meter__fill meter__fill--alert" : "meter__fill"}"
                   style="width:${Math.min(100, ((value || 0) / budget) * 100)}%"></div></div>
              <span class="num" style="font-size:11.5px;min-width:52px;text-align:right">${fmtSeconds(value)}</span>
            </div>`).join("")}
        </div>
      </div>
    </div>`;
}

function viewHistory(runs) {
  if (!runs.length) {
    return `<div class="panel"><div class="empty">
      <div class="empty__title">No research yet</div>
      <p>Briefs you generate are kept here.</p>
      <button class="btn btn--primary" data-nav="new" style="margin-top:14px">New research</button>
    </div></div>`;
  }
  return `
    <section>
      <h1 style="font-size:22px;letter-spacing:-0.03em;margin-bottom:14px">Research history</h1>
      <div class="panel">
        <table class="rows">
          <thead><tr><th>Question</th><th>Sources</th><th>Confidence</th><th>Time</th><th>Flags</th></tr></thead>
          <tbody>${runs.map(rowFor).join("")}</tbody>
        </table>
      </div>
    </section>`;
}

function viewAbout(config) {
  return `
    <section style="max-width:660px">
      <h1 style="font-size:22px;letter-spacing:-0.03em">How this works</h1>
      <p class="muted" style="margin:8px 0 18px">
        Five stages, each with its own slice of a 60-second budget.
      </p>
      <div class="panel">
        <ul class="flaglist">
          <li><b style="min-width:96px">Plan</b><span>The agent decides what to search for and in what order. A seed search runs at the same time, so planning time still buys sources.</span></li>
          <li><b style="min-width:96px">Search</b><span>Every step runs concurrently. Duplicate URLs, syndicated copies and low-signal domains are dropped before they count.</span></li>
          <li><b style="min-width:96px">Ground</b><span>Each source is stored with the passage and timestamp it was read from. A claim with no passage cannot enter the brief.</span></li>
          <li><b style="min-width:96px">Verify</b><span>Confidence combines corroboration, source authority, recency and ambiguity. Weak sections trigger one more search round, then get flagged rather than asserted.</span></li>
          <li><b style="min-width:96px">Synthesize</b><span>The brief is assembled and every citation is re-checked against the store.</span></li>
        </ul>
      </div>
      <div class="panel panel--pad" style="margin-top:14px">
        <span class="eyebrow">This deployment</span>
        <div class="grid-2" style="margin-top:10px;font-size:13px">
          <div class="row"><span class="muted" style="min-width:130px">Search backend</span>
            <span>${esc(config?.search_provider || "unknown")}</span></div>
          <div class="row"><span class="muted" style="min-width:130px">Brief writing</span>
            <span>${config?.has_model
              ? `Model-written (${esc(config.model_provider)})`
              : "Extractive (no model key)"}</span></div>
          <div class="row"><span class="muted" style="min-width:130px">Domain floor</span>
            <span>${config?.min_distinct_domains ?? 5} distinct domains</span></div>
          <div class="row"><span class="muted" style="min-width:130px">Confidence threshold</span>
            <span>${config?.confidence_threshold ?? 0.55}</span></div>
        </div>
        ${!config?.has_model ? `<p class="muted" style="font-size:12px;margin-top:12px">
          Without a model key the plan is templated and sections quote passages verbatim
          instead of being written. Grounding, confidence and conflict detection are
          unaffected.</p>` : ""}
      </div>
    </section>`;
}

/* ------------------------------------------------------------------ streaming */

function closeStream() {
  if (state.stream) { state.stream.close(); state.stream = null; }
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
}

function openStream(runId) {
  closeStream();
  state.startedAt = Date.now();
  state.stream = new EventSource(`/api/runs/${runId}/stream`);

  state.stream.onmessage = (event) => {
    const run = JSON.parse(event.data);
    if (run.status === "done") {
      closeStream();
      state.tab = "summary";
      navigate(`#/run/${run.id}`);
      return;
    }
    if (run.status === "failed") {
      closeStream();
      el("view").innerHTML = viewProgress(run);
      return;
    }
    el("view").innerHTML = viewProgress(run);
  };

  state.stream.onerror = () => {
    // The browser retries on its own; a closed stream after completion is expected.
    if (state.stream && state.stream.readyState === EventSource.CLOSED) closeStream();
  };

  state.timer = setInterval(() => {
    const clock = el("clock");
    if (clock && state.startedAt) clock.textContent = fmtClock(Date.now() - state.startedAt);
  }, 100);
}

/* ------------------------------------------------------------------ router */

async function render() {
  const hash = location.hash || "#/";
  const view = el("view");
  const crumb = el("crumb");
  const actions = el("topActions");
  actions.innerHTML = "";

  document.querySelectorAll(".rail__btn").forEach((b) => b.removeAttribute("aria-current"));

  if (hash.startsWith("#/run/")) {
    const runId = hash.slice(6);
    crumb.textContent = "Research result";
    try {
      const run = await api(`/api/runs/${runId}`);
      if (run.status === "running" || run.status === "queued") {
        crumb.textContent = "Research in progress";
        view.innerHTML = viewProgress(run);
        openStream(runId);
        return;
      }
      closeStream();
      view.innerHTML = viewBrief(run);
      actions.innerHTML = `
        <button class="btn btn--sm" id="copyLink">Copy link</button>
        <button class="btn btn--sm" id="downloadJson">Download JSON</button>
        <button class="btn btn--sm btn--primary" data-nav="new">New research</button>`;
      el("copyLink").onclick = async () => {
        await navigator.clipboard.writeText(location.href);
        toast("Link copied");
      };
      el("downloadJson").onclick = () => {
        const blob = new Blob([JSON.stringify(run.result, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `brief-${run.id}.json`;
        link.click();
        URL.revokeObjectURL(url);
      };
    } catch (error) {
      view.innerHTML = `<div class="panel"><div class="empty">
        <div class="empty__title">That brief could not be loaded</div>
        <p>${esc(error.message)}</p></div></div>`;
    }
    return;
  }

  closeStream();

  if (hash === "#/new") {
    document.querySelector('[data-nav="new"]')?.setAttribute("aria-current", "page");
    crumb.textContent = "New research";
    view.innerHTML = viewNew();
    return;
  }

  if (hash === "#/history") {
    document.querySelector('[data-nav="history"]')?.setAttribute("aria-current", "page");
    crumb.textContent = "Research history";
    const { runs } = await api("/api/runs");
    view.innerHTML = viewHistory(runs);
    return;
  }

  if (hash === "#/about") {
    document.querySelector('[data-nav="about"]')?.setAttribute("aria-current", "page");
    crumb.textContent = "How this works";
    view.innerHTML = viewAbout(state.config);
    return;
  }

  document.querySelector('[data-nav="home"]')?.setAttribute("aria-current", "page");
  crumb.textContent = "Home";
  const { runs } = await api("/api/runs");
  view.innerHTML = viewHome(runs);
  const host = el("contourHost");
  if (host) host.appendChild(el("tpl-contour").content.cloneNode(true));
}

async function submitResearch(query, depth) {
  const focus = [...state.focus];
  const framed = focus.length ? `${query} (focus on ${focus.join(", ").toLowerCase()})` : query;
  try {
    const { id } = await api("/api/research", {
      method: "POST",
      body: JSON.stringify({ query: framed, depth }),
    });
    navigate(`#/run/${id}`);
  } catch (error) {
    toast(error.message);
  }
}

/* ------------------------------------------------------------------ events */

document.addEventListener("click", (event) => {
  const nav = event.target.closest("[data-nav]");
  if (nav) {
    const target = nav.dataset.nav;
    navigate(target === "home" ? "#/" : `#/${target}`);
    return;
  }

  const depth = event.target.closest("[data-depth]");
  if (depth) {
    state.depth = depth.dataset.depth;
    document.querySelectorAll("[data-depth]").forEach((node) =>
      node.setAttribute("aria-pressed", node.dataset.depth === state.depth));
    return;
  }

  const focus = event.target.closest("[data-focus]");
  if (focus) {
    const area = focus.dataset.focus;
    if (state.focus.has(area)) state.focus.delete(area);
    else state.focus.add(area);
    focus.setAttribute("aria-pressed", state.focus.has(area));
    return;
  }

  const tab = event.target.closest("[data-tab]");
  if (tab) {
    state.tab = tab.dataset.tab;
    document.querySelectorAll("[data-tab]").forEach((node) =>
      node.setAttribute("aria-selected", node.dataset.tab === state.tab));
    api(`/api/runs/${location.hash.slice(6)}`).then((run) => {
      el("tabPanel").innerHTML = renderTab(run);
    });
    return;
  }

  const claim = event.target.closest("[data-claim]");
  if (claim) {
    state.activeClaim = claim.dataset.claim;
    api(`/api/runs/${location.hash.slice(6)}`).then((run) => {
      el("tabPanel").innerHTML = renderTab(run);
    });
    return;
  }

  const cite = event.target.closest("[data-src]");
  if (cite) {
    state.tab = "evidence";
    api(`/api/runs/${location.hash.slice(6)}`).then((run) => {
      document.querySelectorAll("[data-tab]").forEach((node) =>
        node.setAttribute("aria-selected", node.dataset.tab === "evidence"));
      el("tabPanel").innerHTML = renderTab(run);
    });
    return;
  }

  const expand = event.target.closest("[data-expand]");
  if (expand) {
    const text = expand.previousElementSibling;
    const clamped = text.classList.toggle("section__text--clamped");
    expand.textContent = clamped ? "Show full text" : "Show less";
    return;
  }

  const row = event.target.closest("[data-run]");
  if (row) navigate(`#/run/${row.dataset.run}`);
});

document.addEventListener("submit", (event) => {
  event.preventDefault();
  if (event.target.id === "askForm") {
    submitResearch(el("askInput").value.trim(), state.depth);
  } else if (event.target.id === "composeForm") {
    submitResearch(el("composeQuery").value.trim(), state.depth);
  }
});

window.addEventListener("hashchange", render);
window.addEventListener("beforeunload", closeStream);

(async function boot() {
  try {
    state.config = await api("/api/config");
  } catch {
    state.config = null;
  }
  render();
})();
