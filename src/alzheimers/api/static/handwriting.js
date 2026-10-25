/* ==========================================================================
   Neuro/AD — handwriting view

   Captures a spiral drawing as timestamped pen points, sends it to
   /handwriting/analyze, and renders the indicator, a speed-coloured trace, and
   per-feature contributions. Also owns the MRI / Handwriting tab switch.
   ========================================================================== */

(() => {
  const W = 720, H = 520;
  const TIER = { Low: "#34D399", Moderate: "#FBBF24", High: "#F43F5E" };
  const RING_C = 2 * Math.PI * 56;
  const $ = (id) => document.getElementById(id);

  const el = {
    tabs: document.querySelectorAll(".tab"),
    viewMri: $("view-mri"), viewHw: $("view-hw"),
    canvas: $("hw-canvas"), hint: $("hw-hint"),
    analyse: $("hw-analyse"), clear: $("hw-clear"),
    demos: document.querySelectorAll(".hw-demo"),
    stage: $("hw-stage"), alert: $("hw-alert"), loading: $("hw-loading"), results: $("hw-results"),
    ringFg: $("hw-ring-fg"), ringValue: $("hw-ring-value"), tier: $("hw-tier"), source: $("hw-source"),
    statTime: $("hw-stat-time"), statStrokes: $("hw-stat-strokes"),
    trace: $("hw-trace"), feats: $("hw-feats"),
    fusion: $("hw-fusion"), fusionChip: $("hw-fusion-chip"), fusionGrid: $("hw-fusion-grid"),
    disclaimer: $("hw-disclaimer"), reset: $("hw-reset"),
  };

  let strokes = [];          // [{x, y, t}] per stroke, canvas coordinates
  let current = null;
  let busy = false;          // true while a replay is animating
  let source = "drawn by you";
  let mriResult = null;

  window.addEventListener("mri-result", (e) => { mriResult = e.detail; });

  /* ------------------------------------------------------------------ tabs */

  function showView(view) {
    el.tabs.forEach((t) => t.classList.toggle("is-active", t.dataset.view === view));
    el.viewMri.hidden = view !== "mri";
    el.viewHw.hidden = view !== "hw";
    if (view === "hw") { sizeCanvas(el.canvas); redraw(); }
    window.scrollTo({ top: 0 });
  }
  el.tabs.forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));

  /* ---------------------------------------------------------------- canvas */

  const ctx = el.canvas.getContext("2d");

  /** Back the canvas with device pixels but keep drawing in logical 720x520. */
  function sizeCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function drawGuide(c) {
    // Faint target spiral and a start dot: tells the person what to draw without
    // tracing a line they would follow slavishly.
    c.save();
    c.strokeStyle = "rgba(255,255,255,0.1)";
    c.setLineDash([5, 7]);
    c.lineWidth = 1.5;
    c.beginPath();
    const turns = 3, R = 0.4 * H;
    for (let i = 0; i <= 600; i++) {
      const th = (i / 600) * turns * 2 * Math.PI;
      const r = R * (i / 600);
      const x = W / 2 + r * Math.cos(th), y = H / 2 + r * Math.sin(th);
      i ? c.lineTo(x, y) : c.moveTo(x, y);
    }
    c.stroke();
    c.setLineDash([]);
    c.fillStyle = "rgb(231, 228, 231)";
    c.beginPath(); c.arc(W / 2, H / 2, 5, 0, 2 * Math.PI); c.fill();
    c.restore();
  }

  function redraw() {
    ctx.clearRect(0, 0, W, H);
    drawGuide(ctx);
    ctx.strokeStyle = "#F4F2F4";
    ctx.lineWidth = 2.5;
    ctx.lineCap = ctx.lineJoin = "round";
    for (const s of strokes) {
      ctx.beginPath();
      s.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.stroke();
    }
    const has = strokes.length > 0;
    el.hint.classList.toggle("is-hidden", has);
    el.analyse.disabled = !has || busy;
  }

  function toCanvas(ev) {
    const rect = el.canvas.getBoundingClientRect();
    return { x: (ev.clientX - rect.left) * (W / rect.width), y: (ev.clientY - rect.top) * (H / rect.height), t: ev.timeStamp };
  }

  el.canvas.addEventListener("pointerdown", (ev) => {
    if (busy) return;
    el.canvas.setPointerCapture(ev.pointerId);
    source = "drawn by you";
    current = [toCanvas(ev)];
    strokes.push(current);
    redraw();
  });

  el.canvas.addEventListener("pointermove", (ev) => {
    if (!current) return;
    // Coalesced events recover the samples the browser merged into one frame;
    // without them a 60 Hz display would cap the tremor we can see at 30 Hz.
    const events = ev.getCoalescedEvents ? ev.getCoalescedEvents() : [];
    for (const e of events.length ? events : [ev]) {
      const p = toCanvas(e);
      const last = current[current.length - 1];
      if (p.t <= last.t) continue;
      current.push(p);
      ctx.strokeStyle = "#F4F2F4"; ctx.lineWidth = 2.5; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    }
    el.hint.classList.add("is-hidden");
    el.analyse.disabled = false;
  });

  const endStroke = () => { current = null; };
  el.canvas.addEventListener("pointerup", endStroke);
  el.canvas.addEventListener("pointercancel", endStroke);

  el.clear.addEventListener("click", () => { if (!busy) { strokes = []; current = null; hideAlert(); redraw(); } });

  /* ---------------------------------------------------------------- replay */

  /** Animate a recorded drawing in (sped-up) real time, then analyse it. */
  async function replay(profile) {
    if (busy) return;
    hideAlert();
    let demo;
    try {
      const seed = Math.floor(Math.random() * 5);
      const resp = await fetch(`/handwriting/demo/${profile}?seed=${seed}`);
      if (!resp.ok) throw new Error();
      demo = await resp.json();
    } catch {
      return showAlert("Could not load the simulated drawing.");
    }
    busy = true;
    el.results.hidden = true;
    el.stage.hidden = false;
    el.demos.forEach((b) => (b.disabled = true));
    strokes = [];
    redraw();

    const SPEEDUP = 3;
    const t0 = demo.strokes[0][0].t;
    const all = demo.strokes.flatMap((s, si) => s.map((p) => ({ ...p, si })));
    await new Promise((resolve) => {
      const start = performance.now();
      let idx = 0;
      (function frame(now) {
        const clock = t0 + (now - start) * SPEEDUP;
        while (idx < all.length && all[idx].t <= clock) {
          const p = all[idx++];
          if (!strokes[p.si]) strokes[p.si] = [];
          strokes[p.si].push({ x: p.x, y: p.y, t: p.t });
        }
        redraw();
        idx < all.length ? requestAnimationFrame(frame) : resolve();
      })(performance.now());
    });

    strokes = demo.strokes.map((s) => s.map((p) => ({ x: p.x, y: p.y, t: p.t })));
    source = `simulated · ${profile}`;
    busy = false;
    el.demos.forEach((b) => (b.disabled = false));
    redraw();
    await analyse();
  }
  el.demos.forEach((b) => b.addEventListener("click", () => replay(b.dataset.profile)));

  /* --------------------------------------------------------------- analyse */

  function showAlert(msg) { el.alert.textContent = msg; el.alert.hidden = false; }
  function hideAlert() { el.alert.hidden = true; }

  async function analyse() {
    if (!strokes.length || busy) return;
    hideAlert();
    el.loading.hidden = false;
    el.results.hidden = true;
    try {
      const resp = await fetch("/handwriting/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strokes, canvas_width: W, canvas_height: H }),
      });
      const body = await resp.json();
      if (!resp.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Could not analyse this drawing.");
      render(body);
    } catch (err) {
      showAlert(err.message || "Could not reach the service.");
    } finally {
      el.loading.hidden = true;
    }
  }
  el.analyse.addEventListener("click", analyse);

  /* ---------------------------------------------------------------- render */

  function lerp(a, b, t) { return a + (b - a) * t; }
  /** slow dim mauve -> fast bright */
  function ramp(s) {
    // mauve chart-3 -> muted-foreground -> near-white: dim is slow, bright is fast
    const stops = [[89, 76, 91], [168, 158, 169], [250, 250, 250]];
    const k = Math.min(Math.max(s, 0), 1) * 2;
    const i = Math.min(Math.floor(k), 1), f = k - i;
    return `rgb(${stops[i].map((c, j) => Math.round(lerp(c, stops[i + 1][j], f))).join(",")})`;
  }

  function drawTrace(trace) {
    sizeCanvas(el.trace);
    const c = el.trace.getContext("2d");
    c.clearRect(0, 0, W, H);
    c.lineWidth = 3; c.lineCap = c.lineJoin = "round";
    for (const stroke of trace) {
      for (let i = 1; i < stroke.length; i++) {
        c.strokeStyle = ramp(stroke[i].s);
        c.beginPath(); c.moveTo(stroke[i - 1].x, stroke[i - 1].y); c.lineTo(stroke[i].x, stroke[i].y); c.stroke();
      }
    }
  }

  function fmt(v, unit) {
    const n = Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
    return unit ? `${n} ${unit}` : n;
  }

  function renderFeatures(rows) {
    const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 0.01);
    el.feats.innerHTML = rows.map((r) => {
      const dir = r.contribution >= 0 ? "up" : "down";
      const width = (Math.abs(r.contribution) / max) * 50;
      return `<div class="hw-feat" title="${r.meaning}">
        <div class="hw-feat-top"><span class="hw-feat-name">${r.label}</span><span class="hw-feat-val">${fmt(r.value, r.unit)}</span></div>
        <div class="hw-bar"><div class="hw-bar-fill ${dir}" data-w="${width}"></div></div>
        <div class="hw-feat-meaning">${r.meaning}</div>
      </div>`;
    }).join("");
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.feats.querySelectorAll(".hw-bar-fill").forEach((n) => { n.style.width = `${n.dataset.w}%`; });
    }));
  }

  function renderFusion(handwriting) {
    if (!mriResult) { el.fusion.hidden = true; return; }
    const mri = mriResult.impairment_probability;
    const combined = (mri + handwriting) / 2;
    const cell = (value, key, color) =>
      `<div class="hw-fuse-cell"><div class="stat-value" style="color:${color}">${(value * 100).toFixed(0)}%</div><div class="stat-key">${key}</div></div>`;
    const tier = combined < 0.35 ? "Low" : combined < 0.65 ? "Moderate" : "High";
    el.fusionGrid.innerHTML = cell(mri, `MRI · ${mriResult.predicted_class}`, "var(--ink)")
      + cell(handwriting, "Handwriting indicator", "var(--ink)")
      + cell(combined, "Combined (illustrative)", TIER[tier]);
    el.fusionChip.textContent = `${tier} combined`;
    el.fusion.hidden = false;
  }

  function render(r) {
    const color = TIER[r.tier];
    el.results.hidden = false;
    el.tier.textContent = `${r.tier} indicator`;
    el.tier.style.color = color;
    el.source.textContent = source;
    el.ringFg.style.stroke = color;
    el.ringFg.style.strokeDasharray = RING_C;
    el.ringFg.style.strokeDashoffset = RING_C;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.ringFg.style.strokeDashoffset = RING_C * (1 - r.probability);
    }));
    el.ringValue.textContent = `${Math.round(r.probability * 100)}%`;

    const duration = r.features.find((f) => f.name === "duration_s");
    el.statTime.textContent = duration ? `${duration.value.toFixed(1)} s` : "—";
    el.statStrokes.textContent = String(r.trace.length);

    drawTrace(r.trace);
    renderFeatures(r.features);
    renderFusion(r.probability);
    el.disclaimer.textContent = r.disclaimer;
    el.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  el.reset.addEventListener("click", () => {
    strokes = []; current = null;
    el.results.hidden = true;
    redraw();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  sizeCanvas(el.canvas);
  redraw();
})();
