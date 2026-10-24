/* ==========================================================================
   Neuro/AD — front-end logic

   Talks to the same FastAPI endpoints the Streamlit dashboard uses, so the two
   surfaces cannot report different numbers. No framework: the whole interface
   is one upload, one request, and one render.
   ========================================================================== */

const STAGE = {
  NC:   { color: "#34D399", glow: "rgba(52, 211, 153, 0.22)",  wash: "rgba(52, 211, 153, 0.09)",  line: "rgba(52, 211, 153, 0.28)",  desc: "Normal cognition" },
  EMCI: { color: "#FBBF24", glow: "rgba(251, 191, 36, 0.22)",  wash: "rgba(251, 191, 36, 0.09)",  line: "rgba(251, 191, 36, 0.28)",  desc: "Early mild cognitive impairment" },
  LMCI: { color: "#FB923C", glow: "rgba(251, 146, 60, 0.22)",  wash: "rgba(251, 146, 60, 0.09)",  line: "rgba(251, 146, 60, 0.28)",  desc: "Late mild cognitive impairment" },
  AD:   { color: "#F43F5E", glow: "rgba(244, 63, 94, 0.22)",   wash: "rgba(244, 63, 94, 0.09)",   line: "rgba(244, 63, 94, 0.3)",    desc: "Alzheimer's disease" },
};
const ORDER = ["NC", "EMCI", "LMCI", "AD"];
const RING_CIRCUMFERENCE = 2 * Math.PI * 56;
const CENTRAL_MASS_THRESHOLD = 0.40;

const $ = (id) => document.getElementById(id);
const el = {
  status: $("status"), statusText: $("status-text"),
  dropzone: $("dropzone"), fileInput: $("file-input"),
  samplesWrap: $("samples-wrap"), samples: $("samples"), samplesHint: $("samples-hint"),
  hero: $("hero"), loading: $("loading"), loadingText: $("loading-text"),
  results: $("results"), alert: $("alert"),
  ringFg: $("ring-fg"), ringValue: $("ring-value"),
  stage: $("verdict-stage"), desc: $("verdict-desc"), file: $("verdict-file"),
  verdict: $("verdict"), impairment: $("stat-impairment"), margin: $("stat-margin"),
  stepper: $("stepper"), probs: $("probs"),
  imgSlice: $("img-slice"), imgOverlay: $("img-overlay"),
  compare: $("compare"), cmpTop: $("cmp-top"), cmpHandle: $("cmp-handle"), compareNote: $("compare-note"),
  viewSeg: $("view-seg"), targetSelect: $("target-select"),
  auditChip: $("audit-chip"), auditGrid: $("audit-grid"), auditNote: $("audit-note"),
  resetBtn: $("reset-btn"),
};

/** The file currently under analysis, so a target-class change can re-run it. */
let currentFile = null;
let currentResult = null;

/* ------------------------------------------------------------------ utils */

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const pct0 = (x) => `${Math.round(x * 100)}%`;

function showAlert(message) {
  el.alert.textContent = message;
  el.alert.hidden = false;
}

function clearAlert() {
  el.alert.hidden = true;
}

/** Count a number up to its value — makes the headline figures feel measured. */
function animateValue(node, target, format, duration = 900) {
  const start = performance.now();
  function frame(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    node.textContent = format(target * eased);
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/* ----------------------------------------------------------------- health */

async function checkHealth() {
  try {
    const health = await (await fetch("/health")).json();
    if (health.model_loaded) {
      el.status.className = "status ok";
      el.statusText.textContent = `Model ready · ${health.device}`;
    } else {
      el.status.className = "status bad";
      el.statusText.textContent = "No checkpoint loaded";
    }
  } catch {
    el.status.className = "status bad";
    el.statusText.textContent = "Service unreachable";
  }
}

/* ---------------------------------------------------------------- samples */

async function loadSamples() {
  try {
    const data = await (await fetch("/samples")).json();
    if (!data.samples || !data.samples.length) return;

    el.samplesHint.textContent = data.location || "";
    el.samples.innerHTML = data.samples.map((sample) => {
      const stage = STAGE[sample.true_stage] || STAGE.NC;
      const errorClass = sample.is_known_error ? " is-error" : "";
      const label = sample.is_known_error ? `${sample.true_stage}?` : sample.true_stage;
      return `
        <div class="sample${errorClass}" data-name="${sample.name}"
             title="${sample.name}">
          <img src="/samples/${encodeURIComponent(sample.name)}" alt="${sample.true_stage} sample" loading="lazy"/>
          <div class="sample-tag" style="color:${stage.color}">${label}</div>
        </div>`;
    }).join("");

    el.samples.querySelectorAll(".sample").forEach((node) => {
      node.addEventListener("click", () => loadSample(node.dataset.name));
    });
    el.samplesWrap.hidden = false;
  } catch {
    /* Samples are a convenience; their absence must not break the page. */
  }
}

async function loadSample(name) {
  try {
    const response = await fetch(`/samples/${encodeURIComponent(name)}`);
    const blob = await response.blob();
    analyse(new File([blob], name, { type: blob.type || "image/png" }));
  } catch {
    showAlert(`Could not load sample "${name}".`);
  }
}

/* ---------------------------------------------------------------- analyse */

async function analyse(file, targetClass = null) {
  currentFile = file;
  clearAlert();
  el.results.hidden = true;
  el.hero.hidden = true;
  el.loading.hidden = false;
  el.loadingText.textContent = targetClass
    ? `Re-explaining as ${targetClass}`
    : "Running inference and computing attribution";

  const form = new FormData();
  form.append("file", file, file.name);
  const query = targetClass ? `?target_class=${encodeURIComponent(targetClass)}` : "";

  try {
    const response = await fetch(`/explain${query}`, { method: "POST", body: form });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }
    const result = await response.json();
    currentResult = result;
    window.dispatchEvent(new CustomEvent("mri-result", { detail: result }));
    render(result, file);
  } catch (error) {
    el.loading.hidden = true;
    showAlert(error.message || "Inference failed. Is the API running?");
  }
}

/* ----------------------------------------------------------------- render */

function render(result, file) {
  el.loading.hidden = true;
  el.results.hidden = false;
  // Collapse the hero once there is something to read: the dropzone has done
  // its job, and a screen of empty upload affordance above the findings just
  // pushes the verdict below the fold. The reset button brings it back.
  el.hero.hidden = true;
  window.scrollTo({ top: 0, behavior: "auto" });

  const predicted = result.predicted_class;
  const stage = STAGE[predicted] || STAGE.NC;

  // Stage identity drives the whole panel's colour through CSS variables.
  el.verdict.style.setProperty("--stage", stage.color);
  el.verdict.style.setProperty("--stage-glow", stage.glow);
  el.verdict.style.setProperty("--stage-wash", stage.wash);
  el.verdict.style.setProperty("--stage-line", stage.line);

  el.stage.textContent = predicted;
  el.desc.textContent = result.predicted_description || stage.desc;
  el.file.textContent = file.name;

  // Ring: reset to empty first so the draw animation always plays.
  el.ringFg.style.strokeDasharray = `${RING_CIRCUMFERENCE} ${RING_CIRCUMFERENCE}`;
  el.ringFg.style.strokeDashoffset = RING_CIRCUMFERENCE;
  requestAnimationFrame(() => {
    el.ringFg.style.strokeDashoffset = RING_CIRCUMFERENCE * (1 - result.confidence);
  });
  animateValue(el.ringValue, result.confidence, (v) => pct0(v));
  animateValue(el.impairment, result.impairment_probability, (v) => pct(v));

  // Margin over the runner-up: a 40/38 split and a 99/1 split are very
  // different clinical situations that a single confidence figure hides.
  const ranked = result.probabilities;
  const margin = ranked.length > 1 ? ranked[0].probability - ranked[1].probability : 1;
  animateValue(el.margin, margin, (v) => `${(v * 100).toFixed(1)}pp`);

  renderStepper(predicted);
  renderProbabilities(ranked, predicted);
  renderImages(result, file);
  renderAudit(result.focus);
  renderTargetSelect(result);
}

function renderStepper(predicted) {
  const index = ORDER.indexOf(predicted);
  el.stepper.innerHTML = ORDER.map((name, i) => {
    const stage = STAGE[name];
    const state = i < index ? "done" : i === index ? "now" : "";
    return `
      <div class="step ${state}" style="--step-c:${stage.color}">
        <div class="step-name">${name}</div>
        <div class="step-desc">${stage.desc}</div>
      </div>`;
  }).join("");
}

function renderProbabilities(ranked, predicted) {
  el.probs.innerHTML = ranked.map((entry) => {
    const stage = STAGE[entry.class] || STAGE.NC;
    const isTop = entry.class === predicted ? " is-top" : "";
    return `
      <div class="prob${isTop}" style="--pc:${stage.color}">
        <div class="prob-top">
          <div>
            <span class="prob-name" style="color:${stage.color}">${entry.class}</span>
            <span class="prob-desc">${entry.description}</span>
          </div>
          <span class="prob-pct">${pct(entry.probability)}</span>
        </div>
        <div class="prob-bar">
          <div class="prob-fill" style="background:${stage.color}"
               data-w="${Math.max(entry.probability * 100, 0.8)}"></div>
        </div>
      </div>`;
  }).join("");

  // Widths applied on the next frame so the transition runs from zero.
  requestAnimationFrame(() => {
    el.probs.querySelectorAll(".prob-fill").forEach((fill) => {
      fill.style.width = `${fill.dataset.w}%`;
    });
  });
}

function renderImages(result, file) {
  el.imgSlice.src = URL.createObjectURL(file);
  if (result.overlay_png_base64) {
    el.imgOverlay.dataset.overlay = `data:image/png;base64,${result.overlay_png_base64}`;
  }
  if (result.heatmap_png_base64) {
    el.imgOverlay.dataset.heatmap = `data:image/png;base64,${result.heatmap_png_base64}`;
  }
  setView(document.querySelector(".seg-btn.is-active")?.dataset.view || "compare");
  setSplit(0.5);
}

function renderAudit(focus) {
  if (!focus) return;
  const healthy = focus.central_share >= CENTRAL_MASS_THRESHOLD;

  el.auditChip.className = `chip ${healthy ? "ok" : "warn"}`;
  el.auditChip.textContent = healthy ? "Attending to tissue" : "Attention off-tissue";

  const quadrant = String(focus.dominant_quadrant || "—").replace(/_/g, " ");
  el.auditGrid.innerHTML = `
    <div class="audit-item"><div class="av">${pct0(focus.central_share)}</div><div class="ak">Central mass</div></div>
    <div class="audit-item"><div class="av">${quadrant}</div><div class="ak">Dominant quadrant</div></div>
    <div class="audit-item"><div class="av">${pct0(focus.activated_fraction)}</div><div class="ak">Activated area</div></div>`;

  el.auditNote.textContent = healthy
    ? "Most attribution falls on central brain structure, which is where stage-relevant atrophy appears. The prediction is at least looking in the right place."
    : `Under ${pct0(CENTRAL_MASS_THRESHOLD)} of attribution falls in the central region. On a registered slice that usually means the model is reading background or an acquisition artefact rather than anatomy — treat this prediction as unverified.`;
}

function renderTargetSelect(result) {
  const explained = result.explained_class || result.predicted_class;
  el.targetSelect.innerHTML =
    `<option value="">predicted stage</option>` +
    ORDER.map((name) => `<option value="${name}">${name}</option>`).join("");
  el.targetSelect.value = explained === result.predicted_class ? "" : explained;
}

/* ------------------------------------------------------- attribution view */

function setView(view) {
  const overlay = el.imgOverlay.dataset.overlay;
  const heatmap = el.imgOverlay.dataset.heatmap;

  if (view === "heatmap" && heatmap) {
    el.imgOverlay.src = heatmap;
  } else if (overlay) {
    el.imgOverlay.src = overlay;
  }

  const split = view === "compare";
  el.compare.classList.toggle("no-split", !split);
  el.compareNote.textContent = split
    ? "Drag the divider to wipe between the scan and the attribution map. Warm regions raised the score for the explained stage."
    : view === "heatmap"
      ? "The raw attribution map. Brighter regions contributed more to the explained stage's score."
      : "The attribution map blended over the scan. Warm regions raised the score for the explained stage.";

  if (split) setSplit(currentSplit);
}

let currentSplit = 0.5;

function setSplit(fraction) {
  currentSplit = Math.max(0, Math.min(1, fraction));
  const percent = currentSplit * 100;
  el.cmpTop.style.width = `${percent}%`;
  // The clipped image must stay the size of the full frame, so it is scaled
  // inversely to the clip width — otherwise the two halves do not line up.
  el.cmpTop.style.setProperty("--cmp-w", `${100 / Math.max(currentSplit, 0.0001)}%`);
  el.cmpHandle.style.left = `${percent}%`;
}

function bindCompare() {
  let dragging = false;

  const update = (clientX) => {
    const rect = el.compare.getBoundingClientRect();
    setSplit((clientX - rect.left) / rect.width);
  };

  el.compare.addEventListener("pointerdown", (event) => {
    if (el.compare.classList.contains("no-split")) return;
    dragging = true;
    el.compare.setPointerCapture(event.pointerId);
    update(event.clientX);
  });
  el.compare.addEventListener("pointermove", (event) => {
    if (dragging) update(event.clientX);
  });
  el.compare.addEventListener("pointerup", () => { dragging = false; });
  el.compare.addEventListener("pointercancel", () => { dragging = false; });
}

/* ----------------------------------------------------------------- events */

function bindUpload() {
  el.dropzone.addEventListener("click", () => el.fileInput.click());
  el.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      el.fileInput.click();
    }
  });
  el.fileInput.addEventListener("change", (event) => {
    if (event.target.files.length) analyse(event.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((type) => {
    el.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      el.dropzone.classList.add("is-drag");
    });
  });
  ["dragleave", "drop"].forEach((type) => {
    el.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      if (type === "dragleave" && el.dropzone.contains(event.relatedTarget)) return;
      el.dropzone.classList.remove("is-drag");
    });
  });
  el.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) analyse(file);
  });
}

function bindControls() {
  el.viewSeg.addEventListener("click", (event) => {
    const button = event.target.closest(".seg-btn");
    if (!button) return;
    el.viewSeg.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("is-active"));
    button.classList.add("is-active");
    setView(button.dataset.view);
  });

  el.targetSelect.addEventListener("change", () => {
    if (currentFile) analyse(currentFile, el.targetSelect.value || null);
  });

  el.resetBtn.addEventListener("click", () => {
    el.results.hidden = true;
    el.hero.hidden = false;
    el.fileInput.value = "";
    currentFile = null;
    currentResult = null;
    clearAlert();
    el.hero.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

/* ------------------------------------------------------------------- init */

checkHealth();
loadSamples();
bindUpload();
bindCompare();
bindControls();
