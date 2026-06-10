const form = document.getElementById("predict-form");
const sequenceInput = document.getElementById("sequence");
const microbeSelect = document.getElementById("microbe");
const resultEl = document.getElementById("result");
const submitBtn = document.getElementById("submit-btn");

// Same physicochemical groupings src/models/mic_baseline.py uses as model
// features (RESIDUE_GROUPS) plus a "structural" bucket for glycine/proline,
// which that grouping leaves uncategorized. Priority resolves the one
// overlap (aromatic residues are also hydrophobic) so each letter gets one color.
const RESIDUE_GROUPS = [
  { name: "aromatic", label: "Aromatic", residues: "FWY", swatch: "var(--residue-aromatic)" },
  { name: "positive", label: "Basic (+)", residues: "KRH", swatch: "var(--residue-positive)" },
  { name: "negative", label: "Acidic (–)", residues: "DE", swatch: "var(--residue-negative)" },
  { name: "polar", label: "Polar", residues: "STNQC", swatch: "var(--residue-polar)" },
  { name: "hydrophobic", label: "Hydrophobic", residues: "AILMV", swatch: "var(--residue-hydrophobic)" },
  { name: "structural", label: "Structural", residues: "GP", swatch: "var(--residue-structural)" },
];

function residueGroupFor(letter) {
  return RESIDUE_GROUPS.find((group) => group.residues.includes(letter)) || null;
}

function renderChain(sequence) {
  const chips = sequence
    .split("")
    .map((letter) => {
      const group = residueGroupFor(letter);
      const color = group ? group.swatch : "var(--ink-soft)";
      return `<span class="residue" style="background:${color}" title="${letter}${group ? " — " + group.label : ""}">${letter}</span>`;
    })
    .join("");

  const legend = RESIDUE_GROUPS.map(
    (group) =>
      `<span class="legend-item"><span class="legend-swatch" style="background:${group.swatch}"></span>${group.label}</span>`
  ).join("");

  return `
    <div class="chain-label">Residue chain</div>
    <div class="chain">${chips}</div>
    <div class="legend">${legend}</div>
  `;
}

async function loadMicrobes() {
  const response = await fetch("/microbes");
  const microbes = await response.json();
  microbeSelect.innerHTML = "";
  for (const microbe of microbes) {
    const option = document.createElement("option");
    option.value = microbe.key;
    option.textContent = microbe.display_name;
    microbeSelect.appendChild(option);
  }
}

function gramLabel(gramStatus) {
  if (gramStatus === "gram_positive") return { text: "Gram-positive", cls: "positive" };
  if (gramStatus === "gram_negative") return { text: "Gram-negative", cls: "negative" };
  return { text: gramStatus, cls: "" };
}

function renderResult(data, sequence) {
  resultEl.classList.remove("hidden");
  const gram = gramLabel(data.gram_status);
  const warningsHtml = data.warnings.length
    ? `<ul class="warnings">${data.warnings.map((w) => `<li>${w}</li>`).join("")}</ul>`
    : "";

  resultEl.innerHTML = `
    <div class="readout-body">
      <div class="organism-line">
        ${data.microbe_display_name}
        <span class="gram-pill ${gram.cls}">${gram.text}</span>
      </div>
      <div class="mic-figure">
        <div class="mic-value">${data.mic_ug_per_ml.toFixed(2)}<span class="unit">µg/mL</span></div>
        <div class="mic-caption">log10(MIC) = ${data.log10_mic.toFixed(3)}</div>
      </div>
      ${renderChain(sequence)}
      ${warningsHtml}
    </div>
  `;
}

function renderError(message) {
  resultEl.classList.remove("hidden");
  resultEl.innerHTML = `<p class="error-box">${message}</p>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Running…";
  const sequence = sequenceInput.value.trim().toUpperCase();
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence, microbe_key: microbeSelect.value }),
    });
    const data = await response.json();
    if (!response.ok) {
      renderError(data.detail || "Prediction failed.");
    } else {
      renderResult(data, sequence);
    }
  } catch (err) {
    renderError("Could not reach the prediction server.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Run prediction";
  }
});

loadMicrobes().catch(() => renderError("Could not load the microbe list."));
