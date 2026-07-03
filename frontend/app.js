/**
 * CattleVision AI — Premium Frontend JS
 */

"use strict";

// ── Breed data (mirrors app.py BREED_INFO) ─────────────────────────────────
const BREED_INFO = {
  "Ayrshire":          "Dairy breed from Scotland. Red & white markings, hardy, efficient milk production.",
  "Brown_Swiss":       "One of the oldest dairy breeds (Switzerland). Large, robust, high-protein milk for cheese.",
  "Gir":               "Famous Indian dairy breed from Gujarat. Rounded forehead, pendulous ears, heat tolerant.",
  "Hallikar":          "Draft breed from Karnataka. Vertical horns, endurance. Progenitor of South Indian breeds.",
  "Holstein_Friesian": "World's highest-production dairy breed. Distinctive black & white markings, European origin.",
  "Jersey":            "Small dairy breed from Channel Islands. Very high butterfat milk, gentle temperament.",
  "Kankrej":           "Dual-purpose breed from Gujarat/Rajasthan. Lyre-shaped horns. Known as Guzerat in Brazil.",
  "Murrah":            "Premier water buffalo from India (Haryana/Punjab). Jet black, tightly curled horns, high yield.",
  "Nagpuri":           "Riverine buffalo from Maharashtra. Black with white patches. Long sword-like horns.",
  "Ongole":            "Large, muscular draft/dual-purpose breed from Andhra Pradesh. Source of American Brahman.",
  "Rathi":             "Dual-purpose breed from the Thar Desert. Brown/red with white patches, extreme heat adapted.",
  "Red_Dane":          "Dairy breed from Denmark. Solid red color, good milk production and fertility.",
  "Red_Sindhi":        "Zebu dairy breed from Pakistan/India. Deep red, compact, heat tolerant, disease resistant.",
  "Sahiwal":           "Top Zebu dairy breed from Punjab. Reddish-brown, tick-resistant, high milk fat content.",
  "Tharparkar":        "Dual-purpose breed from the Thar Desert. White/grey coat, drought and disease resistant.",
};

// ── Disease triage colors ──────────────────────────────────────────────────
const DISEASE_COLORS = {
  "healthy":           "green",
  "foot-and-mouth":    "red",
  "lumpy":             "amber",
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const fileInput       = document.getElementById("file-input");
const uploadArea      = document.getElementById("upload-area");
const previewImage    = document.getElementById("preview-image");
const previewWrap     = document.getElementById("upload-preview-wrap");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const analyzeButton   = document.getElementById("analyze-button");
const processingLabel = document.getElementById("processing-label");
const breedStats      = document.getElementById("breed-stats");
const diseaseStats    = document.getElementById("disease-stats");
const breedInfo       = document.getElementById("breed-info");
const diseaseInfo     = document.getElementById("disease-info");
const breedNarrative  = document.getElementById("breed-narrative");
const diseaseNarrative= document.getElementById("disease-narrative");
const clearButton     = document.getElementById("clear-button");
const scanLines       = document.getElementById("scan-lines");
const btnSpinner      = document.getElementById("btn-spinner");
const btnText         = document.getElementById("btn-text");
const statusDot       = document.getElementById("status-dot");
const statusText      = document.getElementById("status-text");
const breedGrid       = document.getElementById("breed-grid");

// ── Scroll-fade observer ───────────────────────────────────────────────────
function setupFadeIn() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("visible");
          observer.unobserve(e.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );
  document.querySelectorAll(".fade-in").forEach((el) => observer.observe(el));
}

// ── Health check ───────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    const data = await res.json();
    statusDot.classList.add("online");
    statusText.textContent = `Online · ${data.num_breeds} breeds · Disease model: ${data.disease_model_available ? "✓" : "✗"}`;
  } catch {
    statusDot.classList.add("error");
    statusText.textContent = "Server offline";
  }
}

// ── Breed cards ────────────────────────────────────────────────────────────
function buildBreedGrid() {
  Object.entries(BREED_INFO).forEach(([name, desc]) => {
    const card = document.createElement("div");
    card.className = "breed-card";
    card.innerHTML = `
      <div class="breed-name">${name.replace(/_/g, " ")}</div>
      <div class="breed-desc">${desc}</div>
    `;
    breedGrid.appendChild(card);
  });
}

// ── Preview helpers ────────────────────────────────────────────────────────
function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    uploadPlaceholder.classList.add("hidden");
    previewWrap.classList.remove("hidden");
    analyzeButton.disabled = false;
    setStatus("Ready to analyze. Click Analyze Image.", "");
  };
  reader.readAsDataURL(file);
}

function clearPreview() {
  fileInput.value = "";
  previewImage.src = "";
  uploadPlaceholder.classList.remove("hidden");
  previewWrap.classList.add("hidden");
  analyzeButton.disabled = true;
  setStatus("Upload an image to begin", "");
  resetResults();
}

function resetResults() {
  breedStats.innerHTML = `<div class="empty-state">
    <svg viewBox="0 0 48 48" fill="none"><circle cx="24" cy="24" r="22" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3 3" opacity=".3"/><path d="M24 16v8M24 30v2" stroke="currentColor" stroke-width="2" stroke-linecap="round" opacity=".4"/></svg>
    <p>Awaiting analysis…</p></div>`;
  diseaseStats.innerHTML = breedStats.innerHTML;
  breedNarrative.classList.add("hidden");
  diseaseNarrative.classList.add("hidden");
}

// ── Status message ─────────────────────────────────────────────────────────
function setStatus(msg, type) {
  processingLabel.textContent = msg;
  processingLabel.className = "status-msg" + (type ? ` ${type}` : "");
}

// ── Confidence bars ────────────────────────────────────────────────────────
function buildConfidenceRows(container, preds, colorFn) {
  container.innerHTML = "";
  if (!preds || preds.length === 0) {
    container.innerHTML = `<div class="empty-state"><p>No predictions returned.</p></div>`;
    return;
  }

  preds.forEach((pred, idx) => {
    const pct = Math.round(pred.score * 100);
    const colorClass = colorFn ? (colorFn(pred.label) || "") : "";
    const topClass = idx === 0 ? " top-1" : "";

    const row = document.createElement("div");
    row.className = `confidence-row${topClass}`;
    row.innerHTML = `
      <div class="conf-header">
        <span class="conf-label">${pred.label.replace(/_/g, " ")}</span>
        <span class="conf-pct">${pct}%</span>
      </div>
      <div class="conf-bar-outer">
        <div class="conf-bar-inner${colorClass ? " " + colorClass : ""}" data-pct="${pct / 100}"></div>
      </div>
    `;
    container.appendChild(row);
  });

  // Animate bars after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll(".conf-bar-inner").forEach((bar) => {
        const val = Math.max(parseFloat(bar.dataset.pct), 0.03);
        bar.style.transform = `scaleX(${val})`;
      });
    });
  });
}

// ── Narrative renderer ─────────────────────────────────────────────────────
function renderNarrative(boxEl, textEl, markdown) {
  if (!markdown) { boxEl.classList.add("hidden"); return; }
  // Bold leading **text**
  const html = markdown
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br/>");
  textEl.innerHTML = html;
  boxEl.classList.remove("hidden");
}

// ── Analyze ────────────────────────────────────────────────────────────────
async function analyze() {
  const file = fileInput.files[0];
  if (!file) { setStatus("Please upload an image first.", "error"); return; }

  analyzeButton.disabled = true;
  btnSpinner.classList.remove("hidden");
  btnText.textContent = "Analyzing…";
  scanLines.classList.add("active");
  setStatus("Analyzing with Hybrid CNN + ViT…", "analyzing");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/predict", { method: "POST", body: formData });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `HTTP ${res.status}`);
    }

    const data = await res.json();

    buildConfidenceRows(breedStats, data.breed_predictions || [], null);
    buildConfidenceRows(
      diseaseStats,
      data.disease_predictions || [],
      (label) => DISEASE_COLORS[label.toLowerCase()] || ""
    );

    renderNarrative(breedNarrative, breedInfo, data.breed_info || "");
    renderNarrative(diseaseNarrative, diseaseInfo, data.disease_info || "");

    setStatus("Analysis complete ✓", "done");
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    analyzeButton.disabled = false;
    btnSpinner.classList.add("hidden");
    btnText.textContent = "Analyze Image";
    scanLines.classList.remove("active");
  }
}

// ── Event wiring ───────────────────────────────────────────────────────────
uploadArea.addEventListener("click", (e) => {
  if (clearButton.contains(e.target)) return;
  fileInput.click();
});
uploadArea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("drag-over");
});
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  const file = e.dataTransfer.files?.[0];
  if (file && file.type.startsWith("image/")) {
    // Set files on the input
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    showPreview(file);
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) showPreview(file);
});

analyzeButton.addEventListener("click", analyze);

clearButton.addEventListener("click", (e) => {
  e.stopPropagation();
  clearPreview();
});

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupFadeIn();
  buildBreedGrid();
  checkHealth();
  // Trigger hero fade
  setTimeout(() => {
    document.querySelector(".hero")?.classList.add("visible");
    document.querySelector(".analyzer-section .panel:first-child")?.classList.add("visible");
  }, 100);
});
