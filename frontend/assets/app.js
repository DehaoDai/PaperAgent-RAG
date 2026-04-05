const healthBadge = document.getElementById("healthBadge");
const uploadForm = document.getElementById("uploadForm");
const datasetForm = document.getElementById("datasetForm");
const synthesisForm = document.getElementById("synthesisForm");
const indexForm = document.getElementById("indexForm");
const queryForm = document.getElementById("queryForm");
const evaluationForm = document.getElementById("evaluationForm");
const answerBox = document.getElementById("answerBox");
const debugBox = document.getElementById("debugBox");
const evidenceGrid = document.getElementById("evidenceGrid");

function setButtonBusy(form, busy, text) {
  const button = form.querySelector("button");
  if (!button) return;
  if (!button.dataset.originalText) {
    button.dataset.originalText = button.textContent;
  }
  button.disabled = busy;
  button.textContent = busy ? text : button.dataset.originalText;
}

function showAnswer(text) {
  answerBox.classList.remove("empty");
  answerBox.textContent = text;
}

function showDebug(payload) {
  debugBox.textContent = JSON.stringify(payload, null, 2);
}

function showSynthesisSummary(results) {
  answerBox.classList.remove("empty");
  if (!results?.length) {
    answerBox.textContent = "Synthetic QA finished, but no pairs were returned.";
    return;
  }
  answerBox.textContent = results
    .map((item) => {
      const status = item.accepted ? "ACCEPTED" : "REJECTED";
      return [
        `[${status}] ${item.document_id} p${item.page_number}`,
        `Q: ${item.question}`,
        `A: ${item.answer}`,
        `Groundedness: ${item.groundedness_passed}${item.groundedness_judgment ? ` (${item.groundedness_judgment})` : ""}`,
        `Standalone: ${item.standalone_passed}${item.standalone_judgment ? ` (${item.standalone_judgment})` : ""}`,
      ].join("\n");
    })
    .join("\n");
}

function showEvaluationSummary(data) {
  const aggregate = data.aggregate || {};
  answerBox.classList.remove("empty");
  answerBox.textContent = [
    `Run ID: ${data.run_id}`,
    `Processed examples: ${data.processed_examples}`,
    `Mean raw score: ${aggregate.mean_raw_score ?? "n/a"}`,
    `Mean normalized score: ${aggregate.mean_normalized_score ?? "n/a"}`,
    `Page-balanced normalized score: ${aggregate.page_balanced_normalized_score ?? "n/a"}`,
    `Weighted normalized score: ${aggregate.weighted_normalized_score ?? "n/a"}`,
    `Weights -> question: ${aggregate.question_weight ?? "n/a"}, page: ${aggregate.page_weight ?? "n/a"}`,
    `Report: ${data.report_path}`,
  ].join("\n");
}

function absoluteImagePathToUrl(imagePath) {
  const marker = "/workspace_data/rendered_pages/";
  const index = imagePath.indexOf(marker);
  if (index === -1) return imagePath;
  return "/rendered_pages/" + imagePath.slice(index + marker.length);
}

function renderEvidences(evidences) {
  evidenceGrid.innerHTML = "";
  for (const item of evidences || []) {
    const card = document.createElement("article");
    card.className = "evidence-card";

    const img = document.createElement("img");
    img.alt = `${item.title} page ${item.page_number}`;
    if (item.image_path) {
      img.src = absoluteImagePathToUrl(item.image_path);
    }

    const meta = document.createElement("div");
    meta.className = "evidence-meta";
    meta.innerHTML = `
      <strong>${item.title}</strong>
      <p>Page ${item.page_number}</p>
      <p>score: ${item.score ?? "n/a"}</p>
      <p>${item.rationale ?? ""}</p>
    `;

    card.append(img, meta);
    evidenceGrid.append(card);
  }
}

async function apiJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || JSON.stringify(data));
  }
  return data;
}

async function refreshHealth() {
  try {
    const data = await apiJson("/health");
    healthBadge.textContent = data.status === "ok" ? "Service healthy" : "Service status unknown";
    healthBadge.classList.remove("muted");
  } catch (error) {
    healthBadge.textContent = "Service unavailable";
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setButtonBusy(uploadForm, true, "Uploading...");
  try {
    const formData = new FormData(uploadForm);
    const data = await apiJson("/documents/upload", {
      method: "POST",
      body: formData,
    });
    showAnswer(`Upload succeeded: ${data.title}, rendered ${data.pages.length} pages.`);
    showDebug(data);
    renderEvidences(data.pages.slice(0, 3).map((page) => ({
      title: data.title,
      page_number: page.page_number,
      image_path: page.image_path,
      score: null,
      rationale: "Preview of rendered pages after registration.",
    })));
  } catch (error) {
    showAnswer(`Upload failed: ${error.message}`);
    showDebug({ error: error.message });
  } finally {
    setButtonBusy(uploadForm, false);
  }
});

datasetForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setButtonBusy(datasetForm, true, "Importing...");
  try {
    const form = new FormData(datasetForm);
    const payload = {
      dataset_name: form.get("dataset_name"),
      split: form.get("split"),
      start_index: Number(form.get("start_index")),
      limit: Number(form.get("limit")),
      document_id_prefix: form.get("document_id_prefix"),
    };
    const data = await apiJson("/datasets/pdfvqa/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showAnswer(
      `Imported ${data.imported_count} documents from ${data.dataset_name} (${data.split}).`
    );
    showDebug(data);
    evidenceGrid.innerHTML = "";
  } catch (error) {
    showAnswer(`Dataset import failed: ${error.message}`);
    showDebug({ error: error.message });
  } finally {
    setButtonBusy(datasetForm, false);
  }
});

synthesisForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setButtonBusy(synthesisForm, true, "Generating QA...");
  try {
    const form = new FormData(synthesisForm);
    const rawDocumentIds = String(form.get("document_ids") || "").trim();
    const rawPromptTemplate = String(form.get("prompt_template") || "").trim();
    const payload = {
      document_ids: rawDocumentIds
        ? rawDocumentIds.split(",").map((item) => item.trim()).filter(Boolean)
        : null,
      limit: Number(form.get("limit")),
      model_name: form.get("model_name"),
      questions_per_page: Number(form.get("questions_per_page")),
      max_new_tokens: Number(form.get("max_new_tokens")),
      overwrite_synthetic: form.get("overwrite_synthetic") === "on",
      filter_groundedness: form.get("filter_groundedness") === "on",
      filter_standalone: form.get("filter_standalone") === "on",
      prompt_template: rawPromptTemplate || null,
    };
    const data = await apiJson("/datasets/pdfvqa/synthesize-qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showSynthesisSummary(data.synthesized_pairs);
    showDebug(data);
    evidenceGrid.innerHTML = "";
  } catch (error) {
    showAnswer(`Synthetic QA generation failed: ${error.message}`);
    showDebug({ error: error.message });
  } finally {
    setButtonBusy(synthesisForm, false);
  }
});

indexForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setButtonBusy(indexForm, true, "Building index...");
  try {
    const form = new FormData(indexForm);
    const payload = {
      index_name: form.get("index_name"),
      model_name: form.get("model_name"),
      overwrite: form.get("overwrite") === "on",
    };
    const data = await apiJson("/indices/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showAnswer(`Index build completed: ${data.index_name}, indexed ${data.indexed_pages} pages.`);
    showDebug(data);
  } catch (error) {
    showAnswer(`Index build failed: ${error.message}`);
    showDebug({ error: error.message });
  } finally {
    setButtonBusy(indexForm, false);
  }
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setButtonBusy(queryForm, true, "Answering...");
  try {
    const form = new FormData(queryForm);
    const payload = {
      question: form.get("question"),
      top_k: Number(form.get("top_k")),
      index_name: form.get("index_name"),
      use_reranker: form.get("use_reranker") === "on",
      reranker_model_name: form.get("reranker_model_name"),
      use_generation_model: form.get("use_generation_model") === "on",
      generation_model_name: form.get("generation_model_name"),
      generation_max_images: Number(form.get("generation_max_images")),
    };
    const data = await apiJson("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showAnswer(data.answer);
    showDebug(data.debug);
    renderEvidences(data.evidences);
  } catch (error) {
    showAnswer(`Query failed: ${error.message}`);
    showDebug({ error: error.message });
  } finally {
    setButtonBusy(queryForm, false);
  }
});

evaluationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setButtonBusy(evaluationForm, true, "Running evaluation...");
  try {
    const form = new FormData(evaluationForm);
    const rawDocumentIds = String(form.get("document_ids") || "").trim();
    const rawQaSources = String(form.get("qa_sources") || "").trim();
    const payload = {
      document_ids: rawDocumentIds
        ? rawDocumentIds.split(",").map((item) => item.trim()).filter(Boolean)
        : null,
      qa_sources: rawQaSources
        ? rawQaSources.split(",").map((item) => item.trim()).filter(Boolean)
        : null,
      limit: Number(form.get("limit")),
      judge_model_name: form.get("judge_model_name"),
      question_weight: Number(form.get("question_weight")),
      page_weight: Number(form.get("page_weight")),
      top_k: Number(form.get("top_k")),
      index_name: form.get("index_name"),
      use_reranker: form.get("use_reranker") === "on",
      reranker_model_name: form.get("reranker_model_name"),
      use_generation_model: true,
      generation_model_name: form.get("generation_model_name"),
      generation_max_images: Number(form.get("generation_max_images")),
    };
    const data = await apiJson("/evaluations/pdfvqa/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showEvaluationSummary(data);
    showDebug(data);
    evidenceGrid.innerHTML = "";
  } catch (error) {
    showAnswer(`Evaluation failed: ${error.message}`);
    showDebug({ error: error.message });
  } finally {
    setButtonBusy(evaluationForm, false);
  }
});

refreshHealth();
