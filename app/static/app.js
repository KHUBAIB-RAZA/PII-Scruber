const uploadForm = document.querySelector("#upload-form");
const loadForm = document.querySelector("#load-form");
const chatForm = document.querySelector("#chat-form");
const sendChatButton = document.querySelector("#send-chat");
const analyzeButton = document.querySelector("#retry-analysis");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
let activeStatement = null;
let pendingChatQuestion = null;

function showStatement(payload) {
  pendingChatQuestion = null;
  document.querySelector("#chat-preview").classList.add("hidden");
  activeStatement = payload.id;
  document.querySelector("#statement-id").value = payload.id;
  document.querySelector("#statement-meta").textContent =
    `Statement ID: ${payload.id} | Files: ${payload.files || "(unnamed)"}`;
  document.querySelector("#scrubbed").textContent = payload.scrubbed_text;
  document.querySelector("#analysis").textContent = payload.analysis || "No analysis saved yet.";
  renderAnalysisCoverage(payload);
  renderProviderPayloads(payload.provider_payloads || [], "#provider-payloads");
  const estimate = payload.provider_request_estimate;
  document.querySelector("#request-estimate").textContent = estimate
    ? `Provider request estimate: ${estimate.initial_requests} initial request(s), up to ${estimate.including_one_json_fallback_per_segment} with one JSON fallback per segment; adaptive recovery may use more, capped at ${estimate.hard_run_limit}. Groq quotas may apply.`
    : "";
  analyzeButton.textContent = payload.analysis_status === "incomplete"
    ? "Retry incomplete analysis with Groq"
    : payload.analysis
      ? "Run analysis again with Groq"
      : "Analyze this saved statement with Groq";
  analyzeButton.disabled = false;
  results.classList.remove("hidden");
}

function renderAnalysisCoverage(payload) {
  const banner = document.querySelector("#analysis-coverage");
  const segments = payload.analysis_segments || [];
  if (!segments.length) {
    banner.textContent = "";
    banner.classList.add("hidden");
    return;
  }
  const incomplete = payload.analysis_status === "incomplete";
  const lines = incomplete
    ? segments
      .filter((segment) => segment.status !== "complete")
      .map((segment) => {
        const categories = (segment.categories || []).join(", ") || "unknown";
        return `Segment ${segment.segment} of ${segment.segment_count}: ${segment.status}; ` +
          `${segment.validated_subchunks} validated part(s), ${segment.failed_subchunks} failed part(s); reason: ${categories}.`;
      })
    : [];
  banner.textContent = incomplete
    ? `ANALYSIS INCOMPLETE — only fully validated parts are shown.\n${lines.join("\n")}\nRetry analysis from this saved statement; no re-upload is needed.`
    : `All ${segments.length} source segment(s) returned validated results. This is a concise summary, not an exhaustive transaction listing; inspect the saved scrubbed text above for full extracted coverage.`;
  banner.classList.toggle("complete", !incomplete);
  banner.classList.remove("hidden");
}

function renderProviderPayloads(payloads, selector) {
  const container = document.querySelector(selector);
  container.replaceChildren();
  for (const segment of payloads) {
    const section = document.createElement("section");
    section.className = "payload-segment";
    const title = document.createElement("h3");
    title.textContent = `Segment ${segment.segment} of ${segment.segment_count} (${segment.chunk_characters} scrubbed characters)`;
    section.append(title);
    for (const [label, body] of [
      ["Initial constrained-JSON request", segment.primary],
      ["One-time plain-JSON recovery request", segment.json_recovery],
    ]) {
      const heading = document.createElement("h4");
      heading.textContent = label;
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(body, null, 2);
      section.append(heading, pre);
    }
    container.append(section);
  }
}

async function loadSavedPreview(statementId) {
  if (!statementId) return;
  status.textContent = "Loading saved scrubbed text and building a local-only Groq preview…";
  try {
    const response = await fetch(`/api/statements/${encodeURIComponent(statementId)}/groq-preview`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Could not load saved statement");
    showStatement(payload);
    document.querySelector("#chat-log").replaceChildren();
    status.textContent = payload.analysis_status === "incomplete"
      ? `Loaded saved statement ${payload.id}; its analysis is incomplete and only validated segment results were kept.`
      : `Loaded saved statement ${payload.id}; no provider request was made.`;
  } catch (error) {
    status.textContent = error.message;
  }
}

async function analyzeSavedStatement() {
  if (!activeStatement) return;
  analyzeButton.disabled = true;
  status.textContent = `Sending the displayed scrubbed payload for saved statement ${activeStatement}…`;
  try {
    const response = await fetch(`/api/statements/${activeStatement}/analyze`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis failed");
    const previewResponse = await fetch(`/api/statements/${activeStatement}/groq-preview`);
    const previewPayload = await previewResponse.json();
    if (!previewResponse.ok) throw new Error(previewPayload.detail || "Analysis saved but preview could not reload");
    showStatement(previewPayload);
    status.textContent = payload.analysis_status === "incomplete"
      ? (payload.message || `Analysis incomplete for statement ${payload.id}. Only fully validated segment results were kept; retry when ready.`)
      : `Analysis saved for statement ${payload.id}.`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    analyzeButton.disabled = false;
  }
}

analyzeButton.addEventListener("click", analyzeSavedStatement);

loadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  loadSavedPreview(document.querySelector("#statement-id").value.trim());
});

function addMessage(role, text) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.textContent = text;
  document.querySelector("#chat-log").append(item);
}

function showChatError(stage, error) {
  const message = error instanceof Error ? error.message : "The request failed.";
  addMessage("error", `${stage}: ${message}`);
  status.textContent = `${stage}: ${message}`;
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = uploadForm.querySelector("button");
  button.disabled = true;
  status.textContent = "Scrubbing locally, saving to PostgreSQL, and preparing preview only; Groq is not called…";
  try {
    const form = new FormData();
    for (const file of document.querySelector("#files").files) form.append("files", file);
    const response = await fetch("/api/statements/preview", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Preview request failed");
    showStatement(payload);
    document.querySelector("#chat-log").replaceChildren();
    status.textContent = `Scrubbed text saved as ${payload.id}; inspect the payload and click Analyze when ready.`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeStatement) {
    addMessage("error", "Load or save a statement before asking a question.");
    return;
  }
  const question = document.querySelector("#question");
  const value = question.value.trim();
  if (!value) return;
  const preview = document.querySelector("#chat-preview");
  preview.classList.add("hidden");
  status.textContent = "Building local-only preview of the sanitized question and exact Groq payload…";
  try {
    const response = await fetch(`/api/statements/${activeStatement}/chat-preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Could not preview chat request");
    pendingChatQuestion = payload.question;
    const localOnly = Boolean(payload.local_only);
    renderProviderPayloads(payload.provider_payloads, "#chat-provider-payloads");
    const previewMessage = document.querySelector("#chat-preview-message");
    previewMessage.textContent = localOnly
      ? `${payload.message} This response is generated locally; no statement content will be sent to Groq.`
      : "This includes the sanitized question, recent sanitized chat history, system prompt, and scrubbed statement segment. It is not sent until you click the button below.";
    sendChatButton.textContent = localOnly
      ? "Show this local explanation"
      : "Send this preview to Groq";
    preview.classList.remove("hidden");
    status.textContent = localOnly
      ? "This question requires an aggregate the app does not calculate; review the local-only explanation."
      : "Review the exact sanitized chat payload; no Groq request has been made.";
  } catch (error) {
    pendingChatQuestion = null;
    showChatError("Could not prepare question", error);
  }
});

document.querySelector("#question").addEventListener("input", () => {
  pendingChatQuestion = null;
  document.querySelector("#chat-preview").classList.add("hidden");
});

sendChatButton.addEventListener("click", async () => {
  if (!activeStatement || !pendingChatQuestion) return;
  const question = pendingChatQuestion;
  sendChatButton.disabled = true;
  status.textContent = "Sending the displayed sanitized question and context to Groq…";
  try {
    const response = await fetch(`/api/statements/${activeStatement}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Chat request failed");
    addMessage("user", question);
    addMessage("assistant", payload.answer);
    document.querySelector("#question").value = "";
    document.querySelector("#chat-preview").classList.add("hidden");
    pendingChatQuestion = null;
    status.textContent = payload.local_only
      ? "Local explanation shown; no Groq request was made."
      : "Chat response received.";
  } catch (error) {
    showChatError("Could not answer question", error);
  } finally {
    sendChatButton.disabled = false;
  }
});
