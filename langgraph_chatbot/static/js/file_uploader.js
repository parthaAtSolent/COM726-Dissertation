/* ══════════════════════════════════════════════════════════════════════
   app/static/js/file_uploader.js
   All interactivity for the custom drag-and-drop file uploader.

   Communicates with Python via Streamlit's component postMessage API:
     • streamlit:componentReady  – signals the component is initialised
     • streamlit:setFrameHeight  – keeps the iframe sized to its content
     • streamlit:setComponentValue – sends file data back to Python as:
         [{ name, mime, size, b64 }, ...]
   ══════════════════════════════════════════════════════════════════════ */

/* ── Streamlit helpers ────────────────────────────────────────────── */

/**
 * Send a value back to the Python layer.
 * @param {Array|null} value
 */
function toStreamlit(value) {
  window.parent.postMessage(
    { isStreamlitMessage: true, type: "streamlit:setComponentValue", value },
    "*"
  );
}

/**
 * Resize the iframe to fit the current content height.
 */
function syncHeight() {
  const h = document.documentElement.scrollHeight || document.body.scrollHeight;
  window.parent.postMessage(
    { isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: h + 8 },
    "*"
  );
}

// Announce readiness to Streamlit
window.parent.postMessage(
  { isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 },
  "*"
);

/* ── Component state ──────────────────────────────────────────────── */
let acceptedTypes = ["pdf", "txt"];   // overridden by render args
let maxFiles      = 10;
let pendingFiles  = [];               // [{ id: int, file: File }]
let uidCounter    = 0;

function uid() { return ++uidCounter; }

/* ── DOM refs ─────────────────────────────────────────────────────── */
const dropZone     = document.getElementById("drop-zone");
const browseBtn    = document.getElementById("browse-btn");
const fileInput    = document.getElementById("file-input");
const fileList     = document.getElementById("file-list");
const progressWrap = document.getElementById("progress-wrap");
const progressBar  = document.getElementById("progress-bar");
const uploadIcon   = document.getElementById("upload-icon");
const mainLabel    = document.getElementById("main-label");
const subLabel     = document.getElementById("sub-label");

/* ── Receive render args from Streamlit ───────────────────────────── */
window.addEventListener("message", function (event) {
  const msg = event.data;
  if (!msg || msg.type !== "streamlit:render") return;

  const args = msg.args || {};

  if (Array.isArray(args.accepted_types) && args.accepted_types.length) {
    acceptedTypes       = args.accepted_types;
    fileInput.accept    = acceptedTypes.map(t => "." + t).join(",");
  }
  if (args.max_files)  maxFiles           = args.max_files;
  if (args.icon)       uploadIcon.textContent = args.icon;
  if (args.label)      mainLabel.textContent  = args.label;

  subLabel.textContent =
    "Accepted: " + acceptedTypes.map(t => "." + t).join(", ") +
    " · Max " + maxFiles + " files";

  syncHeight();
});

/* ── Utilities ────────────────────────────────────────────────────── */

function fmtSize(bytes) {
  if (bytes < 1024)         return bytes + " B";
  if (bytes < 1024 * 1024)  return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/* ── Render staged-file chips ─────────────────────────────────────── */
function renderChips() {
  fileList.innerHTML = "";

  pendingFiles.forEach(function ({ id, file }) {
    const chip = document.createElement("div");
    chip.className = "file-chip";
    chip.innerHTML =
      '<span class="chip-name" title="' + file.name + '">' + file.name + "</span>" +
      '<span class="chip-size">' + fmtSize(file.size) + "</span>" +
      '<span class="chip-remove" data-id="' + id + '" title="Remove">✕</span>';
    fileList.appendChild(chip);
  });

  fileList.querySelectorAll(".chip-remove").forEach(function (btn) {
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      const removeId = parseInt(this.dataset.id, 10);
      pendingFiles = pendingFiles.filter(function (f) { return f.id !== removeId; });
      renderChips();
      if (pendingFiles.length === 0) toStreamlit(null);
      syncHeight();
    });
  });

  syncHeight();
}

/* ── Base64-encode files then send to Python ──────────────────────── */
function encodeAndSend() {
  if (pendingFiles.length === 0) return;

  progressWrap.style.display = "block";
  progressBar.style.width    = "0%";

  const results = [];
  let   done    = 0;
  const total   = pendingFiles.length;

  pendingFiles.forEach(function ({ file }) {
    const reader = new FileReader();

    reader.onload = function (e) {
      // e.target.result  →  "data:<mime>;base64,<data>"
      const b64 = e.target.result.split(",")[1];
      results.push({ name: file.name, mime: file.type || "application/octet-stream", size: file.size, b64: b64 });

      done++;
      progressBar.style.width = Math.round((done / total) * 100) + "%";

      if (done === total) {
        setTimeout(function () {
          progressWrap.style.display = "none";
          progressBar.style.width    = "0%";
        }, 400);
        toStreamlit(results);   // ← Python receives this
      }
    };

    reader.readAsDataURL(file);
  });
}

/* ── Merge new File objects, dedup, respect limits ────────────────── */
function addFiles(newFiles) {
  const allowed = acceptedTypes.map(function (t) { return t.toLowerCase(); });

  Array.from(newFiles).forEach(function (file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext))    return;
    if (pendingFiles.length >= maxFiles) return;
    const dup = pendingFiles.some(function (f) {
      return f.file.name === file.name && f.file.size === file.size;
    });
    if (!dup) pendingFiles.push({ id: uid(), file: file });
  });

  renderChips();
  encodeAndSend();
}

/* ── Event listeners ──────────────────────────────────────────────── */

// Click on drop-zone body → open picker (but not on chips)
dropZone.addEventListener("click", function (e) {
  if (e.target === browseBtn || e.target.closest(".chip-remove")) return;
  fileInput.click();
});

browseBtn.addEventListener("click", function (e) {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener("change", function () {
  addFiles(this.files);
  this.value = "";   // reset so same file can be reselected after removal
});

// Keyboard accessibility on the drop-zone
dropZone.addEventListener("keydown", function (e) {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

// Drag-and-drop
dropZone.addEventListener("dragover",  function (e) { e.preventDefault(); this.classList.add("dragging"); });
dropZone.addEventListener("dragleave", function ()  { this.classList.remove("dragging"); });
dropZone.addEventListener("drop",      function (e) {
  e.preventDefault();
  this.classList.remove("dragging");
  addFiles(e.dataTransfer.files);
});

// Initial height sync once the DOM is ready
window.addEventListener("load", syncHeight);