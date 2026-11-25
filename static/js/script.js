(() => {
  const EMOTION_META = {
    Angry: { color: "#ef4444", icon: "bi-emoji-angry" },
    Disgust: { color: "#10b981", icon: "bi-emoji-dizzy" },
    Fear: { color: "#a855f7", icon: "bi-emoji-astonished" },
    Happy: { color: "#f59e0b", icon: "bi-emoji-laughing" },
    Sad: { color: "#3b82f6", icon: "bi-emoji-frown" },
    Surprise: { color: "#ec4899", icon: "bi-emoji-surprise" },
    Neutral: { color: "#94a3b8", icon: "bi-emoji-neutral" },
  };

  const ALLOWED_TYPES = new Set(["image/jpeg", "image/jpg", "image/png"]);

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("imageInput");
  const browseBtn = document.getElementById("browseBtn");
  const previewCard = document.getElementById("previewCard");
  const previewImg = document.getElementById("previewImage");
  const fileNameEl = document.getElementById("fileName");
  const fileSizeEl = document.getElementById("fileSize");
  const changeBtn = document.getElementById("changeImageBtn");
  const removeBtn = document.getElementById("removeImageBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const loadingState = document.getElementById("loadingState");
  const resultsSection = document.getElementById("resultsSection");
  const errorAlert = document.getElementById("errorAlert");
  const errorMessage = document.getElementById("errorMessage");
  const dismissErrorBtn = document.getElementById("dismissErrorBtn");
  const analyzeAnotherBtn = document.getElementById("analyzeAnotherBtn");

  // Result elements
  const annotatedImage = document.getElementById("annotatedImage");
  const faceCountBadge = document.getElementById("faceCountBadge");
  const faceSwitcher = document.getElementById("faceSwitcher");
  const activeFaceThumb = document.getElementById("activeFaceThumb");
  const dominantEmotionText = document.getElementById("dominantEmotionText");
  const confidenceScoreText = document.getElementById("confidenceScoreText");
  const activeFaceTitle = document.getElementById("activeFaceTitle");
  const progressBarsContainer = document.getElementById("progressBarsContainer");
  const allFacesGrid = document.getElementById("allFacesGrid");
  const allFacesWrapper = document.getElementById("allFacesWrapper");

  if (!dropzone) return;

  let selectedFile = null;
  let previewObjectUrl = null;
  let latestResponse = null;
  let doughnutChart = null;
  let barChart = null;

  function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.classList.remove("d-none");
    errorAlert.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideError() {
    errorAlert.classList.add("d-none");
    errorMessage.textContent = "";
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function validateFile(file) {
    if (!file) return false;
    const ext = file.name ? file.name.split(".").pop().toLowerCase() : "";
    const validType = ALLOWED_TYPES.has(file.type) || ["jpg", "jpeg", "png"].includes(ext);
    if (!validType) {
      showError("Invalid file format. Please select a JPG, JPEG, or PNG image.");
      return false;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError("Image exceeds the 10 MB file size limit.");
      return false;
    }
    return true;
  }

  function setSelectedFile(file) {
    hideError();
    if (!validateFile(file)) return;

    selectedFile = file;
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
    }
    previewObjectUrl = URL.createObjectURL(file);
    previewImg.src = previewObjectUrl;
    fileNameEl.textContent = file.name || "pasted-image.png";
    fileSizeEl.textContent = formatBytes(file.size);

    dropzone.classList.add("d-none");
    previewCard.classList.remove("d-none");
    resultsSection.classList.add("d-none");
    analyzeBtn.disabled = false;
  }

  function resetAll() {
    hideError();
    selectedFile = null;
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
    fileInput.value = "";
    previewImg.src = "";
    previewCard.classList.add("d-none");
    loadingState.classList.add("d-none");
    resultsSection.classList.add("d-none");
    dropzone.classList.remove("d-none");
    analyzeBtn.disabled = true;
    latestResponse = null;
  }

  // Click / File Picker
  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", () => fileInput.click());

  changeBtn.addEventListener("click", () => fileInput.click());
  removeBtn.addEventListener("click", resetAll);
  analyzeAnotherBtn.addEventListener("click", () => {
    resetAll();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  dismissErrorBtn.addEventListener("click", hideError);

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  });

  // Drag & Drop
  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      setSelectedFile(dt.files[0]);
    }
  });

  // Clipboard Paste (Ctrl + V)
  document.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const blob = item.getAsFile();
        if (blob) {
          const ext = item.type === "image/png" ? "png" : "jpg";
          const pastedFile = new File([blob], `clipboard-image.${ext}`, {
            type: item.type,
          });
          setSelectedFile(pastedFile);
          break;
        }
      }
    }
  });

  function renderProgressBars(probabilities, dominantEmotion) {
    progressBarsContainer.innerHTML = "";
    const entries = Object.entries(probabilities);

    entries.forEach(([label, pct]) => {
      const meta = EMOTION_META[label] || { color: "#6366f1", icon: "bi-emoji-smile" };
      const isDominant = label === dominantEmotion;

      const row = document.createElement("div");
      row.className = `emotion-bar-row mb-2 ${isDominant ? "is-dominant" : ""}`;
      row.innerHTML = `
        <div class="d-flex justify-content-between align-items-center mb-1">
          <div class="d-flex align-items-center gap-2">
            <i class="bi ${meta.icon}" style="color: ${meta.color}; font-size: 1.05rem;"></i>
            <span class="fw-semibold ${isDominant ? "text-white" : "text-light"}">${label}</span>
            ${isDominant ? '<span class="badge rounded-pill text-bg-primary ms-1">Dominant</span>' : ""}
          </div>
          <span class="fw-bold ${isDominant ? "text-white" : "text-secondary"}">${pct.toFixed(1)}%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" data-width="${Math.min(100, Math.max(0, pct))}" style="background: ${meta.color};"></div>
        </div>
      `;
      progressBarsContainer.appendChild(row);
    });

    // Trigger smooth CSS width transition
    requestAnimationFrame(() => {
      progressBarsContainer.querySelectorAll(".progress-fill").forEach((bar) => {
        bar.style.width = `${bar.dataset.width}%`;
      });
    });
  }

  function renderCharts(probabilities) {
    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities);
    const colors = labels.map((lbl) => (EMOTION_META[lbl] ? EMOTION_META[lbl].color : "#6366f1"));

    if (doughnutChart) doughnutChart.destroy();
    if (barChart) barChart.destroy();

    const doughnutCtx = document.getElementById("doughnutChart").getContext("2d");
    doughnutChart = new Chart(doughnutCtx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [
          {
            data: values,
            backgroundColor: colors,
            borderColor: "#0e1630",
            borderWidth: 2,
            hoverOffset: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "62%",
        plugins: {
          legend: {
            position: "right",
            labels: {
              color: "#cbd5e1",
              boxWidth: 12,
              padding: 12,
              font: { size: 12 },
            },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.label}: ${Number(ctx.raw).toFixed(2)}%`,
            },
          },
        },
      },
    });

    const barCtx = document.getElementById("barChart").getContext("2d");
    barChart = new Chart(barCtx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Confidence (%)",
            data: values,
            backgroundColor: colors.map((c) => `${c}cc`),
            borderColor: colors,
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            min: 0,
            max: 100,
            grid: { color: "rgba(148, 163, 184, 0.12)" },
            ticks: {
              color: "#94a3b8",
              callback: (v) => `${v}%`,
            },
          },
          y: {
            grid: { display: false },
            ticks: { color: "#e2e8f0", font: { weight: "600" } },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${Number(ctx.raw).toFixed(2)}%`,
            },
          },
        },
      },
    });
  }

  function displayFaceDetails(face) {
    const meta = EMOTION_META[face.dominant_emotion] || {
      color: "#6366f1",
      icon: "bi-emoji-smile",
    };

    activeFaceTitle.textContent = `Face #${face.face_id} Prediction`;
    activeFaceThumb.src = face.face_crop;
    dominantEmotionText.innerHTML = `<i class="bi ${meta.icon} me-2" style="color: ${meta.color};"></i>${face.dominant_emotion}`;
    confidenceScoreText.textContent = `${face.confidence.toFixed(1)}%`;

    renderProgressBars(face.probabilities, face.dominant_emotion);
    renderCharts(face.probabilities);
  }

  function renderAllFacesSummary(faces) {
    if (faces.length <= 1) {
      allFacesWrapper.classList.add("d-none");
      allFacesGrid.innerHTML = "";
      return;
    }

    allFacesWrapper.classList.remove("d-none");
    allFacesGrid.innerHTML = "";

    faces.forEach((face) => {
      const meta = EMOTION_META[face.dominant_emotion] || {
        color: "#6366f1",
        icon: "bi-emoji-smile",
      };
      const sortedTop = Object.entries(face.probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);

      const col = document.createElement("div");
      col.className = "col-md-6 col-xl-4";
      col.innerHTML = `
        <div class="glass-card p-3 h-100">
          <div class="d-flex align-items-center gap-3 mb-3">
            <img src="${face.face_crop}" alt="Face ${face.face_id}" class="face-thumb" />
            <div>
              <div class="text-secondary small">Face #${face.face_id}</div>
              <div class="h5 mb-0 fw-bold text-white">
                <i class="bi ${meta.icon} me-1" style="color: ${meta.color};"></i>
                ${face.dominant_emotion}
              </div>
              <span class="badge bg-primary-subtle text-primary-emphasis mt-1">
                ${face.confidence.toFixed(1)}% Confidence
              </span>
            </div>
          </div>
          <div class="small text-secondary mb-1">Top Probabilities:</div>
          ${sortedTop
            .map(
              ([lbl, val]) => `
            <div class="d-flex justify-content-between small mb-1">
              <span>${lbl}</span>
              <span class="fw-semibold text-white">${val.toFixed(1)}%</span>
            </div>`
            )
            .join("")}
        </div>
      `;
      allFacesGrid.appendChild(col);
    });
  }

  function renderResults(data) {
    latestResponse = data;
    annotatedImage.src = data.annotated_image;
    faceCountBadge.textContent = `${data.face_count} ${data.face_count === 1 ? "Face" : "Faces"} Detected`;

    // Build face switcher tabs if multiple faces exist
    faceSwitcher.innerHTML = "";
    if (data.faces.length > 1) {
      faceSwitcher.classList.remove("d-none");
      data.faces.forEach((face, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `face-tab-btn ${idx === 0 ? "active" : ""}`;
        btn.textContent = `Face #${face.face_id}: ${face.dominant_emotion} (${face.confidence.toFixed(0)}%)`;
        btn.addEventListener("click", () => {
          faceSwitcher.querySelectorAll(".face-tab-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          displayFaceDetails(face);
        });
        faceSwitcher.appendChild(btn);
      });
    } else {
      faceSwitcher.classList.add("d-none");
    }

    displayFaceDetails(data.faces[0]);
    renderAllFacesSummary(data.faces);

    resultsSection.classList.remove("d-none");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      showError("Please upload, drag & drop, or paste an image first.");
      return;
    }

    hideError();
    analyzeBtn.disabled = true;
    loadingState.classList.remove("d-none");
    resultsSection.classList.add("d-none");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok || !data.success) {
        showError(data.error || "Unable to analyze the image. Please try again.");
        analyzeBtn.disabled = false;
        return;
      }

      renderResults(data);
    } catch (err) {
      showError("Network or server error while communicating with Flask backend.");
    } finally {
      loadingState.classList.add("d-none");
      analyzeBtn.disabled = false;
    }
  });
})();
