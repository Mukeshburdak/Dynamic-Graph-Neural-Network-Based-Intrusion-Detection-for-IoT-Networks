/* ===========================================================
   dashboard/static/js/dashboard.js
   Fetches data from the Flask API routes in app.py and renders
   charts (Chart.js) + drives the live prediction demo.
=========================================================== */

const CHART_COLORS = [
    "#5eead4", "#60a5fa", "#f472b6", "#facc15",
    "#a78bfa", "#fb923c", "#34d399", "#f87171",
];

document.addEventListener("DOMContentLoaded", () => {
    initComparisonChart();
    initClassDistributionChart();
    initTrainingCurves();
    initConfusionMatrixSelector();
    initPredictionForm();
});

// ---------------------------------------------------------------
// 1. MODEL COMPARISON CHART
// ---------------------------------------------------------------
async function initComparisonChart() {
    const res = await fetch("/api/comparison");
    if (!res.ok) return;
    const data = await res.json();
    if (data.error) return;

    const ctx = document.getElementById("comparisonChart");
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: data.map(d => d.model_name),
            datasets: [
                {
                    label: "Accuracy",
                    data: data.map(d => d.accuracy),
                    backgroundColor: CHART_COLORS[0],
                },
                {
                    label: "Macro F1",
                    data: data.map(d => d.macro_f1),
                    backgroundColor: CHART_COLORS[1],
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 1, ticks: { color: "#8b90a1" } },
                x: { ticks: { color: "#8b90a1" } },
            },
            plugins: {
                legend: { labels: { color: "#e6e8ee" } },
            },
        },
    });
}

// ---------------------------------------------------------------
// 2. CLASS DISTRIBUTION CHART
// ---------------------------------------------------------------
async function initClassDistributionChart() {
    const res = await fetch("/api/class-distribution");
    if (!res.ok) return;
    const data = await res.json();
    if (data.error) return;

    const labels = Object.keys(data);
    const values = Object.values(data);

    const ctx = document.getElementById("classDistributionChart");
    new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Row count",
                data: values,
                backgroundColor: labels.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, ticks: { color: "#8b90a1" } },
                x: { ticks: { color: "#8b90a1" } },
            },
        },
    });
}

// ---------------------------------------------------------------
// 3. TRAINING CURVES (loss + F1, model-selectable)
// ---------------------------------------------------------------
let lossChartInstance = null;
let f1ChartInstance = null;

async function loadTrainingCurves(modelName) {
    const res = await fetch(`/api/training-history/${modelName}`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.error) return;

    const epochs = data.map(d => d.epoch);

    if (lossChartInstance) lossChartInstance.destroy();
    if (f1ChartInstance) f1ChartInstance.destroy();

    lossChartInstance = new Chart(document.getElementById("trainingLossChart"), {
        type: "line",
        data: {
            labels: epochs,
            datasets: [
                { label: "Train Loss", data: data.map(d => d.train_loss), borderColor: CHART_COLORS[0], fill: false },
                { label: "Val Loss", data: data.map(d => d.val_loss), borderColor: CHART_COLORS[2], fill: false },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: "#e6e8ee" } }, title: { display: true, text: "Loss", color: "#e6e8ee" } },
            scales: { y: { ticks: { color: "#8b90a1" } }, x: { ticks: { color: "#8b90a1" } } },
        },
    });

    f1ChartInstance = new Chart(document.getElementById("trainingF1Chart"), {
        type: "line",
        data: {
            labels: epochs,
            datasets: [
                { label: "Val Macro-F1", data: data.map(d => d.val_macro_f1), borderColor: CHART_COLORS[3], fill: false },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: "#e6e8ee" } }, title: { display: true, text: "Validation Macro-F1", color: "#e6e8ee" } },
            scales: { y: { beginAtZero: true, max: 1, ticks: { color: "#8b90a1" } }, x: { ticks: { color: "#8b90a1" } } },
        },
    });
}

function initTrainingCurves() {
    const select = document.getElementById("trainingModelSelect");
    loadTrainingCurves(select.value);
    select.addEventListener("change", () => loadTrainingCurves(select.value));
}

// ---------------------------------------------------------------
// 4. CONFUSION MATRIX IMAGE SELECTOR
// ---------------------------------------------------------------
function initConfusionMatrixSelector() {
    const select = document.getElementById("cmModelSelect");
    const img = document.getElementById("cmImage");
    const fallback = document.getElementById("cmFallback");

    select.addEventListener("change", () => {
        img.style.display = "block";
        fallback.style.display = "none";
        img.src = `/figures/confusion_matrix_${select.value}.png?t=${Date.now()}`;
    });
}

// ---------------------------------------------------------------
// 5. LIVE PREDICTION DEMO
// ---------------------------------------------------------------
let probabilityChartInstance = null;
let featureNamesCache = [];

async function initPredictionForm() {
    const res = await fetch("/api/feature-names");
    if (!res.ok) return;
    featureNamesCache = await res.json();

    const container = document.getElementById("featureFormContainer");
    container.innerHTML = "";

    featureNamesCache.forEach(name => {
        const field = document.createElement("div");
        field.className = "feature-field";
        field.innerHTML = `
            <label for="feat-${name}" title="${name}">${name}</label>
            <input type="number" step="any" id="feat-${name}" data-feature="${name}" placeholder="0">
        `;
        container.appendChild(field);
    });

    document.getElementById("loadSampleBtn").addEventListener("click", fillRandomValues);
    document.getElementById("predictBtn").addEventListener("click", runPrediction);
}

function fillRandomValues() {
    featureNamesCache.forEach(name => {
        const input = document.getElementById(`feat-${name}`);
        // Random value roughly in a typical scaled-feature-adjacent range;
        // this is purely for demo convenience, not realistic traffic data.
        input.value = (Math.random() * 4 - 2).toFixed(3);
    });
}

async function runPrediction() {
    const errorEl = document.getElementById("predictionError");
    const resultEl = document.getElementById("predictionResult");
    errorEl.style.display = "none";
    resultEl.style.display = "none";

    const modelKey = document.getElementById("predictModelSelect").value;
    const features = {};
    featureNamesCache.forEach(name => {
        const input = document.getElementById(`feat-${name}`);
        const val = parseFloat(input.value);
        features[name] = isNaN(val) ? 0.0 : val;
    });

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: modelKey, features }),
        });
        const data = await res.json();

        if (!res.ok || data.error) {
            errorEl.textContent = data.error || "Prediction failed.";
            errorEl.style.display = "block";
            return;
        }

        document.getElementById("predictedClass").textContent = data.predicted_class;
        resultEl.style.display = "block";

        if (data.probabilities) {
            const labels = Object.keys(data.probabilities);
            const values = Object.values(data.probabilities);

            if (probabilityChartInstance) probabilityChartInstance.destroy();
            probabilityChartInstance = new Chart(document.getElementById("probabilityChart"), {
                type: "bar",
                data: {
                    labels,
                    datasets: [{
                        label: "Predicted probability",
                        data: values,
                        backgroundColor: labels.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
                    }],
                },
                options: {
                    indexAxis: "y",
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { beginAtZero: true, max: 1, ticks: { color: "#8b90a1" } },
                        y: { ticks: { color: "#8b90a1" } },
                    },
                },
            });
        }
    } catch (err) {
        errorEl.textContent = `Request failed: ${err.message}`;
        errorEl.style.display = "block";
    }
}
