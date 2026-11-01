
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  Brain,
  CheckCircle2,
  Database,
  Search,
  Trophy,
  RefreshCw,
  TrendingUp,
} from "lucide-react";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Line,
  LineChart,
} from "recharts";

import "./ModelAnalytics.css";

const metricLabels = [
  { key: "accuracy", label: "Accuracy" },
  { key: "macro_f1", label: "Macro F1" },
  { key: "macro_precision", label: "Macro Precision" },
  { key: "macro_recall", label: "Macro Recall" },
];

const BASELINE_MODELS = [
  "RandomForest",
  "XGBoost",
  "LogisticRegression",
];

function percentage(value) {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(Number(value))
  ) {
    return "—";
  }

  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatLoss(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "—";
}

export default function ModelAnalytics() {
  const [models, setModels] = useState([]);
  const [features, setFeatures] = useState([]);
  const [search, setSearch] = useState("");

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedModel, setSelectedModel] = useState("");
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [confusionData, setConfusionData] = useState(null);

  const [historyLoading, setHistoryLoading] = useState(false);
  const [matrixLoading, setMatrixLoading] = useState(false);

  const [historyError, setHistoryError] = useState("");
  const [matrixError, setMatrixError] = useState("");

  const [chartMetric, setChartMetric] = useState("accuracy");

  // Load model comparison and feature information.
  const loadAnalytics = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const [comparisonResponse, featureResponse] = await Promise.all([
        fetch("/api/comparison"),
        fetch("/api/feature-names"),
      ]);

      if (!comparisonResponse.ok) {
        throw new Error(
          `Model comparison API failed: ${comparisonResponse.status}`
        );
      }

      if (!featureResponse.ok) {
        throw new Error(
          `Feature names API failed: ${featureResponse.status}`
        );
      }

      const [comparisonData, featureData] = await Promise.all([
        comparisonResponse.json(),
        featureResponse.json(),
      ]);

      if (!Array.isArray(comparisonData)) {
        throw new Error("Invalid model comparison response.");
      }

      if (!Array.isArray(featureData)) {
        throw new Error("Invalid feature names response.");
      }

      setModels(comparisonData);
      setFeatures(featureData);

      setSelectedModel((current) => {
        if (
          current &&
          comparisonData.some(
            (model) => model.model_name === current
          )
        ) {
          return current;
        }

        return comparisonData[0]?.model_name || "";
      });
    } catch (err) {
      setError(err.message || "Unable to load analytics.");
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial page loading.
  useEffect(() => {
    loadAnalytics();
  }, [loadAnalytics]);

  // Load details independently for the selected model.
  useEffect(() => {
    if (!selectedModel) {
      setTrainingHistory([]);
      setConfusionData(null);
      setHistoryLoading(false);
      setMatrixLoading(false);
      return;
    }

    let cancelled = false;

    const model = encodeURIComponent(selectedModel);

    async function loadTrainingHistory() {
      setHistoryLoading(true);
      setHistoryError("");
      setTrainingHistory([]);

      try {
        const response = await fetch(
          `/api/training-history/${model}`
        );

        if (response.status === 404) {
          return;
        }

        if (!response.ok) {
          throw new Error(
            `Training history API failed: ${response.status}`
          );
        }

        const data = await response.json();

        if (!cancelled) {
          setTrainingHistory(Array.isArray(data) ? data : []);
        }
      } catch (err) {
        if (!cancelled) {
          setHistoryError(
            err.message || "Unable to load training history."
          );
        }
      } finally {
        if (!cancelled) {
          setHistoryLoading(false);
        }
      }
    }

    async function loadConfusionMatrix() {
      setMatrixLoading(true);
      setMatrixError("");
      setConfusionData(null);

      try {
        const response = await fetch(
          `/api/confusion-matrix/${model}`
        );

        if (response.status === 404) {
          return;
        }

        if (!response.ok) {
          throw new Error(
            `Confusion matrix API failed: ${response.status}`
          );
        }

        const data = await response.json();

        if (
          !Array.isArray(data?.labels) ||
          !Array.isArray(data?.matrix)
        ) {
          throw new Error("Invalid confusion matrix response.");
        }

        if (!cancelled) {
          setConfusionData(data);
        }
      } catch (err) {
        if (!cancelled) {
          setMatrixError(
            err.message || "Unable to load confusion matrix."
          );
        }
      } finally {
        if (!cancelled) {
          setMatrixLoading(false);
        }
      }
    }

    loadTrainingHistory();
    loadConfusionMatrix();

    return () => {
      cancelled = true;
    };
  }, [selectedModel]);

  // Sort models by accuracy.
  const sortedModels = useMemo(
    () =>
      [...models].sort(
        (a, b) => Number(b.accuracy) - Number(a.accuracy)
      ),
    [models]
  );

  const bestModel = sortedModels[0];

  // Identify baseline models.
  const isBaselineModel = BASELINE_MODELS.includes(selectedModel);

  // Search available input features.
  const filteredFeatures = useMemo(
    () =>
      features.filter((feature) =>
        String(feature)
          .toLowerCase()
          .includes(search.toLowerCase())
      ),
    [features, search]
  );

  // Training chart configurations.
  const chartSeries = {
    accuracy: [
      {
        key: "train_accuracy",
        name: "Training Accuracy",
        color: "#38bdf8",
      },
      {
        key: "val_accuracy",
        name: "Validation Accuracy",
        color: "#a78bfa",
      },
    ],

    loss: [
      {
        key: "train_loss",
        name: "Training Loss",
        color: "#fbbf24",
      },
      {
        key: "val_loss",
        name: "Validation Loss",
        color: "#fb7185",
      },
    ],

    f1: [
      {
        key: "val_macro_f1",
        name: "Validation Macro F1",
        color: "#34d399",
      },
    ],
  };

  const matrixMaximum = confusionData?.matrix?.length
    ? Math.max(
        ...confusionData.matrix
          .flat()
          .map(Number)
          .filter(Number.isFinite),
        1
      )
    : 1;

  const totalFeatures = features.length;

  // Initial loading state.
  if (loading) {
    return (
      <div className="analytics-message">
        <Activity className="loading-icon" size={20} />
        Loading model analytics...
      </div>
    );
  }

  // Main API error state.
  if (error) {
    return (
      <div className="analytics-message analytics-error">
        <p>{error}</p>

        <button onClick={loadAnalytics}>
          <RefreshCw size={15} />
          Retry
        </button>
      </div>
    );
  }

  return (
    <main className="model-analytics">
      {/* Page Header */}
      <header className="analytics-header">
        <div>
          <p className="analytics-eyebrow">MODEL INTELLIGENCE</p>

          <h1>Model Analytics</h1>

          <p className="analytics-subtitle">
            Performance evaluation of traditional machine learning
            models and graph neural networks.
          </p>
        </div>

        <button
          className="analytics-refresh"
          onClick={loadAnalytics}
          disabled={loading}
        >
          <RefreshCw size={16} />
          Refresh
        </button>
      </header>

      {/* Summary Cards */}
      <section className="analytics-summary">
        <article className="analytics-stat">
          <div className="analytics-stat-icon blue-icon">
            <Brain size={21} />
          </div>

          <span>Total Models</span>
          <strong>{models.length}</strong>
          <small>Models evaluated</small>
        </article>

        <article className="analytics-stat">
          <div className="analytics-stat-icon purple-icon">
            <Database size={21} />
          </div>

          <span>Input Features</span>
          <strong>{totalFeatures}</strong>
          <small>Features used by the dataset pipeline</small>
        </article>

        <article className="analytics-stat">
          <div className="analytics-stat-icon green-icon">
            <Trophy size={21} />
          </div>

          <span>Top Accuracy</span>
          <strong>
            {bestModel ? percentage(bestModel.accuracy) : "—"}
          </strong>
          <small>
            {bestModel?.model_name || "No model available"}
          </small>
        </article>
      </section>

      {/* Model Performance Comparison */}
      <section className="analytics-panel">
        <div className="analytics-panel-heading">
          <div>
            <h2>Model Performance Comparison</h2>
            <p>Compare accuracy, precision, recall and F1 score</p>
          </div>

          <span className="analytics-tag">TEST METRICS</span>
        </div>

        <div className="analytics-chart">
          {sortedModels.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={sortedModels}
                margin={{
                  top: 15,
                  right: 20,
                  left: 0,
                  bottom: 45,
                }}
              >
                <CartesianGrid
                  stroke="#273449"
                  strokeDasharray="3 3"
                />

                <XAxis
                  dataKey="model_name"
                  stroke="#94a3b8"
                  angle={-18}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 11 }}
                />

                <YAxis
                  domain={[0, 1]}
                  stroke="#94a3b8"
                  tickFormatter={(value) =>
                    `${(value * 100).toFixed(0)}%`
                  }
                />

                <Tooltip
                  formatter={(value) => percentage(value)}
                  contentStyle={{
                    background: "#111c2e",
                    border: "1px solid #334155",
                    borderRadius: 8,
                    color: "#fff",
                  }}
                />

                <Bar
                  dataKey="accuracy"
                  name="Accuracy"
                  fill="#38bdf8"
                  radius={[5, 5, 0, 0]}
                />

                <Bar
                  dataKey="macro_f1"
                  name="Macro F1"
                  fill="#a78bfa"
                  radius={[5, 5, 0, 0]}
                />

                <Bar
                  dataKey="macro_precision"
                  name="Macro Precision"
                  fill="#34d399"
                  radius={[5, 5, 0, 0]}
                />

                <Bar
                  dataKey="macro_recall"
                  name="Macro Recall"
                  fill="#fbbf24"
                  radius={[5, 5, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="analytics-message">
              No model comparison data available.
            </div>
          )}
        </div>
      </section>

      {/* Detailed Model Metrics */}
      <section className="analytics-panel">
        <div className="analytics-panel-heading">
          <div>
            <h2>Detailed Model Metrics</h2>
            <p>Compare all evaluated models across four metrics</p>
          </div>

          <CheckCircle2 size={20} color="#34d399" />
        </div>

        <div className="analytics-table-wrapper">
          <table className="analytics-table">
            <thead>
              <tr>
                <th>Model</th>

                {metricLabels.map((metric) => (
                  <th key={metric.key}>{metric.label}</th>
                ))}
              </tr>
            </thead>

            <tbody>
              {sortedModels.map((model, index) => (
                <tr key={model.model_name}>
                  <td>
                    <div className="analytics-model-name">
                      {index === 0 && (
                        <Trophy size={15} color="#fbbf24" />
                      )}

                      {model.model_name}
                    </div>
                  </td>

                  {metricLabels.map((metric) => (
                    <td key={metric.key}>
                      {percentage(model[metric.key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Input Feature Explorer */}
      <section className="analytics-panel">
        <div className="analytics-panel-heading">
          <div>
            <h2>Input Feature Explorer</h2>
            <p>Explore the {features.length} available input features</p>
          </div>

          <span className="analytics-tag">FEATURES</span>
        </div>

        <div className="feature-search">
          <Search size={17} />

          <input
            type="text"
            placeholder="Search features..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>

        <div className="feature-grid">
          {filteredFeatures.map((feature, index) => (
            <div className="feature-item" key={feature}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{feature}</strong>
            </div>
          ))}
        </div>

        {filteredFeatures.length === 0 && (
          <p className="feature-empty">
            No matching features found.
          </p>
        )}
      </section>

      {/* Training Performance */}
      <section className="analytics-panel">
        <div className="analytics-panel-heading">
          <div>
            <h2>Training Performance</h2>

            <p>
              Training and validation metrics across epochs
            </p>
          </div>

          <select
            className="analytics-select"
            value={selectedModel}
            onChange={(event) =>
              setSelectedModel(event.target.value)
            }
          >
            {models.map((model) => (
              <option
                key={model.model_name}
                value={model.model_name}
              >
                {model.model_name}
              </option>
            ))}
          </select>
        </div>

        <div className="metric-tabs">
          <button
            className={chartMetric === "accuracy" ? "active" : ""}
            onClick={() => setChartMetric("accuracy")}
          >
            <TrendingUp size={15} />
            Accuracy
          </button>

          <button
            className={chartMetric === "loss" ? "active" : ""}
            onClick={() => setChartMetric("loss")}
          >
            Loss
          </button>

          <button
            className={chartMetric === "f1" ? "active" : ""}
            onClick={() => setChartMetric("f1")}
          >
            Macro F1
          </button>
        </div>

        {historyLoading ? (
          <div className="analytics-message">
            Loading training history...
          </div>
        ) : historyError ? (
          <div className="analytics-message analytics-error">
            {historyError}
          </div>
        ) : trainingHistory.length > 0 ? (
          <div className="analytics-chart training-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={trainingHistory}
                margin={{
                  top: 15,
                  right: 20,
                  left: 5,
                  bottom: 5,
                }}
              >
                <CartesianGrid
                  stroke="#273449"
                  strokeDasharray="3 3"
                />

                <XAxis
                  dataKey="epoch"
                  stroke="#94a3b8"
                  tick={{ fontSize: 11 }}
                />

                <YAxis
                  stroke="#94a3b8"
                  domain={
                    chartMetric === "loss"
                      ? ["auto", "auto"]
                      : [0, 1]
                  }
                  tickFormatter={(value) =>
                    chartMetric === "loss"
                      ? Number(value).toFixed(2)
                      : `${(value * 100).toFixed(0)}%`
                  }
                />

                <Tooltip
                  formatter={(value) =>
                    chartMetric === "loss"
                      ? formatLoss(value)
                      : percentage(value)
                  }
                  contentStyle={{
                    background: "#111c2e",
                    border: "1px solid #334155",
                    borderRadius: 8,
                    color: "#fff",
                  }}
                />

                {chartSeries[chartMetric].map((series) => (
                  <Line
                    key={series.key}
                    type="monotone"
                    dataKey={series.key}
                    name={series.name}
                    stroke={series.color}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 5 }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : isBaselineModel ? (
          <div className="analytics-message">
            <Brain size={22} />

            <div>
              <strong>
                Epoch-wise training history unavailable
              </strong>

              <p>
                {selectedModel} is a traditional machine learning
                model. Its evaluation metrics are available above,
                but epoch-wise training history was not recorded
                during training.
              </p>
            </div>
          </div>
        ) : (
          <div className="analytics-message">
            No training history available for this model.
          </div>
        )}
      </section>

      {/* Confusion Matrix */}
      <section className="analytics-panel">
        <div className="analytics-panel-heading">
          <div>
            <h2>Confusion Matrix</h2>

            <p>
              Actual versus predicted class distribution for{" "}
              {selectedModel || "selected model"}
            </p>
          </div>

          <span className="analytics-tag">CLASSIFICATION</span>
        </div>

        {matrixLoading ? (
          <div className="analytics-message">
            Loading confusion matrix...
          </div>
        ) : matrixError ? (
          <div className="analytics-message analytics-error">
            {matrixError}
          </div>
        ) : confusionData?.matrix?.length > 0 &&
          confusionData?.labels?.length > 0 ? (
          <>
            <div className="matrix-wrapper">
              <p className="matrix-axis-title">
                Predicted Class →
              </p>

              <div className="matrix-layout">
                <div className="matrix-y-label">
                  Actual Class →
                </div>

                <div className="matrix-scroll">
                  <div
                    className="confusion-matrix"
                    style={{
                      gridTemplateColumns: `repeat(${confusionData.labels.length}, minmax(58px, 1fr))`,
                    }}
                  >
                    {confusionData.labels.map((label) => (
                      <div
                        className="matrix-label"
                        key={`head-${label}`}
                      >
                        {label}
                      </div>
                    ))}

                    {confusionData.matrix.map(
                      (row, rowIndex) =>
                        row.map((value, columnIndex) => {
                          const numericValue =
                            Number(value) || 0;

                          const intensity =
                            numericValue / matrixMaximum;

                          const isCorrect =
                            rowIndex === columnIndex;

                          return (
                            <div
                              key={`${rowIndex}-${columnIndex}`}
                              className={`matrix-cell ${
                                isCorrect
                                  ? "matrix-correct"
                                  : ""
                              }`}
                              style={{
                                background: `rgba(56, 189, 248, ${
                                  0.08 + intensity * 0.75
                                })`,
                              }}
                              title={`Actual: ${
                                confusionData.labels[rowIndex]
                              } | Predicted: ${
                                confusionData.labels[columnIndex]
                              } | Count: ${numericValue}`}
                            >
                              {numericValue}
                            </div>
                          );
                        })
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="matrix-legend">
              <span>
                <i className="legend-blue" />
                Prediction count
              </span>

              <span>
                <i className="legend-green" />
                Correct predictions (diagonal)
              </span>
            </div>
          </>
        ) : (
          <div className="analytics-message">
            No confusion matrix available for this model.
          </div>
        )}
      </section>

      {/* Disclaimer */}
      <p className="analytics-disclaimer">
        Metrics represent values returned by the existing backend
        evaluation APIs. Training history and confusion matrices
        are loaded independently from the selected model's backend
        endpoints. Feature names are retrieved from the backend.
        This page does not independently calculate model performance
        or feature importance.
      </p>
    </main>
  );
}
