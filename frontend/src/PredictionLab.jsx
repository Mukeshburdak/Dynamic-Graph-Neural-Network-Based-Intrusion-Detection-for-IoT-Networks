
import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  LoaderCircle,
  Play,
  RotateCcw,
  Shield,
  ShieldAlert,
} from "lucide-react";

import "./PredictionLab.css";

const MODELS = ["RandomForest", "XGBoost", "LogisticRegression"];

function formatModelName(name) {
  if (name === "RandomForest") return "Random Forest";
  if (name === "LogisticRegression") return "Logistic Regression";
  return name;
}

function PredictionLab() {
  const [features, setFeatures] = useState([]);
  const [values, setValues] = useState({});
  const [selectedModel, setSelectedModel] = useState("RandomForest");

  const [loadingFeatures, setLoadingFeatures] = useState(true);
  const [loadingSample, setLoadingSample] = useState(false);
  const [predicting, setPredicting] = useState(false);

  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [actualClass, setActualClass] = useState("");

  // Load the selected model features from Flask.
  useEffect(() => {
    let cancelled = false;

    async function loadFeatures() {
      try {
        setLoadingFeatures(true);
        setError("");

        const response = await fetch("/api/feature-names");

        if (!response.ok) {
          throw new Error("Could not load prediction features.");
        }

        const data = await response.json();

        if (!Array.isArray(data) || data.length === 0) {
          throw new Error("The backend returned an empty feature list.");
        }

        if (cancelled) return;

        setFeatures(data);

        const initialValues = {};
        data.forEach((feature) => {
          initialValues[feature] = "0";
        });

        setValues(initialValues);
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Unable to load features.");
        }
      } finally {
        if (!cancelled) {
          setLoadingFeatures(false);
        }
      }
    }

    loadFeatures();

    return () => {
      cancelled = true;
    };
  }, []);

  // Count valid numeric input fields.
  const validFeatureCount = useMemo(() => {
    return features.filter((feature) => {
      const value = values[feature];

      return (
        value !== "" &&
        value !== undefined &&
        Number.isFinite(Number(value))
      );
    }).length;
  }, [features, values]);

  // Update a feature when the user changes its value.
  function handleFeatureChange(feature, value) {
    setValues((previous) => ({
      ...previous,
      [feature]: value,
    }));

    setResult(null);
    setActualClass("");
    setError("");
  }

  // Reset every feature to zero.
  function resetFeatures() {
    const resetValues = {};

    features.forEach((feature) => {
      resetValues[feature] = "0";
    });

    setValues(resetValues);
    setResult(null);
    setActualClass("");
    setError("");
  }

  // Load one real sample from the processed test dataset.
  async function loadSampleFlow() {
    try {
      setLoadingSample(true);
      setError("");
      setResult(null);
      setActualClass("");

      const response = await fetch("/api/sample-flow");
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Could not load sample flow.");
      }

      if (!data.features || typeof data.features !== "object") {
        throw new Error("The sample response has an invalid feature format.");
      }

      const sampleValues = {};

      features.forEach((feature) => {
        sampleValues[feature] = String(data.features[feature] ?? 0);
      });

      setValues(sampleValues);
      setActualClass(data.actual_class || "");
    } catch (err) {
      setError(err.message || "Unable to load sample data.");
    } finally {
      setLoadingSample(false);
    }
  }

  // Submit feature values to the Flask prediction API.
  async function handlePrediction(event) {
    event.preventDefault();

    setError("");
    setResult(null);

    if (features.length === 0) {
      setError("No prediction features are available.");
      return;
    }

    const invalidFeature = features.find((feature) => {
      const value = values[feature];

      return (
        value === "" ||
        value === undefined ||
        !Number.isFinite(Number(value))
      );
    });

    if (invalidFeature) {
      setError(`Enter a valid numeric value for ${invalidFeature}.`);
      return;
    }

    const featurePayload = {};

    features.forEach((feature) => {
      featurePayload[feature] = Number(values[feature]);
    });

    try {
      setPredicting(true);

      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: selectedModel,
          features: featurePayload,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction request failed.");
      }

      if (!data.predicted_class) {
        throw new Error("The backend returned an invalid prediction.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to complete prediction.");
    } finally {
      setPredicting(false);
    }
  }

  // Sort probabilities from highest to lowest.
  const probabilities = result?.probabilities
    ? Object.entries(result.probabilities).sort(
        (a, b) => Number(b[1]) - Number(a[1])
      )
    : [];

  const isBenign =
    result?.predicted_class?.toLowerCase() === "benign";

  // Use the highest returned class probability as confidence.
  const confidence = probabilities.length
    ? Number(probabilities[0][1])
    : null;

  return (
    <div className="prediction-page">
      {/* Introduction */}
      <section className="prediction-intro">
        <div className="prediction-intro-icon">
          <Activity size={25} />
        </div>

        <div>
          <h2>Network Traffic Prediction</h2>
          <p>
            Analyze an individual IoT network flow using trained machine
            learning classification models.
          </p>
        </div>

        <span className="prediction-mode-tag">
          SINGLE-FLOW ANALYSIS
        </span>
      </section>

      {/* Summary cards */}
      <section className="prediction-stats">
        <div className="prediction-stat-card">
          <span>Available features</span>
          <strong>
            {loadingFeatures ? "..." : features.length}
          </strong>
          <small>Selected model input features</small>
        </div>

        <div className="prediction-stat-card">
          <span>Available models</span>
          <strong>{MODELS.length}</strong>
          <small>Trained baseline classifiers</small>
        </div>

        <div className="prediction-stat-card">
          <span>Input completion</span>
          <strong>
            {features.length
              ? `${Math.round(
                  (validFeatureCount / features.length) * 100
                )}%`
              : "—"}
          </strong>
          <small>Valid numeric input fields</small>
        </div>
      </section>

      {/* Prediction form */}
      <form
        className="prediction-form-panel"
        onSubmit={handlePrediction}
      >
        <div className="prediction-header-actions">
          <button
            type="button"
            className="prediction-sample-button"
            onClick={loadSampleFlow}
            disabled={
              loadingFeatures || loadingSample || predicting
            }
          >
            {loadingSample ? (
              <>
                <LoaderCircle className="spin" size={15} />
                Loading sample...
              </>
            ) : (
              "Load test sample"
            )}
          </button>

          <button
            type="button"
            className="prediction-reset-button"
            onClick={resetFeatures}
            disabled={
              loadingFeatures || predicting || loadingSample
            }
          >
            <RotateCcw size={15} />
            Reset values
          </button>
        </div>

        {/* Model selection */}
        <div className="prediction-model-control">
          <label htmlFor="prediction-model">
            Classification model
          </label>

          <select
            id="prediction-model"
            value={selectedModel}
            onChange={(event) => {
              setSelectedModel(event.target.value);
              setResult(null);
              setError("");
            }}
            disabled={predicting || loadingSample}
          >
            {MODELS.map((model) => (
              <option key={model} value={model}>
                {formatModelName(model)}
              </option>
            ))}
          </select>
        </div>

        {/* Feature inputs */}
        <div className="prediction-section-heading feature-heading">
          <div>
            <h3>Network Flow Features</h3>
            <p>
              Enter numeric values for the selected model features.
            </p>
          </div>

          <span className="feature-count">
            {features.length} FEATURES
          </span>
        </div>

        {loadingFeatures ? (
          <div className="prediction-loading">
            <LoaderCircle className="spin" size={22} />
            Loading feature configuration...
          </div>
        ) : features.length > 0 ? (
          <div className="prediction-feature-grid">
            {features.map((feature, index) => (
              <div
                className="prediction-input-group"
                key={feature}
              >
                <label htmlFor={`feature-${index}`}>
                  {feature}
                </label>

                <input
                  id={`feature-${index}`}
                  type="number"
                  step="any"
                  value={values[feature] ?? "0"}
                  onChange={(event) =>
                    handleFeatureChange(
                      feature,
                      event.target.value
                    )
                  }
                  placeholder="Enter value"
                  disabled={predicting || loadingSample}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="prediction-empty">
            No feature configuration is available.
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="prediction-error">
            <AlertTriangle size={17} />
            <span>{error}</span>
          </div>
        )}

        {/* Submit */}
        <div className="prediction-submit-row">
          <div className="prediction-note">
            <Shield size={16} />
            Predictions use the existing trained baseline models.
          </div>

          <button
            type="submit"
            className="prediction-submit-button"
            disabled={
              loadingFeatures ||
              predicting ||
              loadingSample ||
              features.length === 0
            }
          >
            {predicting ? (
              <>
                <LoaderCircle className="spin" size={17} />
                Analyzing traffic...
              </>
            ) : (
              <>
                <Play size={17} />
                Run prediction
              </>
            )}
          </button>
        </div>
      </form>

      {/* Prediction result */}
      {result && (
        <section className="prediction-result-panel">
          <div className="prediction-result-heading">
            <div>
              <h3>Prediction Result</h3>
              <p>
                Analysis completed using{" "}
                {formatModelName(result.model_used)}.
              </p>
            </div>

            <span className="prediction-success-tag">
              <CheckCircle2 size={15} />
              COMPLETED
            </span>
          </div>

          {/* Predicted category */}
          <div
            className={`prediction-verdict ${
              isBenign ? "benign" : "attack"
            }`}
          >
            <div className="verdict-icon">
              {isBenign ? (
                <Shield size={26} />
              ) : (
                <ShieldAlert size={26} />
              )}
            </div>

            <div className="verdict-content">
              <span>Predicted traffic category</span>
              <strong>{result.predicted_class}</strong>

              <small>
                {isBenign
                  ? "The selected model classified this flow as benign."
                  : "The selected model classified this flow as a potential attack."}
              </small>
            </div>

            {confidence !== null && (
              <div className="verdict-confidence">
                <strong>
                  {(confidence * 100).toFixed(2)}%
                </strong>
                <span>Confidence estimate</span>
              </div>
            )}
          </div>

          {/* Actual versus predicted comparison */}
          {actualClass && (
            <div className="prediction-comparison">
              <div>
                <span>Actual dataset category</span>
                <strong>{actualClass}</strong>
              </div>

              <div>
                <span>Model prediction</span>
                <strong>{result.predicted_class}</strong>
              </div>

              <div
                className={
                  actualClass === result.predicted_class
                    ? "comparison-match"
                    : "comparison-mismatch"
                }
              >
                {actualClass === result.predicted_class
                  ? "Correct classification"
                  : "Classification mismatch"}
              </div>
            </div>
          )}

          {/* Class probabilities */}
          {probabilities.length > 0 && (
            <div className="prediction-probabilities">
              <div className="prediction-section-heading">
                <div>
                  <h3>Class Probability Distribution</h3>
                  <p>
                    Probability estimates returned by the selected model.
                  </p>
                </div>
              </div>

              <div className="probability-list">
                {probabilities.map(
                  ([className, probability]) => {
                    const percentage = Math.max(
                      0,
                      Math.min(
                        100,
                        Number(probability) * 100
                      )
                    );

                    return (
                      <div
                        className="probability-row"
                        key={className}
                      >
                        <div className="probability-label">
                          <span>{className}</span>
                          <strong>
                            {percentage.toFixed(2)}%
                          </strong>
                        </div>

                        <div className="probability-track">
                          <div
                            className="probability-fill"
                            style={{
                              width: `${percentage}%`,
                            }}
                          />
                        </div>
                      </div>
                    );
                  }
                )}
              </div>
            </div>
          )}
        </section>
      )}

      {/* Research disclaimer */}
      <div className="prediction-disclaimer">
        <AlertTriangle size={17} />

        <p>
          <strong>Research demonstration:</strong> This page performs
          single-flow classification using trained baseline ML models.
          It does not monitor live network traffic. Prediction quality
          depends on valid inputs and consistency with the training
          preprocessing pipeline. The confidence estimate is the
          highest class probability returned by the model, not a
          guarantee of prediction correctness.
        </p>
      </div>
    </div>
  );
}

export default PredictionLab;
