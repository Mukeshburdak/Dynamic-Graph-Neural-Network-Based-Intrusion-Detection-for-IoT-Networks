
import GraphExplorer from "./GraphExplorer";
import ThreatMonitoring from "./ThreatMonitoring";
import ModelAnalytics from "./ModelAnalytics";
import PredictionLab from "./PredictionLab";

import { useEffect, useState } from "react";

import {
  Activity,
  Shield,
  ShieldAlert,
  Network,
  BrainCircuit,
  LayoutDashboard,
  GitBranch,
  Bell,
  FlaskConical,
  RefreshCw,
} from "lucide-react";

import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
} from "recharts";

import "./App.css";

const navigation = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Graph Explorer", icon: GitBranch },
  { label: "Threat Monitoring", icon: Bell },
  { label: "Model Analytics", icon: BrainCircuit },
  { label: "Prediction Lab", icon: FlaskConical },
];

const COLORS = [
  "#22c55e",
  "#ef4444",
  "#f97316",
  "#eab308",
  "#a855f7",
  "#06b6d4",
  "#ec4899",
  "#64748b",
];

function MetricCard({ title, value, description, icon: Icon }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <Icon size={19} />
      </div>
      <h2>{value}</h2>
      <p>{description}</p>
    </div>
  );
}

function App() {
  const [comparison, setComparison] = useState([]);
  const [distribution, setDistribution] = useState([]);
  const [loading, setLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState("Checking");
  const [activePage, setActivePage] = useState("Overview");

  async function loadDashboard() {
    setLoading(true);

    try {
      const [comparisonResponse, distributionResponse] =
        await Promise.all([
          fetch("/api/comparison"),
          fetch("/api/class-distribution"),
        ]);

      if (!comparisonResponse.ok || !distributionResponse.ok) {
        throw new Error("Dashboard data is not available yet");
      }

      const comparisonData = await comparisonResponse.json();
      const distributionData = await distributionResponse.json();

      setComparison(comparisonData);

      setDistribution(
        Object.entries(distributionData).map(([name, value]) => ({
          name,
          value,
        }))
      );

      setApiStatus("Connected");
    } catch (error) {
      console.error(error);
      setApiStatus("Unavailable");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard();
  }, []);

  const bestModel = comparison.reduce(
    (best, current) =>
      !best || Number(current.accuracy) > Number(best.accuracy)
        ? current
        : best,
    null
  );

  const totalFlows = distribution.reduce(
    (sum, item) => sum + Number(item.value || 0),
    0
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Shield size={23} />
          </div>
          <div>
            <h2>IoT Sentinel</h2>
            <span>INTRUSION DETECTION</span>
          </div>
        </div>

        <div className="nav-heading">SECURITY CENTER</div>

        <nav>
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-item ${
                activePage === label ? "active" : ""
              }`}
              onClick={() => setActivePage(label)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" />
            <div>
              <strong>System status</strong>
              <small>{apiStatus}</small>
            </div>
          </div>
          <span className="version">
            Dynamic GNN · CICIoT2023
          </span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">
              SECURITY OPERATIONS CENTER
            </div>
            <h1>{activePage}</h1>
          </div>

          <div className="topbar-actions">
            <span className="connection">
              <span className="status-dot" />
              {apiStatus}
            </span>

            <button
              className="icon-button"
              onClick={loadDashboard}
              title="Refresh dashboard"
            >
              <RefreshCw size={18} />
            </button>

            <div className="avatar">ID</div>
          </div>
        </header>

        {activePage === "Graph Explorer" ? (
          <GraphExplorer />
        ) : activePage === "Threat Monitoring" ? (
          <ThreatMonitoring />
        ) : activePage === "Model Analytics" ? (
          <ModelAnalytics />
        ) : activePage === "Prediction Lab" ? (
          <PredictionLab />
        ) : (
          <>
            <section className="welcome-row">
              <div>
                <h2>Network Security Overview</h2>
                <p>
                  Dynamic graph neural network-based IoT threat
                  detection and model intelligence.
                </p>
              </div>

              <button
                className="refresh-button"
                onClick={loadDashboard}
              >
                <RefreshCw size={16} />
                Refresh data
              </button>
            </section>

            <section className="metrics-grid">
              <MetricCard
                title="Processed flows"
                value={
                  loading ? "..." : totalFlows.toLocaleString()
                }
                description="Records in available training distribution"
                icon={Activity}
              />

              <MetricCard
                title="Attack categories"
                value={distribution.length || "—"}
                description="Categories found in training data"
                icon={ShieldAlert}
              />

              <MetricCard
                title="Evaluated models"
                value={comparison.length || "—"}
                description="Models in comparison results"
                icon={BrainCircuit}
              />

              <MetricCard
                title="Best model accuracy"
                value={
                  bestModel
                    ? `${(
                        Number(bestModel.accuracy) * 100
                      ).toFixed(2)}%`
                    : "—"
                }
                description={
                  bestModel?.model_name || "Evaluation unavailable"
                }
                icon={Shield}
              />
            </section>

            <section className="content-grid">
              <div className="panel distribution-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Traffic Class Distribution</h3>
                    <p>Training dataset · CICIoT2023</p>
                  </div>
                  <span className="panel-tag">DATASET</span>
                </div>

                {distribution.length > 0 ? (
                  <>
                    <div className="donut-wrap">
                      <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                          <Pie
                            data={distribution}
                            dataKey="value"
                            nameKey="name"
                            innerRadius={72}
                            outerRadius={100}
                            paddingAngle={3}
                          >
                            {distribution.map((item, index) => (
                              <Cell
                                key={item.name}
                                fill={
                                  COLORS[index % COLORS.length]
                                }
                              />
                            ))}
                          </Pie>

                          <Tooltip
                            contentStyle={{
                              background: "#111827",
                              border: "1px solid #334155",
                              borderRadius: 10,
                              color: "#fff",
                            }}
                          />
                        </PieChart>
                      </ResponsiveContainer>

                      <div className="donut-center">
                        <strong>
                          {totalFlows.toLocaleString()}
                        </strong>
                        <span>Total flows</span>
                      </div>
                    </div>

                    <div className="legend">
                      {distribution.map((item, index) => (
                        <div
                          className="legend-item"
                          key={item.name}
                        >
                          <span
                            className="legend-dot"
                            style={{
                              background:
                                COLORS[index % COLORS.length],
                            }}
                          />
                          <span>{item.name}</span>
                          <strong>
                            {Number(item.value).toLocaleString()}
                          </strong>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    {loading
                      ? "Loading dataset distribution..."
                      : "No distribution data available. Run preprocessing first."}
                  </div>
                )}
              </div>

              <div className="panel models-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Model Performance</h3>
                    <p>
                      Evaluation results from the existing pipeline
                    </p>
                  </div>
                  <span className="panel-tag">ML + GNN</span>
                </div>

                {comparison.length > 0 ? (
                  <div className="model-list">
                    {comparison.map((model) => (
                      <div
                        className="model-row"
                        key={model.model_name}
                      >
                        <div>
                          <strong>{model.model_name}</strong>
                          <small>Accuracy</small>
                        </div>

                        <div className="accuracy-value">
                          {(
                            Number(model.accuracy) * 100
                          ).toFixed(2)}%
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state">
                    {loading
                      ? "Loading evaluation results..."
                      : "No comparison results. Run the evaluation stage first."}
                  </div>
                )}
              </div>
            </section>

            <footer className="footer">
              <span>
                IoT Sentinel · Dynamic Graph Neural Networks
              </span>
              <span>
                CICIoT2023 · Research dashboard
              </span>
            </footer>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
