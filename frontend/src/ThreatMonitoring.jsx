
import { useEffect, useMemo, useState } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  AlertTriangle,
  Search,
  RefreshCw,
} from "lucide-react";
import "./ThreatMonitoring.css";

export default function ThreatMonitoring() {
  const [distribution, setDistribution] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");

  async function loadThreatData() {
    try {
      setLoading(true);
      setError("");

      const response = await fetch("/api/class-distribution");

      if (!response.ok) {
        throw new Error("Unable to retrieve traffic statistics.");
      }

      const data = await response.json();

      const formatted = Object.entries(data).map(([name, value]) => ({
        name,
        value: Number(value) || 0,
      }));

      setDistribution(formatted);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadThreatData();
  }, []);

  const benignFlows = distribution
    .filter((item) => item.name.toLowerCase().includes("benign"))
    .reduce((sum, item) => sum + item.value, 0);

  const totalFlows = distribution.reduce(
    (sum, item) => sum + item.value,
    0
  );

  const attackFlows = Math.max(0, totalFlows - benignFlows);

  const attackPercentage =
    totalFlows > 0 ? (attackFlows / totalFlows) * 100 : 0;

  const attackCategories = distribution.filter(
    (item) => !item.name.toLowerCase().includes("benign")
  );

  const filteredCategories = useMemo(() => {
    return distribution.filter((item) => {
      const matchesSearch = item.name
        .toLowerCase()
        .includes(search.toLowerCase());

      const isAttack = !item.name.toLowerCase().includes("benign");

      const matchesFilter =
        filter === "All" ||
        (filter === "Attacks" && isAttack) ||
        (filter === "Benign" && !isAttack);

      return matchesSearch && matchesFilter;
    });
  }, [distribution, search, filter]);

  const maximumValue = Math.max(
    1,
    ...distribution.map((item) => item.value)
  );

  function getPercentage(value) {
    return totalFlows > 0
      ? ((value / totalFlows) * 100).toFixed(2)
      : "0.00";
  }

  if (loading) {
    return (
      <div className="threat-message">
        Loading threat monitoring data...
      </div>
    );
  }

  return (
    <section className="threat-monitoring">

      <div className="threat-header">
        <div>
          <h2>Threat Monitoring Center</h2>
          <p>
            Dataset-based analysis of IoT network traffic
            and attack categories.
          </p>
        </div>

        <button
          className="threat-refresh"
          onClick={loadThreatData}
        >
          <RefreshCw size={16} />
          Refresh Data
        </button>
      </div>

      {error && (
        <div className="threat-error">
          {error}
        </div>
      )}

      <div className="monitoring-status">
        <span className="monitoring-dot" />
        Dataset Analysis Available
        <span className="monitoring-source">
          Source: CICIoT2023
        </span>
      </div>

      <div className="threat-metrics">

        <div className="threat-card">
          <div className="threat-card-top">
            <span>Total Traffic Flows</span>
            <Activity size={20} />
          </div>
          <h2>{totalFlows.toLocaleString()}</h2>
          <p>Analyzed dataset records</p>
        </div>

        <div className="threat-card">
          <div className="threat-card-top">
            <span>Attack Flows</span>
            <ShieldAlert size={20} />
          </div>
          <h2>{attackFlows.toLocaleString()}</h2>
          <p>Non-benign traffic records</p>
        </div>

        <div className="threat-card">
          <div className="threat-card-top">
            <span>Benign Flows</span>
            <ShieldCheck size={20} />
          </div>
          <h2>{benignFlows.toLocaleString()}</h2>
          <p>Records labelled benign</p>
        </div>

        <div className="threat-card">
          <div className="threat-card-top">
            <span>Attack Percentage</span>
            <AlertTriangle size={20} />
          </div>
          <h2>{attackPercentage.toFixed(2)}%</h2>
          <p>Share of non-benign records</p>
        </div>

      </div>

      <div className="threat-content">

        <div className="threat-panel">
          <div className="threat-panel-heading">
            <div>
              <h3>Attack Category Distribution</h3>
              <p>Number of records in each attack category</p>
            </div>
          </div>

          {attackCategories.length > 0 ? (
            <div className="attack-bars">
              {attackCategories
                .slice()
                .sort((a, b) => b.value - a.value)
                .map((item, index) => (
                  <div className="attack-bar-row" key={item.name}>
                    <div className="attack-bar-label">
                      <span>{item.name}</span>
                      <strong>
                        {item.value.toLocaleString()}
                      </strong>
                    </div>

                    <div className="attack-bar-track">
                      <div
                        className="attack-bar-fill"
                        style={{
                          width: `${(item.value / maximumValue) * 100}%`,
                          background: [
                            "#ef4444",
                            "#f97316",
                            "#eab308",
                            "#a855f7",
                            "#06b6d4",
                            "#ec4899",
                            "#64748b",
                          ][index % 7],
                        }}
                      />
                    </div>

                    <small>
                      {getPercentage(item.value)}% of total traffic
                    </small>
                  </div>
                ))}
            </div>
          ) : (
            <div className="threat-empty">
              No attack category data available.
            </div>
          )}
        </div>

        <div className="threat-panel summary-panel">
          <h3>Traffic Summary</h3>

          <div className="summary-item">
            <span>Benign traffic</span>
            <strong>{getPercentage(benignFlows)}%</strong>
          </div>

          <div className="summary-track">
            <div
              className="summary-benign"
              style={{
                width: `${getPercentage(benignFlows)}%`,
              }}
            />
          </div>

          <div className="summary-item">
            <span>Attack traffic</span>
            <strong>{getPercentage(attackFlows)}%</strong>
          </div>

          <div className="summary-track">
            <div
              className="summary-attack"
              style={{
                width: `${getPercentage(attackFlows)}%`,
              }}
            />
          </div>

          <div className="summary-divider" />

          <div className="summary-stat">
            <span>Attack categories</span>
            <strong>{attackCategories.length}</strong>
          </div>

          <div className="summary-stat">
            <span>Benign categories</span>
            <strong>{distribution.length - attackCategories.length}</strong>
          </div>

        </div>

      </div>

      <div className="threat-panel category-panel">

        <div className="threat-panel-heading">
          <div>
            <h3>Traffic Classification Records</h3>
            <p>Filter and inspect available dataset categories</p>
          </div>
        </div>

        <div className="threat-table-controls">

          <div className="threat-search">
            <Search size={17} />
            <input
              type="text"
              placeholder="Search traffic category..."
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>

          <select
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          >
            <option value="All">All Categories</option>
            <option value="Attacks">Attacks Only</option>
            <option value="Benign">Benign Only</option>
          </select>

        </div>

        <div className="threat-table-wrapper">
          <table className="threat-table">
            <thead>
              <tr>
                <th>Traffic Category</th>
                <th>Classification</th>
                <th>Flow Count</th>
                <th>Percentage</th>
              </tr>
            </thead>

            <tbody>
              {filteredCategories.map((item) => {
                const isBenign = item.name
                  .toLowerCase()
                  .includes("benign");

                return (
                  <tr key={item.name}>
                    <td>{item.name}</td>

                    <td>
                      <span
                        className={`threat-badge ${
                          isBenign ? "benign-badge" : "attack-badge"
                        }`}
                      >
                        {isBenign ? "Benign" : "Attack"}
                      </span>
                    </td>

                    <td>{item.value.toLocaleString()}</td>

                    <td>{getPercentage(item.value)}%</td>
                  </tr>
                );
              })}

              {filteredCategories.length === 0 && (
                <tr>
                  <td colSpan="4" className="threat-empty">
                    No matching categories found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

      </div>

      <div className="threat-disclaimer">
        <AlertTriangle size={16} />
        This page presents statistics from the available CICIoT2023
        dataset. It does not represent live network traffic or
        real-time threat detection.
      </div>

    </section>
  );
}
