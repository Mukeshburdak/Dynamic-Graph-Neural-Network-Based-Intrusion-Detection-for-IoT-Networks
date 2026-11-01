
import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import "./GraphExplorer.css";

const COLORS = [
  "#22c55e", // Benign
  "#ef4444",
  "#f97316",
  "#eab308",
  "#a855f7",
  "#06b6d4",
  "#ec4899",
  "#64748b",
];

export default function GraphExplorer() {
  const containerRef = useRef(null);

  const [dimensions, setDimensions] = useState({
    width: 800,
    height: 540,
  });

  const [snapshots, setSnapshots] = useState([]);
  const [selectedSplit, setSelectedSplit] = useState("train");
  const [selectedIndex, setSelectedIndex] = useState(0);

  const [graph, setGraph] = useState({
    nodes: [],
    links: [],
  });

  const [selectedNode, setSelectedNode] = useState(null);
  const [loadingSnapshots, setLoadingSnapshots] = useState(true);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [error, setError] = useState("");

  // Measure graph container for responsive rendering
  useEffect(() => {
    const container = containerRef.current;

    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;

      setDimensions({
        width: Math.max(300, Math.floor(width)),
        height: Math.max(300, Math.floor(height)),
      });
    });

    observer.observe(container);

    return () => observer.disconnect();
  }, []);

  // Load available graph snapshots
  useEffect(() => {
    let cancelled = false;

    async function loadSnapshots() {
      try {
        setError("");

        const response = await fetch("/api/graphs");

        if (!response.ok) {
          throw new Error(
            `Could not load snapshots. Server returned ${response.status}`
          );
        }

        const data = await response.json();
        const items = data.snapshots || [];

        if (cancelled) return;

        setSnapshots(items);

        if (items.length > 0) {
          const first =
            items.find((item) => item.split === "train") || items[0];

          setSelectedSplit(first.split);
          setSelectedIndex(first.index);
        } else {
          setError("No graph snapshots were found.");
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingSnapshots(false);
        }
      }
    }

    loadSnapshots();

    return () => {
      cancelled = true;
    };
  }, []);

  // Get snapshots belonging to the selected split
  const availableSnapshots = useMemo(() => {
    return snapshots.filter(
      (item) => item.split === selectedSplit
    );
  }, [snapshots, selectedSplit]);

  // Get splits that actually exist
  const availableSplits = useMemo(() => {
    return [...new Set(snapshots.map((item) => item.split))];
  }, [snapshots]);

  // Ensure selected snapshot exists after changing split
  useEffect(() => {
    if (!availableSnapshots.length) {
      setGraph({
        nodes: [],
        links: [],
      });

      return;
    }

    const exists = availableSnapshots.some(
      (item) => item.index === selectedIndex
    );

    if (!exists) {
      setSelectedIndex(availableSnapshots[0].index);
    }
  }, [availableSnapshots, selectedIndex]);

  // Load selected graph data
  useEffect(() => {
    if (!availableSnapshots.length) return;

    let cancelled = false;

    async function loadGraph() {
      try {
        setLoadingGraph(true);
        setError("");
        setSelectedNode(null);

        const response = await fetch(
          `/api/graphs/${selectedSplit}/${selectedIndex}`
        );

        if (!response.ok) {
          throw new Error(
            `Graph request failed with status ${response.status}`
          );
        }

        const data = await response.json();

        console.log("Graph API response:", data);
        console.log("Number of nodes:", data.nodes?.length);
        console.log("Number of edges:", data.edges?.length);

        const nodes = (data.nodes || []).map((node) => ({
          ...node,
          id: node.id,
        }));

        const nodeIds = new Set(nodes.map((node) => String(node.id)));

        const links = (data.edges || [])
          .filter(
            (edge) =>
              nodeIds.has(String(edge.source)) &&
              nodeIds.has(String(edge.target))
          )
          .map((edge) => ({
            source: edge.source,
            target: edge.target,
            weight: Number(edge.weight ?? 1),
          }));

        if (!cancelled) {
          setGraph({
            nodes,
            links,
          });

          if (nodes.length === 0) {
            setError("The selected graph contains no nodes.");
          }
        }
      } catch (err) {
        if (!cancelled) {
          setGraph({
            nodes: [],
            links: [],
          });

          setError(err.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingGraph(false);
        }
      }
    }

    loadGraph();

    return () => {
      cancelled = true;
    };
  }, [selectedSplit, selectedIndex, availableSnapshots.length]);

  // Loading screen
  if (loadingSnapshots) {
    return (
      <div className="graph-message">
        Loading graph snapshots...
      </div>
    );
  }

  // Select graph split
  function handleSplitChange(event) {
    const split = event.target.value;

    setSelectedSplit(split);
    setSelectedNode(null);

    const first = snapshots.find(
      (item) => item.split === split
    );

    if (first) {
      setSelectedIndex(first.index);
    }
  }

  return (
    <section className="graph-explorer">

      {/* Header */}
      <div className="graph-heading">
        <div>
          <h2>Graph Explorer</h2>
          <p>
            Explore dynamic IoT traffic-flow similarity graphs.
          </p>
        </div>

        <div className="graph-controls">

          <select
            value={selectedSplit}
            onChange={handleSplitChange}
          >
            {availableSplits.map((split) => (
              <option key={split} value={split}>
                {split.charAt(0).toUpperCase() + split.slice(1)}
              </option>
            ))}
          </select>

          <select
            value={selectedIndex}
            onChange={(event) =>
              setSelectedIndex(Number(event.target.value))
            }
            disabled={!availableSnapshots.length}
          >
            {availableSnapshots.map((item) => (
              <option
                key={item.filename}
                value={item.index}
              >
                {item.name}
              </option>
            ))}
          </select>

        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="graph-error">
          {error}
        </div>
      )}

      {/* Graph statistics */}
      <div className="graph-stats">

        <div>
          <span>Nodes</span>
          <strong>{graph.nodes.length}</strong>
        </div>

        <div>
          <span>Edges</span>
          <strong>{graph.links.length}</strong>
        </div>

        <div>
          <span>Dataset Split</span>
          <strong>{selectedSplit}</strong>
        </div>

      </div>

      {/* Graph and node inspector */}
      <div className="graph-layout">

        <div
          className="graph-canvas"
          ref={containerRef}
        >

          {loadingGraph ? (
            <div className="graph-message">
              Loading graph data...
            </div>
          ) : graph.nodes.length > 0 ? (

            <ForceGraph2D
              width={dimensions.width}
              height={dimensions.height}
              graphData={graph}
              backgroundColor="#080d18"

              nodeRelSize={5}
              nodeVal={() => 2}

              nodeColor={(node) => {
                const classId = Number(node.class_id);

                return Number.isInteger(classId)
                  ? COLORS[classId % COLORS.length]
                  : "#94a3b8";
              }}

              nodeLabel={(node) =>
                `Node ${node.id} | ${
                  node.class_name || "Unknown"
                }`
              }

              linkColor={() =>
                "rgba(100, 116, 139, 0.45)"
              }

              linkWidth={(link) =>
                Math.max(
                  0.5,
                  Math.min(2, Number(link.weight || 1))
                )
              }

              onNodeClick={(node) => {
                setSelectedNode(node);
              }}

              cooldownTicks={100}
              enableNodeDrag
              enableZoomInteraction
              enablePanInteraction
            />

          ) : (
            <div className="graph-message">
              {error || "No graph data available."}
            </div>
          )}

        </div>

        {/* Node inspector */}
        <aside className="node-inspector">

          <h3>Node Inspector</h3>

          {selectedNode ? (
            <>

              <p>
                <span>Node ID</span>
                <strong>{selectedNode.id}</strong>
              </p>

              <p>
                <span>Traffic Class</span>
                <strong>
                  {selectedNode.class_name || "Unknown"}
                </strong>
              </p>

              <p>
                <span>Class ID</span>
                <strong>{selectedNode.class_id}</strong>
              </p>

              <h4>Feature Values</h4>

              <div className="feature-list">

                {(selectedNode.features || []).map(
                  (value, index) => (
                    <div key={index}>
                      <span>Feature {index + 1}</span>

                      <strong>
                        {typeof value === "number"
                          ? value.toFixed(4)
                          : value}
                      </strong>
                    </div>
                  )
                )}

              </div>

            </>
          ) : (

            <p className="inspector-hint">
              Click any node in the graph to inspect its
              traffic class and feature values.
            </p>

          )}

        </aside>

      </div>

      {/* Class legend */}
      <div className="graph-legend">

        <h3>Traffic Class Legend</h3>

        <div className="legend-items">

          {[
            "Benign",
            "Brute Force",
            "DDoS",
            "DoS",
            "Mirai",
            "Spoofing",
            "Recon",
            "Web-based",
          ].map((name, index) => (

            <div className="legend-item" key={name}>

              <span
                className="legend-dot"
                style={{
                  background: COLORS[index],
                }}
              />

              <span>{name}</span>

            </div>

          ))}

        </div>

      </div>

      {/* Information */}
      <div className="graph-note">
        Each node represents an IoT traffic flow. Edges
        represent k-NN feature similarity, not verified
        physical or IP network connections.
      </div>

    </section>
  );
}
