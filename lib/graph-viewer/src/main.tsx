import React, { memo, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Controls,
  Handle,
  Node,
  NodeProps,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useViewport
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./styles.css";

type NeuronStats = {
  index: number;
  value: number | null;
  gradient: number | null;
};
type ArchitectureNode = {
  id: string;
  name: string;
  type: string;
  shape: Array<number | null>;
  neurons: number | null;
  inputFeatures?: number;
  outputFeatures?: number;
  neuronStats?: {
    input: NeuronStats[];
    output: NeuronStats[];
  };
};
type Architecture = {
  schemaVersion: number;
  model: {
    name: string;
    totalParameters: number;
    parameterTensors: number;
    inputShape: Array<number | null>;
    outputShape: Array<number | null>;
  };
  nodes: ArchitectureNode[];
  edges: Array<{ id: string; source: string; target: string }>;
};

type NeuronNodeData = {
  label: string;
  role: "input" | "output";
  index: number;
  dense: boolean;
  isFinal: boolean;
  column: number;
  layerId: string;
};

function shapeLabel(shape: Array<number | null>) {
  return shape.map((dimension) => dimension ?? "batch").join(" × ");
}

const NeuronNode = memo(({ data }: NodeProps<Node<NeuronNodeData>>) => (
  <div
    className={`neuron neuron--${data.role}${data.dense ? " neuron--dense" : ""}${data.isFinal ? " neuron--final" : ""}`}
    aria-label={`${data.role} neuron ${data.label}`}
    title={data.label}
  >
    <span>{data.label}</span>
    <Handle type="target" position={Position.Left} />
    <Handle type="source" position={Position.Right} />
  </div>
));

const nodeTypes = { neuron: NeuronNode };

function ArchitectureOverview({
  architecture,
  open,
  onToggle
}: {
  architecture: Architecture;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <section
      className={`architecture-section${open ? "" : " architecture-section--collapsed"}`}
      aria-labelledby="architecture-title"
    >
      <header className="architecture-section__header">
        <div>
          <h2 id="architecture-title">Architecture</h2>
          <span>{architecture.model.inputShape.at(-1) ?? "?"} inputs</span>
        </div>
        <button
          className="section-toggle"
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          aria-controls="architecture-track"
          title={open ? "Collapse architecture" : "Expand architecture"}
        >
          {open ? "⌃" : "⌄"}
        </button>
      </header>
      {open && (
        <div className="architecture-track" id="architecture-track">
          {architecture.nodes.map((node, index) => (
            <React.Fragment key={node.id}>
              {index > 0 && <span className="architecture-arrow" aria-hidden="true">→</span>}
              <article
                className={`architecture-card architecture-card--${node.type.toLowerCase()}`}
              >
                <span>{node.type}</span>
                <strong>{node.neurons ?? shapeLabel(node.shape)}</strong>
                <small>{node.name}</small>
              </article>
            </React.Fragment>
          ))}
        </div>
      )}
    </section>
  );
}

function modelNeuronGraph(architecture: Architecture) {
  const layers = architecture.nodes.filter(
    (node): node is ArchitectureNode & {
      inputFeatures: number;
      outputFeatures: number;
    } =>
      node.type === "Linear" &&
      node.inputFeatures !== undefined &&
      node.outputFeatures !== undefined
  );
  if (!layers.length) {
    return {
      nodes: [] as Node[],
      connections: 0,
      path: "No linear layers",
      width: 960,
      height: 680
    };
  }

  const counts = [layers[0].inputFeatures, ...layers.map((layer) => layer.outputFeatures)];
  const dense = Math.max(...counts) > 24;
  const nodeSize = dense ? 34 : 36;
  const gap = dense ? 46 : 48;
  const graphHeight = (Math.max(...counts) - 1) * gap;
  const columnGap = dense ? 680 : 360;
  const margin = dense ? 80 : 64;
  const nodes: Node[] = [];

  counts.forEach((count, column) => {
    const offset = (graphHeight - (count - 1) * gap) / 2;
    const layer = column === 0 ? layers[0] : layers[column - 1];
    for (let index = 0; index < count; index += 1) {
      nodes.push({
        id: `column-${column}-neuron-${index}`,
        type: "neuron",
        position: {
          x: margin + column * columnGap,
          y: margin + offset + index * gap
        },
        data: {
          label: `${
            column === 0
              ? "x"
              : column === counts.length - 1
                ? "y"
                : `h${column}.`
          }${index + 1}`,
          role: column === 0 ? "input" : "output",
          index,
          dense,
          isFinal: column === counts.length - 1,
          column,
          layerId: layer.id
        },
        width: nodeSize,
        height: nodeSize
      });
    }
  });

  const connections = layers.reduce(
    (total, layer) => total + layer.inputFeatures * layer.outputFeatures,
    0
  );
  return {
    nodes,
    connections,
    path: counts.map((count) => count.toLocaleString()).join(" → "),
    width: margin * 2 + (counts.length - 1) * columnGap + nodeSize,
    height: margin * 2 + graphHeight + nodeSize
  };
}

function MeshConnections({
  nodes,
  selected
}: {
  nodes: Node[];
  selected: NeuronNodeData | undefined;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const viewport = useViewport();

  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.round(entry.contentRect.width),
        height: Math.round(entry.contentRect.height)
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const element = canvas.current;
    if (!element || size.width === 0 || size.height === 0) return;
    const pixelRatio = window.devicePixelRatio || 1;
    element.width = Math.round(size.width * pixelRatio);
    element.height = Math.round(size.height * pixelRatio);
    const context = element.getContext("2d");
    if (!context) return;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, size.width, size.height);

    const columns = new Map<number, Node[]>();
    for (const node of nodes) {
      const data = node.data as NeuronNodeData;
      const column = columns.get(data.column) ?? [];
      column.push(node);
      columns.set(data.column, column);
    }
    const ordered = [...columns.entries()].sort(([left], [right]) => left - right);
    if (ordered.length < 2) return;

    const nodeSize = (ordered[0][1][0].data as NeuronNodeData).dense ? 34 : 36;
    const style = getComputedStyle(element);
    context.strokeStyle = style.getPropertyValue("--mesh-line").trim() || "#777";
    context.lineWidth = Math.max(0.7, Math.min(1.05, viewport.zoom * 1.15));
    const connectionCount = ordered.slice(0, -1).reduce(
      (total, [, column], index) => total + column.length * ordered[index + 1][1].length,
      0
    );
    context.globalAlpha = Math.max(
      0.025,
      Math.min(0.12, 10 / Math.sqrt(connectionCount))
    );
    context.beginPath();
    for (let columnIndex = 0; columnIndex < ordered.length - 1; columnIndex += 1) {
      const sources = ordered[columnIndex][1];
      const targets = ordered[columnIndex + 1][1];
      const sourceX =
        (sources[0].position.x + nodeSize) * viewport.zoom + viewport.x;
      const targetX = targets[0].position.x * viewport.zoom + viewport.x;
      for (const source of sources) {
        const sourceY =
          (source.position.y + nodeSize / 2) * viewport.zoom + viewport.y;
        for (const target of targets) {
          const targetY =
            (target.position.y + nodeSize / 2) * viewport.zoom + viewport.y;
          context.moveTo(sourceX, sourceY);
          context.lineTo(targetX, targetY);
        }
      }
    }
    context.stroke();

    if (selected) {
      const selectedNode = nodes.find((node) => {
        const data = node.data as NeuronNodeData;
        return data.column === selected.column && data.index === selected.index;
      });
      if (selectedNode) {
        const selectedSize = (selectedNode.data as NeuronNodeData).dense ? 34 : 36;
        const selectedXLeft =
          selectedNode.position.x * viewport.zoom + viewport.x;
        const selectedXRight =
          (selectedNode.position.x + selectedSize) * viewport.zoom + viewport.x;
        const selectedY =
          (selectedNode.position.y + selectedSize / 2) * viewport.zoom + viewport.y;
        const previous = columns.get(selected.column - 1) ?? [];
        const next = columns.get(selected.column + 1) ?? [];

        context.strokeStyle =
          style.getPropertyValue("--mesh-active").trim() || "#fff";
        context.globalAlpha = 0.72;
        context.lineWidth = Math.max(0.9, Math.min(1.25, viewport.zoom * 1.2));
        context.beginPath();
        for (const source of previous) {
          const sourceSize = (source.data as NeuronNodeData).dense ? 34 : 36;
          context.moveTo(
            (source.position.x + sourceSize) * viewport.zoom + viewport.x,
            (source.position.y + sourceSize / 2) * viewport.zoom + viewport.y
          );
          context.lineTo(selectedXLeft, selectedY);
        }
        for (const target of next) {
          const targetSize = (target.data as NeuronNodeData).dense ? 34 : 36;
          context.moveTo(selectedXRight, selectedY);
          context.lineTo(
            target.position.x * viewport.zoom + viewport.x,
            (target.position.y + targetSize / 2) * viewport.zoom + viewport.y
          );
        }
        context.stroke();
      }
    }
  }, [nodes, selected, size, viewport.x, viewport.y, viewport.zoom]);

  return <canvas ref={canvas} className="mesh-connections" aria-hidden="true" />;
}

function Inspector({
  node,
  neuron,
  open,
  onToggle
}: {
  node: ArchitectureNode | undefined;
  neuron: NeuronNodeData | undefined;
  open: boolean;
  onToggle: () => void;
}) {
  const toggle = (
    <button
      className="inspector-toggle"
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      title={open ? "Collapse neuron inspector" : "Expand neuron inspector"}
    >
      {open ? "›" : "‹"}
    </button>
  );
  if (!open) {
    return (
      <aside className="inspector inspector--collapsed">
        {toggle}
        <span>Neuron</span>
      </aside>
    );
  }
  if (!node || !neuron || !node.neuronStats) {
    return (
      <aside className="inspector inspector--empty">
        {toggle}
        Select a neuron to inspect it.
      </aside>
    );
  }
  const stats = node.neuronStats[neuron.role][neuron.index];
  return (
    <aside className="inspector">
      {toggle}
      <div className="inspector__eyebrow">{node.name}</div>
      <h2>{neuron.label}</h2>
      <dl className="inspector__facts inspector__facts--primary">
        <div>
          <dt>Value</dt>
          <dd>{stats.value == null ? "not captured" : stats.value.toExponential(6)}</dd>
        </div>
        <div>
          <dt>Grad</dt>
          <dd>
            {stats.gradient == null
              ? "not computed"
              : stats.gradient.toExponential(6)}
          </dd>
        </div>
      </dl>
    </aside>
  );
}

function Viewer({ architecture }: { architecture: Architecture }) {
  const graph = useMemo(() => modelNeuronGraph(architecture), [architecture]);
  const scrollContainer = useRef<HTMLElement>(null);
  const initialZoom = 0.55;
  const [architectureOpen, setArchitectureOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [selectedNeuron, setSelectedNeuron] = useState<NeuronNodeData>(() => {
    const firstHiddenColumn = graph.nodes.filter(
      (node) => (node.data as NeuronNodeData).column === 1
    );
    const middleHidden =
      firstHiddenColumn[Math.floor(firstHiddenColumn.length / 2)];
    return (middleHidden ?? graph.nodes[0])?.data as NeuronNodeData;
  });
  const selectedLayer = architecture.nodes.find(
    (node) => node.id === selectedNeuron?.layerId
  );

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      const element = scrollContainer.current;
      if (!element) return;
      element.scrollLeft = (element.scrollWidth - element.clientWidth) / 2;
      element.scrollTop = (element.scrollHeight - element.clientHeight) / 2;
    });
    return () => cancelAnimationFrame(frame);
  }, [graph]);

  return (
    <main className="viewer-shell">
      <header className="viewer-header">
        <div>
          <div className="brand">MiniTorch <span>model explorer</span></div>
          <h1>{architecture.model.name}</h1>
        </div>
        <div className="model-stats">
          <span><strong>{architecture.model.totalParameters.toLocaleString()}</strong> parameters</span>
          <span><strong>{architecture.nodes.length - 1}</strong> layers</span>
          <span><strong>{shapeLabel(architecture.model.outputShape)}</strong> output</span>
        </div>
      </header>
      <ArchitectureOverview
        architecture={architecture}
        open={architectureOpen}
        onToggle={() => setArchitectureOpen((value) => !value)}
      />
      <div
        className={`viewer-content${inspectorOpen ? "" : " viewer-content--inspector-closed"}`}
      >
        <section
          ref={scrollContainer}
          className="canvas-scroll"
          aria-label={`${architecture.model.name} full neuron map`}
        >
          <div
            className="canvas-stage"
            style={{ width: graph.width, height: graph.height }}
          >
            <ReactFlow
              nodes={graph.nodes}
              edges={[]}
              nodeTypes={nodeTypes}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable
              minZoom={0.2}
              maxZoom={2.5}
              defaultViewport={{
                x: graph.width * (1 - initialZoom) / 2,
                y: graph.height * (1 - initialZoom) / 2,
                zoom: initialZoom
              }}
              zoomOnScroll
              zoomOnPinch
              zoomOnDoubleClick
              panOnScroll={false}
              panOnDrag
              preventScrolling={false}
              onNodeClick={(_, node) => {
                setSelectedNeuron(node.data as NeuronNodeData);
              }}
            >
              <MeshConnections nodes={graph.nodes} selected={selectedNeuron} />
              <Controls
                position="bottom-left"
                showFitView={false}
                showInteractive={false}
              />
            </ReactFlow>
          </div>
        </section>
        <Inspector
          node={selectedLayer}
          neuron={selectedNeuron}
          open={inspectorOpen}
          onToggle={() => setInspectorOpen((value) => !value)}
        />
      </div>
    </main>
  );
}

const payload = document.getElementById("minitorch-model-data");
const root = document.getElementById("minitorch-model-viewer");
if (!payload || !root) {
  throw new Error("MiniTorch model viewer host elements are missing.");
}
const architecture = JSON.parse(payload.textContent ?? "{}") as Architecture;
createRoot(root).render(
  <React.StrictMode>
    <ReactFlowProvider>
      <Viewer architecture={architecture} />
    </ReactFlowProvider>
  </React.StrictMode>
);
