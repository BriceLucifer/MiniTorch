"""
Computational-graph visualiser for MiniTorch.

visualize_graph(output_var, filename="graph.html")

Structure
---------
Every node is pinned to an explicit column (level) derived from the
Variable/Function generation counter so the graph always looks like:

  col 0          col 1     col 2          col 3     col 4
  ┌─────────┐           ┌─────────┐           ┌─────────┐
  │ leaf    │──► (Add) ──► inter  │──► (Pow) ──► output  │
  │  Var    │     op        Var              │    Var    │
  └─────────┘           └─────────┘           └─────────┘
  ┌─────────┐  ╱
  │ leaf    │╱
  │  Var    │
  └─────────┘

  level = 2 * generation       for Variable nodes
  level = 2 * generation + 1   for Function nodes
"""
from __future__ import annotations

import textwrap

import numpy as np
from pyvis.network import Network


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
_BG   = "#f1f5f9"   # page background (slate-100)

_C = {
    "leaf_bg":  "#dbeafe", "leaf_bdr":  "#1d4ed8",   # blue — input / param
    "mid_bg":   "#ede9fe", "mid_bdr":   "#6d28d9",   # violet — intermediate
    "out_bg":   "#d1fae5", "out_bdr":   "#047857",   # emerald — output / loss
    "fn_bg":    "#fff7ed", "fn_bdr":    "#c2410c",   # orange — operation
    "fwd":      "#94a3b8",                            # slate-400 forward edge
    "grad":     "#f43f5e",                            # rose-500 gradient edge
    "txt":      "#0f172a",                            # slate-900
    "txt_fn":   "#7c2d12",                            # orange-950
}

_OP_HINTS: dict[str, str] = {
    "Add":                 "⊕ add",
    "Sub":                 "⊖ sub",
    "Mul":                 "⊗ mul",
    "Div":                 "÷ div",
    "Neg":                 "− neg",
    "Pow":                 "xⁿ pow",
    "Exp":                 "eˣ exp",
    "Log":                 "ln log",
    "Sin":                 "sin",
    "Cos":                 "cos",
    "Tanh":                "tanh",
    "ReLU":                "max(0,x)",
    "Sigmoid":             "σ(x)",
    "MatMul":              "A@B",
    "Reshape":             "reshape",
    "Transpose":           "ᵀ",
    "BroadcastTo":         "broadcast",
    "Sum":                 "Σ sum",
    "SumTo":               "Σ→",
    "Square":              "x²",
    "MeanSquaredError":    "MSE",
    "SoftmaxCrossEntropy": "CE loss",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vid(v) -> str:  return f"v{id(v)}"
def _fid(f) -> str:  return f"f{id(f)}"


def _preview(arr: np.ndarray, n: int = 3) -> str:
    if arr is None:
        return "—"
    flat = arr.flatten()
    snippet = ", ".join(f"{x:.3g}" for x in flat[:n])
    tail = " …" if flat.size > n else ""
    return f"[{snippet}{tail}]"


def _grad_info(v) -> tuple[str, bool]:
    """Return (label_string, has_grad)."""
    g = getattr(v, "grad", None)
    if g is None:
        return "∇ —", False
    data = g.data if hasattr(g, "data") else g
    norm = float(np.linalg.norm(data.flatten()))
    return f"∇ {norm:.3g}  {_preview(data)}", True


# ─────────────────────────────────────────────────────────────────────────────
# Graph traversal — collect all nodes and compute levels
# ─────────────────────────────────────────────────────────────────────────────

def _collect_graph(root):
    """
    BFS from root backwards.
    Returns:
        vars_  : {id(v): variable}
        funcs_ : {id(f): function}
    """
    vars_  = {}
    funcs_ = {}
    queue  = [root]
    seen_v = set()
    seen_f = set()

    while queue:
        v = queue.pop()
        if id(v) in seen_v:
            continue
        seen_v.add(id(v))
        vars_[id(v)] = v

        if v.creator is None:
            continue
        f = v.creator
        if id(f) not in seen_f:
            seen_f.add(id(f))
            funcs_[id(f)] = f
            for inp in f.inputs:
                queue.append(inp)

    return vars_, funcs_


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def visualize_graph(
    output_var,
    filename: str = "graph.html",
    *,
    height: str  = "860px",
    width:  str  = "100%",
    show_grad_edges: bool = True,
) -> None:
    """
    Render the computational graph as a structured, interactive HTML file.

    Every Variable sits at column  2 * generation.
    Every Function  sits at column  2 * generation + 1.
    This forces a strict alternating grid: var → op → var → op → …

    Parameters
    ----------
    output_var      : root Variable (loss or any terminal node)
    filename        : output HTML path
    height / width  : canvas dimensions
    show_grad_edges : colour edges red where gradients exist
    """
    vars_, funcs_ = _collect_graph(output_var)

    # ── Network ───────────────────────────────────────────────────────────
    net = Network(
        directed=True,
        bgcolor=_BG,
        font_color=_C["txt"],
        height=height,
        width=width,
    )

    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 220,
          "nodeSpacing": 140,
          "treeSpacing": 220,
          "blockShifting": false,
          "edgeMinimization": false,
          "parentCentralization": true
        }
      },
      "physics": { "enabled": false },
      "edges": {
        "smooth": {
          "type": "straightCross",
          "roundness": 0
        },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
        "color":  { "inherit": false },
        "width":  2,
        "selectionWidth": 3
      },
      "nodes": {
        "shape": "box",
        "margin": 10,
        "shadow": { "enabled": true, "size": 8, "x": 2, "y": 2,
                    "color": "rgba(0,0,0,0.10)" }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": { "enabled": true }
      }
    }
    """)

    output_id = id(output_var)

    # ── Variable nodes ─────────────────────────────────────────────────────
    for vid, v in vars_.items():
        is_out  = (vid == output_id)
        is_leaf = (v.creator is None)
        data    = getattr(v, "data", None)
        grad_str, has_g = _grad_info(v)

        # Colour
        if is_out:
            bg, bdr, role = _C["out_bg"], _C["out_bdr"], "OUTPUT"
        elif is_leaf:
            bg, bdr, role = _C["leaf_bg"], _C["leaf_bdr"], "INPUT"
        else:
            bg, bdr, role = _C["mid_bg"], _C["mid_bdr"], "TENSOR"

        # Level — strict column assignment
        level = 2 * v.generation

        # Node label (4 tightly packed lines)
        name     = v.name or "var"
        shape_s  = str(data.shape)     if data is not None else "?"
        dtype_s  = str(data.dtype)[:8] if data is not None else "?"
        data_s   = _preview(data)      if data is not None else "—"

        label = f"{name}\n{shape_s}  {dtype_s}\n{data_s}\n{grad_str}"

        # Hover tooltip
        tooltip = (
            f"<b>{name}</b> [{role}]<br>"
            f"shape: {shape_s} | dtype: {dtype_s}<br>"
            f"data: {data_s}<br>"
            f"<span style='color:{'#f43f5e' if has_g else '#64748b'}'>"
            f"{grad_str}</span>"
        )

        net.add_node(
            _vid(v),
            label=label,
            title=tooltip,
            level=level,
            shape="box",
            widthConstraint={"minimum": 130, "maximum": 200},
            borderWidth=3 if has_g else 2,
            color={
                "background": bg, "border": bdr,
                "highlight": {"background": bg, "border": _C["txt"]},
                "hover":     {"background": bg, "border": bdr},
            },
            font={
                "face":  "Menlo, Monaco, monospace",
                "size":  11,
                "color": _C["txt"],
            },
        )

    # ── Function nodes ─────────────────────────────────────────────────────
    for fid, f in funcs_.items():
        cls   = f.__class__.__name__
        hint  = _OP_HINTS.get(cls, "")
        level = 2 * f.generation + 1          # always between its inputs and output

        label   = f"{cls}\n{hint}" if hint else cls
        tooltip = (
            f"<b>{cls}</b><br>"
            f"{hint}<br>"
            f"generation: {f.generation}<br>"
            f"inputs: {len(f.inputs)}"
        )

        net.add_node(
            _fid(f),
            label=label,
            title=tooltip,
            level=level,
            shape="ellipse",
            widthConstraint={"minimum": 90, "maximum": 130},
            borderWidth=2,
            color={
                "background": _C["fn_bg"], "border": _C["fn_bdr"],
                "highlight": {"background": "#ffedd5", "border": "#9a3412"},
                "hover":     {"background": "#ffedd5", "border": _C["fn_bdr"]},
            },
            font={
                "face":  "system-ui, sans-serif",
                "size":  12,
                "bold":  True,
                "color": _C["txt_fn"],
            },
        )

    # ── Edges ──────────────────────────────────────────────────────────────
    for fid, f in funcs_.items():
        # Variable → Function  (input edges)
        for inp in f.inputs:
            has_g  = getattr(inp, "grad", None) is not None
            color  = _C["grad"] if (has_g and show_grad_edges) else _C["fwd"]
            net.add_edge(_vid(inp), _fid(f),
                         color={"color": color, "highlight": color},
                         width=2, arrows="to")

        # Function → Variable  (output edge)
        for out_ref in f.outputs:
            out = out_ref()
            if out is None or id(out) not in vars_:
                continue
            has_g  = getattr(out, "grad", None) is not None
            color  = _C["grad"] if (has_g and show_grad_edges) else _C["fwd"]
            net.add_edge(_fid(f), _vid(out),
                         color={"color": color, "highlight": color},
                         width=2, arrows="to")

    # ── Legend HTML ────────────────────────────────────────────────────────
    def _swatch(bg, bdr, shape="square"):
        r = "50%" if shape == "circle" else "3px"
        return (
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'border-radius:{r};background:{bg};border:2px solid {bdr};'
            f'vertical-align:middle;"></span>'
        )

    legend = textwrap.dedent(f"""
    <div style="
        position:fixed; bottom:20px; left:20px; z-index:9999;
        background:rgba(255,255,255,0.96);
        border:1px solid #cbd5e1; border-radius:12px;
        padding:14px 18px 12px; font-family:system-ui,sans-serif;
        font-size:12px; line-height:1.8;
        box-shadow:0 4px 20px rgba(0,0,0,0.12); min-width:210px;">
      <div style="font-weight:700;font-size:13px;margin-bottom:6px;
                  color:#0f172a;letter-spacing:0.01em;">
        MiniTorch — Computation Graph
      </div>
      {_swatch(_C['leaf_bg'],_C['leaf_bdr'])}
      <span style="color:#1e40af;margin-left:6px;">Input / Parameter</span><br>
      {_swatch(_C['mid_bg'],_C['mid_bdr'])}
      <span style="color:#5b21b6;margin-left:6px;">Intermediate Tensor</span><br>
      {_swatch(_C['out_bg'],_C['out_bdr'])}
      <span style="color:#065f46;margin-left:6px;">Output / Loss</span><br>
      {_swatch(_C['fn_bg'],_C['fn_bdr'],'circle')}
      <span style="color:{_C['txt_fn']};margin-left:6px;">Operation</span>
      <hr style="border:none;border-top:1px solid #e2e8f0;margin:8px 0 6px;">
      <span style="display:inline-block;width:24px;height:3px;
        background:{_C['fwd']};border-radius:2px;vertical-align:middle;"></span>
      <span style="color:#475569;margin-left:6px;">Forward pass</span><br>
      <span style="display:inline-block;width:24px;height:3px;
        background:{_C['grad']};border-radius:2px;vertical-align:middle;"></span>
      <span style="color:#be123c;margin-left:6px;">Gradient computed</span>
      <hr style="border:none;border-top:1px solid #e2e8f0;margin:8px 0 6px;">
      <div style="color:#64748b;font-size:10px;line-height:1.6;">
        Thick border → grad available<br>
        Columns: <b>var → op → var → op …</b><br>
        Hover nodes · scroll to zoom
      </div>
    </div>
    """)

    # ── Write file ─────────────────────────────────────────────────────────
    net.write_html(filename)

    try:
        with open(filename, "r", encoding="utf-8") as fh:
            html = fh.read()
        html = html.replace("</body>", legend + "\n</body>")
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(html)
    except Exception:
        pass

    print(f"[viz] Saved → {filename}")
