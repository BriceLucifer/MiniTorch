from __future__ import annotations

import numpy as np
from pyvis.network import Network

# =========================
# ID helpers
# =========================


def _var_id(v) -> str:
    return f"var_{id(v)}"


def _func_id(f) -> str:
    return f"func_{id(f)}"


def _fmt(x, max_len: int = 6) -> str:
    if x is None:
        return "None"
    if isinstance(x, np.ndarray):
        flat = x.flatten()
        if flat.size > max_len:
            return f"{flat[:max_len]}..."
        return str(flat)
    return str(x)


# =========================
# Static computational graph (industrial style)
# =========================


def visualize_graph(
    output_var,
    filename: str = "graph.html",
    *,
    notebook: bool = True,
    height: str = "800px",
    width: str = "100%",
) -> None:
    """
    Render a clean, industrial-style static computational graph.

    - Input on the left, output on the right
    - Straight edges
    - No physics / no animation
    - Variables show data + grad
    """

    net = Network(
        directed=True,
        notebook=notebook,
        bgcolor="#ffffff",
        font_color="#111827",  # slate-900 # type: ignore
        height=height,
        width=width,
    )

    # ---------- Layout & style (KEY PART) ----------
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 180,
          "nodeSpacing": 260
        }
      },
      "physics": {
        "enabled": false
      },
      "edges": {
        "smooth": {
          "type": "horizontal",
          "roundness": 0
        },
        "arrows": {
          "to": { "enabled": true }
        },
        "color": {
          "color": "#9ca3af",
          "highlight": "#6b7280"
        },
        "width": 1.5
      }
    }
    """)

    visited_vars = set()
    visited_funcs = set()

    # ---------- Variable node ----------
    def add_variable(v):
        vid = _var_id(v)
        if vid in visited_vars:
            return
        visited_vars.add(vid)

        data = getattr(v, "data", None)

        dim = data.ndim if data is not None else None
        dtype = data.dtype if data is not None else None

        label = (
            f"{v.name or 'Variable'}\n"
            f"dim={dim}\n"
            f"dtype={dtype}\n"
            f"data={_fmt(data)}\n"
            f"grad={_fmt(getattr(v, 'grad', None))}"
        )

        net.add_node(
            vid,
            label=label,
            shape="box",
            borderWidth=1.5,
            borderWidthSelected=2,
            color={  # type: ignore
                "background": "#eef2ff",  # indigo-50
                "border": "#6366f1",  # indigo-500
                "highlight": {"background": "#e0e7ff", "border": "#4f46e5"},
            },
            font={
                "face": "Menlo, Monaco, Consolas, monospace",
                "size": 13,
                "color": "#111827",
            },
        )

        if v.creator is not None:
            add_function(v.creator)
            net.add_edge(_func_id(v.creator), vid)

    # ---------- Function node ----------
    def add_function(f):
        fid = _func_id(f)
        if fid in visited_funcs:
            return
        visited_funcs.add(fid)

        net.add_node(
            fid,
            label=f.__class__.__name__,
            shape="box",
            borderWidth=1.5,
            color={  # type: ignore
                "background": "#fff7ed",  # orange-50
                "border": "#f97316",  # orange-500
                "highlight": {"background": "#ffedd5", "border": "#ea580c"},
            },
            font={
                "face": "Inter, system-ui, sans-serif",
                "size": 14,
                "color": "#7c2d12",
            },
        )

        for x in f.inputs:
            add_variable(x)
            net.add_edge(_var_id(x), fid)

    # ---------- Build graph ----------
    add_variable(output_var)

    # ---------- Render ----------
    net.show(filename)
