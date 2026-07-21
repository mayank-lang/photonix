"""Draw a netlist as a connectivity graph."""
from __future__ import annotations

__all__ = ["plot_netlist"]


def plot_netlist(netlist, *, ax=None):
    """Draw a :class:`~photonix.circuit.Netlist` as a node-link graph.

    Uses ``networkx`` spring layout if available, otherwise a simple circular
    placement. Instances are nodes; connections and exposed ports are edges.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import math

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    insts = list(netlist.instances)
    edges = []
    for (ia, _pa), (ib, _pb) in netlist.connections.items():
        edges.append((ia, ib))

    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(insts)
        g.add_edges_from(edges)
        for ext, (inst, _p) in netlist.ports.items():
            g.add_node(ext)
            g.add_edge(ext, inst)
        pos = nx.spring_layout(g, seed=0)
        nx.draw_networkx(g, pos, ax=ax, node_color="#bcd", font_size=8, node_size=900)
    except Exception:  # noqa: BLE001 - fallback without networkx
        nodes = insts + list(netlist.ports)
        n = max(len(nodes), 1)
        pos = {nm: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i, nm in enumerate(nodes)}
        for a, b in edges:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], "k-", alpha=0.5)
        for nm, (x, y) in pos.items():
            ax.plot(x, y, "o", markersize=18, color="#bcd")
            ax.annotate(nm, (x, y), ha="center", va="center", fontsize=8)
        ax.set_aspect("equal")
    ax.set_title(netlist.name or "circuit")
    ax.axis("off")
    return ax
