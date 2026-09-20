from pathlib import Path
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go


def make_labels(nodes, prefix=None):
    if prefix:
        return {node: f"{prefix}{i + 1}" for i, node in enumerate(nodes)}

    labels = {}
    for node in nodes:
        text = str(node).strip()
        try:
            labels[node] = f"{text[0].upper()}{int(text[1:])}"
        except (ValueError, IndexError):
            labels[node] = text
    return labels


def create_layout(graph, seed, k, iterations):
    return nx.spring_layout(
        graph,
        seed=seed,
        k=k,
        iterations=iterations,
        weight="weight",
    )


def draw_static_network(
    graph,
    output_path,
    title,
    labels,
    figure_size,
    seed,
    k,
    iterations,
    domain_styles=None,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figure_size)
    pos = create_layout(graph, seed, k, iterations)

    degrees = dict(graph.degree())
    node_sizes = {node: 800 + 100 * degrees[node] for node in graph.nodes()}

    positive, negative = [], []
    for u, v, data in graph.edges(data=True):
        (positive if data["correlation"] >= 0 else negative).append((u, v))

    def widths(edges):
        return [0.7 + 4.5 * abs(graph[u][v]["correlation"]) for u, v in edges]

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=positive,
        width=widths(positive), edge_color="gray", alpha=0.35,
    )
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=negative,
        width=widths(negative), edge_color="black",
        style="dashed", alpha=0.50,
    )

    if domain_styles:
        for domain, style in domain_styles.items():
            nodes = [
                node for node in graph.nodes()
                if graph.nodes[node].get("domain") == domain
            ]
            if nodes:
                nx.draw_networkx_nodes(
                    graph, pos, ax=ax, nodelist=nodes,
                    node_color=style.get("color", "white"),
                    node_shape=style.get("shape", "o"),
                    node_size=[node_sizes[n] for n in nodes],
                    alpha=0.90, edgecolors="black", linewidths=1.5,
                )
    else:
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_color="white",
            node_size=[node_sizes[n] for n in graph.nodes()],
            alpha=0.90, edgecolors="black", linewidths=1.5,
        )

    nx.draw_networkx_labels(
        graph, pos, ax=ax, labels=labels,
        font_size=10, font_weight="bold", font_color="black",
    )

    ax.set_title(title, fontsize=22, fontweight="bold", pad=25)

    legend = [
        Line2D([0], [0], color="gray", linewidth=3, label="Positive correlation"),
        Line2D([0], [0], color="black", linewidth=3,
               linestyle="--", label="Negative correlation"),
    ]

    if domain_styles:
        for domain, style in domain_styles.items():
            legend.append(
                Line2D(
                    [0], [0],
                    marker=style.get("shape", "o"),
                    color="white",
                    markerfacecolor=style.get("color", "white"),
                    markeredgecolor="black",
                    markersize=9,
                    label=domain,
                )
            )

    ax.legend(
        handles=legend,
        loc="upper left",
        fontsize=11,
        title="Network legend",
        title_fontsize=12,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_results(results, title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, column, label in zip(
        axes.flat,
        ["density", "average_degree", "giant_fraction", "clustering"],
        ["Density", "Average degree", "Giant component fraction", "Average clustering"],
    ):
        ax.plot(results["threshold"], results[column], marker="o")
        ax.set_xlabel("|Spearman ρ| threshold")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.show()
