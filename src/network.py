import numpy as np
import networkx as nx
import pandas as pd


def extract_edges(matrix: pd.DataFrame) -> list[dict]:
    nodes = matrix.index.tolist()
    edges = []

    for i, source in enumerate(nodes):
        for target in nodes[i + 1:]:
            rho = matrix.loc[source, target]
            if pd.isna(rho):
                continue

            rho = float(rho)
            edges.append({
                "source": source,
                "target": target,
                "correlation": rho,
                "weight": abs(rho),
            })

    return edges


def build_threshold_graph(nodes, edges: list[dict], threshold: float) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    for edge in edges:
        if edge["weight"] >= threshold:
            graph.add_edge(
                edge["source"],
                edge["target"],
                correlation=edge["correlation"],
                weight=edge["weight"],
            )

    return graph


def network_statistics(graph: nx.Graph) -> dict:
    n = graph.number_of_nodes()
    e = graph.number_of_edges()

    density = nx.density(graph) if n > 1 else 0.0
    average_degree = 2 * e / n if n else 0.0

    components = list(nx.connected_components(graph))
    largest = max(components, key=len) if components else set()
    largest_size = len(largest)

    if largest_size >= 2:
        largest_graph = graph.subgraph(largest)
        average_path_length = nx.average_shortest_path_length(largest_graph)
        diameter = nx.diameter(largest_graph)
    else:
        average_path_length = np.nan
        diameter = np.nan

    return {
        "nodes": n,
        "edges": e,
        "density": density,
        "average_degree": average_degree,
        "components": len(components),
        "largest_component": largest_size,
        "giant_fraction": largest_size / n if n else 0.0,
        "clustering": nx.average_clustering(graph) if n > 1 else 0.0,
        "average_path_length": average_path_length,
        "diameter": diameter,
    }


def er_statistics(n: int, density: float, simulations: int, seed: int) -> dict:
    if n == 0:
        return {key: np.nan for key in (
            "er_clustering_mean", "er_clustering_std",
            "er_path_length_mean", "er_path_length_std",
            "er_giant_fraction_mean", "er_giant_fraction_std",
        )}

    if density <= 0:
        return {
            "er_clustering_mean": 0.0,
            "er_clustering_std": 0.0,
            "er_path_length_mean": np.nan,
            "er_path_length_std": np.nan,
            "er_giant_fraction_mean": 1 / n,
            "er_giant_fraction_std": 0.0,
        }

    rng = np.random.default_rng(seed)
    clustering, path_lengths, giant_fraction = [], [], []

    for _ in range(simulations):
        random_seed = int(rng.integers(0, 1_000_000_000))
        graph = nx.gnp_random_graph(n, density, seed=random_seed)

        clustering.append(nx.average_clustering(graph))

        components = list(nx.connected_components(graph))
        largest = max(components, key=len)
        giant_fraction.append(len(largest) / n)

        if len(largest) >= 2:
            path_lengths.append(
                nx.average_shortest_path_length(graph.subgraph(largest))
            )

    return {
        "er_clustering_mean": np.mean(clustering),
        "er_clustering_std": np.std(clustering),
        "er_path_length_mean": np.mean(path_lengths) if path_lengths else np.nan,
        "er_path_length_std": np.std(path_lengths) if path_lengths else np.nan,
        "er_giant_fraction_mean": np.mean(giant_fraction),
        "er_giant_fraction_std": np.std(giant_fraction),
    }


def make_thresholds(minimum: float, maximum: float, step: float):
    count = round((maximum - minimum) / step)
    return np.round(np.linspace(minimum, maximum, count + 1), 2)


def threshold_sweep(matrix, thresholds, er_simulations, random_seed):
    edges = extract_edges(matrix)
    results = []

    for threshold in thresholds:
        graph = build_threshold_graph(matrix.index, edges, threshold)
        observed = network_statistics(graph)
        expected = er_statistics(
            n=graph.number_of_nodes(),
            density=observed["density"],
            simulations=er_simulations,
            seed=random_seed,
        )
        results.append({"threshold": threshold, **observed, **expected})

    return pd.DataFrame(results)


def domain_edge_counts(edges, threshold, domain_of):
    counts = {}

    for edge in edges:
        if edge["weight"] < threshold:
            continue

        pair = tuple(sorted((
            domain_of(edge["source"]),
            domain_of(edge["target"]),
        )))
        counts[pair] = counts.get(pair, 0) + 1

    return pd.DataFrame([
        {
            "domain_1": pair[0],
            "domain_2": pair[1],
            "edges": count,
            "threshold": threshold,
        }
        for pair, count in counts.items()
    ])
