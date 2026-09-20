import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import load_correlation_matrix
from src.network import (
    build_threshold_graph,
    domain_edge_counts,
    extract_edges,
    make_thresholds,
    threshold_sweep,
)
from src.plotting import (
    draw_static_network,
    make_labels,
)


def question_domain_lookup(config):
    domains = config["network"].get("domains", {})

    def get_domain(question):
        prefix = str(question).strip()[:1].upper()
        return domains.get(prefix, {}).get("name", "Unknown")

    return get_domain


def question_domain_styles(config):
    return {
        item["name"]: item
        for item in config["network"].get("domains", {}).values()
    }


def analyze(entity: str):
    config = load_config(f"{entity}.yaml")
    network = config["network"]

    matrix = load_correlation_matrix(network["matrix_input"])
    nodes = matrix.index.tolist()
    edges = extract_edges(matrix)

    print("=" * 60)
    print(f"{entity.upper()} NETWORK ANALYSIS")
    print("=" * 60)
    print(f"Nodes: {len(nodes)}")
    print(f"Possible edges: {len(nodes) * (len(nodes) - 1) // 2}")
    print(f"Non-missing pairs: {len(edges)}")

    thresholds = make_thresholds(
        network["threshold_min"],
        network["threshold_max"],
        network["threshold_step"],
    )

    results = threshold_sweep(
        matrix,
        thresholds,
        network["er_simulations"],
        network["random_seed"],
    )

    Path(network["threshold_output"]).parent.mkdir(
        parents=True, exist_ok=True
    )
    results.to_csv(network["threshold_output"], index=False)

    labels = make_labels(
        nodes,
        network["display"].get("label_prefix"),
    )

    is_questions = entity == "questions"
    get_domain = question_domain_lookup(config) if is_questions else None
    styles = question_domain_styles(config) if is_questions else None

    for threshold in network["static_thresholds"]:
        graph = build_threshold_graph(nodes, edges, threshold)

        if get_domain:
            for node in graph.nodes:
                graph.nodes[node]["domain"] = get_domain(node)

        output = (
            Path(network["static_output_dir"])
            / f"{entity.capitalize()}_Network_rho_{threshold:.2f}.png"
        )

        draw_static_network(
            graph,
            str(output),
            (
                f"{entity.capitalize()} Correlation Network\n"
                f"|Spearman ρ| ≥ {threshold:.2f} • "
                f"{graph.number_of_nodes()} nodes • "
                f"{graph.number_of_edges()} edges"
            ),
            labels,
            tuple(network["display"]["figure_size"]),
            network["random_seed"],
            network["display"]["layout_k"],
            network["display"]["layout_iterations"],
            styles,
        )

        print(f"Static network: {output}")



    if is_questions:
        domain_results = domain_edge_counts(
            edges,
            network["domain_analysis_threshold"],
            get_domain,
        )
        domain_results.to_csv(network["domain_output"], index=False)

    print(f"Threshold analysis: {network['threshold_output']}")
    if is_questions:
        print(f"Domain analysis: {network['domain_output']}")
    print("Analysis complete.")


if __name__ == "__main__":
    entity = sys.argv[1] if len(sys.argv) > 1 else "people"
    if entity not in {"people", "questions"}:
        raise SystemExit(
            "Usage: python scripts/analyze_network.py [people|questions]"
        )
    analyze(entity)
