import pickle
import argparse
import os
import pathlib
import utils
from evaluation.topology_eval_utils import (
    load_bipartite_graphs,
    valid_graph_statistics,
    is_valid_bipartite_graph
)
from evaluation.isomorphism_check import check_uniqueness, check_novelty

from optimization.BiDiffCkt_DiffCkt.netlist_generator import MulticlassBipartite2Netlist
from optimization.BiDiffCkt_DiffCkt_multilabel.netlist_generator import Bipartite2Netlist as MultilabelBipartite2Netlist
from optimization.BiDiffCkt_OCB.netlist_generator import valid_bidiffckt_OCB, OCB_bipartite_graph_to_netlist

def BiDiffCkt_multiclass_topo_eval(dataset_path):
    """Evaluate the topologies in the given dataset."""
    # Load generated data
    generated_data = load_bipartite_graphs(dataset_path)

    # Validity Check
    connectivity = valid_graph_statistics(generated_data)['valid_rate']
    # print(
    #     f"valid rate: {valid_rate:.2%}"
    # )
    success_count = 0
    for i, g in enumerate(generated_data):
        try:
            if is_valid_bipartite_graph(g):
                netlist = MulticlassBipartite2Netlist(g, i, output_dir=None)
                if netlist:
                    success_count += 1
        except ValueError as e:
            pass
    validity = success_count / len(generated_data)
    return connectivity, validity

def BiDiffCkt_multilabel_topo_eval(dataset_path):
    """Evaluate the topologies in the given dataset."""
    # Load generated data
    generated_data = load_bipartite_graphs(dataset_path)

    # Validity Check
    connectivity = valid_graph_statistics(generated_data)['valid_rate']
    # print(
    #     f"valid rate: {valid_rate:.2%}"
    # )
    success_count = 0
    for i, g in enumerate(generated_data):
        try:
            if is_valid_bipartite_graph(g):
                netlist = MultilabelBipartite2Netlist(g, i, output_dir=None)
                if netlist:
                    success_count += 1
        except ValueError as e:
            pass
    validity = success_count / len(generated_data)
    # print(f"Simulatability: {success_count / len(generated_data):.2%}")
    return connectivity, validity

def BiDiffCkt_OCB_topo_eval(dataset_path):
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[0].parents[0]
    train_data_path = root_dir / "datasets" / "OCB" / "OCB_in_Bipartite.pkl"
    if not os.path.exists(train_data_path):
        utils.dataset_download("OCB_in_Bipartite.pkl", root_dir / "datasets" / "OCB")
    
    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)['G']

    with open(dataset_path, "rb") as f:
        generated_data = pickle.load(f)['G']

    valid_graph_count = 0
    for i, g in enumerate(generated_data):
        if valid_bidiffckt_OCB(g):
            valid_graph_count += 1
    # print(f"Valid rate: {valid_graph_count / len(generated_data):.2%}")
    connectivity = valid_graph_count / len(generated_data)

    success_count = 0
    simulatable_g = []
    for i, g in enumerate(generated_data):
        if valid_bidiffckt_OCB(g):
            # netlists init
            try:
                OCB_bipartite_graph_to_netlist(g, i, output_dir=None)
                success_count +=1
                simulatable_g.append(g)
            except ValueError as e:
                print(f"Error in graph {i}: {e}")
    validity = success_count / valid_graph_count
    # print(f"Simulatability: {success_count / valid_graph_count:.2%}")

    uniqueness = check_uniqueness(simulatable_g)
    novelty = check_novelty(simulatable_g, train_data)

    return connectivity, validity, uniqueness, novelty

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Evaluate the topologies in the given generated results.")    
    argparse.add_argument("--path", type=str, required=False, help="Path to the dataset containing generated topologies.")
    argparse.add_argument("--exp_type", type=str, required=True, choices=["BiDiffCkt", "BiDiffCkt-multilabel", "BiDiffCkt-Digress", "BiDiffCkt-OCB"], default="BiDiffCkt")
    args = argparse.parse_args()

    if args.exp_type == "BiDiffCkt":
        if args.path is None:
            path = "evaluation/generated_results/BiDiffCkt_DiffCkt_generated_1000_results.pkl"
        else:
            path = args.path
        connectivity, validity = BiDiffCkt_multiclass_topo_eval(path)
        print("Dataset: DiffCkt")
        print("BiDiffCkt Topology Evaluation Results:")
        print(f"connectivity: {connectivity:.2%}. validity: {validity:.2%}")
    elif args.exp_type == "BiDiffCkt-multilabel":
        if args.path is None:
            path = "evaluation/generated_results/BiDiffCkt_DiffCkt_multilabel_generated_1000_results.pkl"
        else:
            path = args.path
        connectivity, validity = BiDiffCkt_multilabel_topo_eval(path)
        print("Dataset: DiffCkt")
        print("[Ablation Study] BiDiffCkt Multilabel Topology Evaluation Results:")
        print(f"connectivity: {connectivity:.2%}. validity: {validity:.2%}")
    elif args.exp_type == "BiDiffCkt-Digress":
        if args.path is None:
            path = "evaluation/generated_results/BiDiffCkt_DiffCkt_Digress_generated_1000_results.pkl"
        else:
            path = args.path
        connectivity, validity = BiDiffCkt_multiclass_topo_eval(path)
        print("Dataset: DiffCkt")
        print("[Ablation Study] BiDiffCkt Digress Topology Evaluation Results:")
        print(f"connectivity: {connectivity:.2%}. validity: {validity:.2%}")
    elif args.exp_type == "BiDiffCkt-OCB":
        if args.path is None:
            path = "evaluation/generated_results/BiDiffCkt_OCB101_generated_1000_results.pkl"
        else:
            path = args.path
        connectivity, validity, uniqueness, novelty = BiDiffCkt_OCB_topo_eval(path)
        print("Dataset: OCB101")
        print("BiDiffCkt OCB Topology Evaluation Results:")
        print(f"connectivity: {connectivity:.2%}. validity: {validity:.2%}. uniqueness: {uniqueness:.2%}, novelty: {novelty:.2%}")