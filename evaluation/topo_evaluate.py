import pickle
from topology_eval_utils import (
    load_bipartite_graphs,
    valid_graph_statistics,
)

from optimization.BiDiffCkt_Diffckt_multiclass.netlist_generator import MulticlassBipartite2Netlist
from optimization.BiDiffCkt_Diffckt.netlist_generator import Bipartite2Netlist as MultilabelBipartite2Netlist
from optimization.BiDiffCkt_OCB.netlist_generator import valid_bidiffckt_OCB, OCB_bipartite_graph_to_netlist

def BiDiffCkt_multiclass_topo_eval(dataset_path):
    """Evaluate the topologies in the given dataset."""
    # Load generated data
    generated_data = load_bipartite_graphs(dataset_path)

    # Validity Check
    valid_rate = valid_graph_statistics(generated_data)['valid_rate']
    # print(
    #     f"valid rate: {valid_rate:.2%}"
    # )
    success_count = 0
    for i, g in enumerate(generated_data):
        try:
            netlist = MulticlassBipartite2Netlist(g, i, output_dir=None)
            if netlist:
                success_count += 1
        except ValueError as e:
            pass
    # print(f"Simulatability: {success_count / len(generated_data):.2%}")
    return valid_rate, success_count / len(generated_data)

def BiDiffCkt_multilabel_topo_eval(dataset_path):
    """Evaluate the topologies in the given dataset."""
    # Load generated data
    generated_data = load_bipartite_graphs(dataset_path)

    # Validity Check
    valid_rate = valid_graph_statistics(generated_data)['valid_rate']
    # print(
    #     f"valid rate: {valid_rate:.2%}"
    # )
    success_count = 0
    for i, g in enumerate(generated_data):
        try:
            netlist = MultilabelBipartite2Netlist(g, i, output_dir=None)
            if netlist:
                success_count += 1
        except ValueError as e:
            pass
    # print(f"Simulatability: {success_count / len(generated_data):.2%}")
    return valid_rate, success_count / len(generated_data)

def BiDiffCkt_OCB_topo_eval(dataset_path):
    with open(dataset_path, "rb") as f:
        generated_data = pickle.load(f)['G']

    valid_graph_count = 0
    for i, g in enumerate(generated_data):
        if valid_bidiffckt_OCB(g):
            valid_graph_count += 1
    # print(f"Valid rate: {valid_graph_count / len(generated_data):.2%}")
    valid_rate = valid_graph_count / len(generated_data)

    success_count = 0
    for i, g in enumerate(generated_data):
        if valid_bidiffckt_OCB(g):
            # netlists init
            try:
                OCB_bipartite_graph_to_netlist(g, i, output_dir=None)
                success_count +=1
            except ValueError as e:
                print(f"Error in graph {i}: {e}")
    # print(f"Simulatability: {success_count / valid_graph_count:.2%}")
    return valid_rate, success_count / valid_graph_count