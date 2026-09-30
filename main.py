import os
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.get_best_gpu())
import pandas
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import pathlib

from train import *
from dummy_testing import *
from torch.utils.data import DataLoader, random_split
import torch.distributions as dist
from modules.datasets.graph_to_bipartite import *
from modules.datasets.BipartiteGraph import Diffckt_Multiclass_BipartiteGraphDataset, OCB_BipartiteGraphDataset, Diffckt_BipartiteGraphDataset, LaMAGIC_BipartiteGraphDataset
import re
import warnings
from tqdm import tqdm

@hydra.main(version_base='1.3', config_path='./modules/configs', config_name='config')
def main(cfg: DictConfig):
    # Settings
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

    project_name = cfg.experiments.general.project_name
    model_name = cfg.experiments.general.model_name
    dataset_name = cfg.experiments.general.dataset_name
    experiment_type = cfg.experiments.general.experiment_type
    # Set seed
    utils.set_seed(cfg.train.seed)
    data_generator = torch.Generator().manual_seed(cfg.train.seed)
    # Read Dataset
    if dataset_name == "OCB101":
        n_U_max = 6
        n_V_max = 4

        data_path = "datasets/OCB/ckt_bench_101.pkl"
        data_path = os.path.join(root_dir, data_path)
        specification_path = "datasets/OCB/perform101_reform.csv"
        specification_path = os.path.join(root_dir, specification_path)

        # Read the dataset and clean Invalid Circuit
        with open(data_path, 'rb') as f:
            datasets = pickle.load(f)
        spec = pandas.read_csv(specification_path)
        G = [g_pair[0] for g_pair in datasets[0]] + [g_pair[0] for g_pair in datasets[1]]
        G, spec = invalid_circuit_remove(G, spec)
        # min-max scaling for spec(gain, pm, bw)
        spec_scaled = spec_scaling(spec)
    elif dataset_name == "Diffckt":
        if experiment_type == "train" or experiment_type == "sample":
            data_path = "datasets/DiffCkt/Diffckt_in_Bipartite.pkl"
        elif experiment_type == "dummy_test":
            use_debug_data = cfg.experiments.dummy_test.get(
                "use_debug_data", False
            )
            data_path = (
                "datasets/DiffCkt/Diffckt_in_Bipartite_debug.pkl"
                if use_debug_data
                else "datasets/DiffCkt/Diffckt_in_Bipartite.pkl"
            )
        elif experiment_type == "debug":
            data_path = "datasets/DiffCkt/Diffckt_in_Bipartite_debug.pkl"
        data_path = os.path.join(root_dir, data_path)
        if experiment_type == "dummy_test":
            print(f"[Probe data] Loading dataset: {data_path}", flush=True)
        # Read the dataset and clean Invalid Circuit
        with open(data_path, 'rb') as f:
            G = pickle.load(f)
        if experiment_type == "dummy_test":
            print(f"[Probe data] Loaded {len(G):,} circuits.", flush=True)
    elif dataset_name == "Diffckt_multiclass":
        if experiment_type == "train" or experiment_type == "sample":
            data_path = "datasets/DiffCkt/Diffckt_in_Bipartite_multiclass.pkl"
            # #debug
            # data_path = "datasets/DiffCkt/Diffckt_in_Bipartite_debug.pkl"
        elif experiment_type == "debug":
            data_path = "datasets/DiffCkt/Diffckt_in_Bipartite_multiclass_debug.pkl"
        elif experiment_type == "dummy_test":
            use_debug_data = cfg.experiments.dummy_test.get(
                "use_debug_data", False
            )
            data_path = (
                "datasets/DiffCkt/Diffckt_in_Bipartite_multiclass_debug.pkl"
                if use_debug_data
                else "datasets/DiffCkt/Diffckt_in_Bipartite_multiclass.pkl"
            )
        data_path = os.path.join(root_dir, data_path)
        # Read the dataset and clean Invalid Circuit
        with open(data_path, 'rb') as f:
            G = pickle.load(f)
    elif dataset_name == "LaMAGIC":
        data_path = "datasets/LaMAGIC/LaMAGIC_345comp_bipartite_graph.pkl"
        data_path = os.path.join(root_dir, data_path)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise NotImplementedError

    # Data Processing
    if dataset_name == "OCB101":
        # min/max number of devices
        MIN_N_DEVICE = 2
        MAX_N_DEVICE = 6
        # Transfer the DAG to Bipartite Graph
        Bipartite_G = []
        for g in G:
            Bipartite_G.append(ocb_to_bipartite(g, n_U_max, n_V_max))

        dataset = OCB_BipartiteGraphDataset(Bipartite_G)
        nodes_dist = dataset.nodes_dist
        pos_weight = None
        # Train/Test split with 9: 1
        train_dataset, test_dataset = random_split(dataset, [0.9, 0.1], generator=data_generator)

        # Marginal X/E
        marginal_U = to_tf_device(torch.from_numpy(dataset.marginal_U_type), None)
        marginal_V = to_tf_device(torch.from_numpy(dataset.marginal_V_type), None)
        marginal_X = [marginal_U, marginal_V]
        marginal_E = to_tf_device(torch.from_numpy(dataset.marginal_E_type), None)
        train_loader = DataLoader(dataset, batch_size=cfg.experiments.train.batch_size, shuffle=True,
                                    num_workers=cfg.train.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=cfg.experiments.test.batch_size, shuffle=False,
                                    num_workers=cfg.train.num_workers)
    elif dataset_name == "LaMAGIC":

        dataset = LaMAGIC_BipartiteGraphDataset(data)
        nodes_dist = dataset.nodes_dist
        pos_weight = None
        # Train/Test split with 9: 1
        train_dataset, test_dataset = random_split(dataset, [0.9, 0.1], generator=data_generator)
        
        # Marginal X/E
        marginal_U = to_tf_device(torch.from_numpy(dataset.marginal_U_type), None)
        marginal_V = to_tf_device(torch.from_numpy(dataset.marginal_V_type), None)
        marginal_X = [marginal_U, marginal_V]
        marginal_E = to_tf_device(torch.from_numpy(dataset.marginal_E_type), None)
        train_loader = DataLoader(dataset, batch_size=cfg.experiments.train.batch_size, shuffle=True,
                                    num_workers=cfg.train.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=cfg.experiments.test.batch_size, shuffle=False,
                                    num_workers=cfg.train.num_workers)
        
    elif dataset_name == "Diffckt":
        if experiment_type == "train" or experiment_type == "debug":
            dataset = Diffckt_BipartiteGraphDataset(G)
        elif experiment_type == "dummy_test":
            dataset = Diffckt_BipartiteGraphDataset(G, conditional=True)
        nodes_dist = dataset.nodes_dist
        pos_weight = dataset.pos_weight

        # Marginal X/E
        marginal_U = to_tf_device(torch.from_numpy(dataset.marginal_U_type), None)
        marginal_V = to_tf_device(torch.from_numpy(dataset.marginal_V_type), None)
        marginal_X = [marginal_U, marginal_V]
        marginal_E = to_tf_device(torch.from_numpy(dataset.marginal_E_type), None)
        if experiment_type == "dummy_test":
            train_dataset, test_dataset = random_split(dataset, [0.9, 0.1], generator=data_generator)
            train_loader = DataLoader(train_dataset, batch_size=cfg.experiments.dummy_test.batch_size, shuffle=True,
                                    num_workers=cfg.train.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=cfg.experiments.dummy_test.batch_size, shuffle=False,
                                    num_workers=cfg.train.num_workers)
        else:
            train_loader = DataLoader(dataset, batch_size=cfg.experiments.train.batch_size, shuffle=True,
                                    num_workers=cfg.train.num_workers)
        
    elif dataset_name == "Diffckt_multiclass":
        if experiment_type == "train" or experiment_type == "debug":
            dataset = Diffckt_Multiclass_BipartiteGraphDataset(G)
        elif experiment_type == "dummy_test":
            dataset = Diffckt_Multiclass_BipartiteGraphDataset(G, conditional=True)
        nodes_dist = dataset.nodes_dist
        pos_weight = None
        
        # Marginal X/E
        marginal_U = to_tf_device(torch.from_numpy(dataset.marginal_U_type), None)
        marginal_V = to_tf_device(torch.from_numpy(dataset.marginal_V_type), None)
        marginal_X = [marginal_U, marginal_V]
        marginal_E = to_tf_device(torch.from_numpy(dataset.marginal_E_type), None)
        if experiment_type == "debug":
            marginal_E = to_tf_device(torch.ones(21) / 21, None)
        if experiment_type == "dummy_test":
            train_dataset, test_dataset = random_split(dataset, [0.9, 0.1], generator=data_generator)
            train_loader = DataLoader(train_dataset, batch_size=cfg.experiments.dummy_test.batch_size, shuffle=True,
                                    num_workers=cfg.train.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=cfg.experiments.dummy_test.batch_size, shuffle=False,
                                    num_workers=cfg.train.num_workers)
        else:
            train_loader = DataLoader(dataset, batch_size=cfg.experiments.train.batch_size, shuffle=True,
                                    num_workers=cfg.train.num_workers)

        
    # Train
    if experiment_type == "train" or experiment_type == "debug":
        train(cfg, model_name, project_name, train_loader, marginal_X, marginal_E, nodes_dist, experiment_type, pos_weight=pos_weight)
    elif experiment_type == "dummy_test":
        model_path = to_absolute_path(cfg.experiments.dummy_test.model_path)
        dummy_train(cfg, model_path, train_loader, test_loader)

if __name__ == '__main__':
    main()

