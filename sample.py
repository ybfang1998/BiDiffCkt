import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import os
import pathlib
from pathlib import Path

from train import *
from modules.datasets.graph_to_bipartite import *
from modules.datasets.BipartiteGraph import OCB_BipartiteGraphDataset, AnalogGenie_BipartiteGraphDataset, Diffckt_BipartiteGraphDataset
import utils
# os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.get_best_gpu())
import pickle

from modules.diffusion.discrete_diffusion import DiscreteDiffusion
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore", category=PossibleUserWarning)
import time

def sample(cfg, experiment_type, model_path, predict_loader, results_path):
    use_gpu = cfg.train.gpus > 0 and torch.cuda.is_available()
    model = DiscreteDiffusion.load_from_checkpoint(model_path, weights_only=False)
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                        # strategy="find_unused_parameters_true",  # Needed to load old checkpoints
                        accelerator='gpu' if use_gpu else 'cpu',
                        devices=[1] if use_gpu else 1,
                        fast_dev_run=experiment_type == 'debug',
                        enable_progress_bar=False,
                        log_every_n_steps=50 if experiment_type != 'debug' else 1,
                        logger=[])
    start_time = time.time()
    G = trainer.predict(model, dataloaders=predict_loader)
    all_G = []
    for batch_out in G:
        if isinstance(batch_out["G"], list):
            all_G.extend(batch_out["G"])
        else:
            all_G.append(batch_out["G"])
    end_time = time.time()
    print(f"Sampling completed in {end_time - start_time:.2f} seconds.")
    results = {'G': all_G}
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
@hydra.main(version_base='1.3', config_path='./modules/configs', config_name='config')
def main(cfg: DictConfig):
    # Settings
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]
    project_name = cfg.experiments.general.project_name
    experiment_type = cfg.experiments.general.experiment_type
    # Set seed
    utils.set_seed(cfg.train.seed)

    # Fake predict_loader to determine the number of samples to generate
    n_samples = cfg.experiments.test.n_samples
    if cfg.experiments.test.num_nodes == "Distributed":
        indices = torch.arange(n_samples)
        predict_dataset = TensorDataset(indices)
    elif cfg.experiments.test.num_nodes == "Random":
        max_num_nodes = cfg.datasets.BiDiffckt.max_num_nodes
        min_num_nodes = cfg.datasets.BiDiffckt.min_num_nodes
        max_num_U = max_num_nodes['U']
        max_num_V = max_num_nodes['V']
        min_num_U = min_num_nodes['U']
        min_num_V = min_num_nodes['V']
        # Generate random number of nodes for each sample
        num_U_nodes = torch.randint(min_num_U, max_num_U + 1, (n_samples,))
        num_V_nodes = torch.randint(min_num_V, max_num_V + 1, (n_samples,))
        predict_dataset = TensorDataset(num_U_nodes, num_V_nodes)
    else:
        raise ValueError(f"Invalid num_nodes value: {cfg.experiments.test.num_nodes}. Must be 'Distributed' or 'Random'.")
    predict_loader = DataLoader(predict_dataset, batch_size=cfg.experiments.test.batch_size)

    model_path = to_absolute_path(cfg.experiments.test.model_path)
    results_path = to_absolute_path(cfg.experiments.test.results_path)
    sample(cfg, experiment_type, model_path, predict_loader, results_path)
if __name__ == '__main__':
    main()