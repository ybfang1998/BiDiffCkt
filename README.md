# BiDiffCkt

## Enviroment Setup
### Train and Sampe setup
This setup requires Anaconda. Run the following command below:
```bash
conda env create -f environment.yml -n BiDiffCkt
```
To activate the environment:
```bash
conda activate BiDiffCkt
```
**[!NOTE]** Our experiments were conducted using an NVIDIA RTX PRO 6000 Blackwell GPU, which requires PyTorch 2.7.0 or later with CUDA 12.8 or later. However, these versions may not be compatible with all GPU environments. Please ensure that you install PyTorch and CUDA versions compatible with your GPU and system configuration.

### Simulation & Optimization setup
#### Transistor-level (DiffCkt):
Transistor-level circuit simulations require the Cadence Spectre Simulator. Additionally, the **cdspythonsrr** package is required to read simulation results stored in PSF files. This package requires Python 3.9.16, NumPy, and Virtuoso Studio IC23.1 ISR13 or later. For more details, please refer to the Virtuoso User Guide, specifically the section "Working with Simulation Data Using SRR-Python."

This setup the header of the testbenches for transistor-level circuits optimisation. Run the following command by replace your TSMC65nm PDK path:
```bash
python optimization/setup.py --pdk-path ../toplevel.scs
```
#### Behavioral-level (LaMAGIC):
Please specify the path to your **ngspice simulator** in [`simulator.py`](./optimization/BiDiffCkt_LaMAGIC/simulator.py):
```python
SIMULATOR_PATH = root / "Spice64" / "bin" / "ngspice_con.exe"
```

Replace the default path with the actual location of the ngspice executable on your system.

## How to Run

### Training

#### 1. Transistor-Level Circuit Generation (DiffCkt)

**Full Model (BiDiffCkt)**

```bash
python3 main.py
```

**Ablation Studies**

- **Without Edge Terminal Encapsulation**

  ```bash
  python3 main.py experiments=BiDiffCkt_Diffckt_multilabel datasets=Diffckt
  ```

- **Without Bipartite Graph Transformer**

  ```bash
  python3 main.py experiments=BiDiffCkt_Digress datasets=Diffckt_multiclass
  ```

#### 2. Behavioral-Level Circuit Generation

**OCB101 Dataset**

```bash
python3 main.py experiments=BiDiffCkt_OCB101 datasets=OCB101
```

**LaMAGIC Dataset**

```bash
python3 main.py experiments=BiDiffCkt_LaMAGIC datasets=LaMAGIC
```

### Optimization

#### 1. Netlist Generation

Generate SPICE netlists from the circuits produced by BiDiffCkt:

```bash
python3 -m optimization.BiDiffCkt_DiffCkt.netlist_generator
```

#### 2. Bayesian Optimization

Run Bayesian optimization to optimize device parameters and evaluate circuit performance:

```bash
python3 optina_bo.py --n-trials 100 --max-workers 8 --target-level high
```

**Arguments:**

- `--n-trials`: Number of optimization trials.
- `--max-workers`: Maximum number of parallel workers.
- `--target-level`: Target specification level. Available options: `external`, `high`, `med`, and `low`.

Adjust these parameters according to your computational resources and optimization requirements.


### Evaluation

#### Topology Evaluation

Evaluate the quality of generated circuit topologies using the following command:

```bash
python3 -m evaluation.topo_evaluation --exp_type BiDiffCkt
```

**Arguments:**

- `--exp_type`: Specifies the experiment to evaluate. Available options:
  - `BiDiffCkt`: Full model.
  - `BiDiffCkt-multilabel`: Without edge terminal encapsulation.
  - `BiDiffCkt-Digress`: Without bipartite graph transformer.
  - `BiDiffCkt-OCB`: Behavioral-level circuit generation on
