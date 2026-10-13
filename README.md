# BiDiffCkt

## Enviroment Settings
conda env create -f environment.yml -n BiDiffCkt
conda activate BiDiffCkt

## train
python3 main.py
ablation train:
python3 main.py experiments=BiDiffCkt_Diffckt_multilabel datasets=Diffckt
python3 main.py experiments=BiDiffCkt_Digress datasets=Diffckt_multiclass 

OCB train
python3 main.py experiments=BiDiffCkt_OCB101 datasets=OCB101

LaMAGIC train
python3 main.py experiments=BiDiffCkt_LaMAGIC datasets=LaMAGIC 

## optimization
### simulation setup
python optimization/setup.py --pdk-path /data/tools/PDKs/TSMC/europractice/extracted/65/CMOS/LP/pdk/cadence_oa/tsmcN65/../models/spectre/toplevel.scs

ng spice
SIMULATOR_PATH = root / "Spice64" / "bin" / "ngspice_con.exe"

### netlist generation
python3 -m optimization.BiDiffCkt_DiffCkt_multilabel.netlist_generator


## evaluation
generated topology evaluation
python3 -m evaluation.topo_evaluation --exp_type=[BiDiffCkt, BiDiffCkt-multilabel, BiDiffCkt-Digress, BiDiffCkt-OCB]
