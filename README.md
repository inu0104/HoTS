# HoTS: Homophily-Aware Temperature Scaling for Graph Neural Network Calibration

Inwoo Tae and Yongjae Lee

HoTS calibrates frozen GNN logits using predictive entropy and estimated local
homophily. A positive scalar temperature preserves each node's predicted class:

$$T_i=T_{\mathrm{base}}+\frac{\beta\sqrt{2K\log K(1-e_i)}}{(|\hat{\tilde h}_i|+\varepsilon)^\alpha}.$$

The public `HoTS` class is the final main implementation: predictor supervision
uses only training/validation labels and observed non-self neighbors, the
predictor width is 32, and the exponent floor is 0.01. Test labels are used for
evaluation and retrospective diagnostics, never to fit the estimator or temperature.

## Installation

The recorded experiment environment is Linux, Python 3.10, PyTorch 2.4.0 and
DGL 2.4.0 with CUDA 12.1. Install the matching binary packages before this project:

```bash
git clone https://github.com/inu0104/HoTS.git
cd HoTS
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
python -m pip install -e .
```

Binary sources: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
and [DGL's matching wheel index](https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html).
The commands below also accept `--gpu -1` to run on CPU in an installed environment.

## Run experiments

Datasets are downloaded automatically on first use.

The data directory defaults to `../data/`; set `HOTS_DATA_ROOT` to override it.
All main experiments use per-seed random 20/10/70 train/validation/test splits.
GCN and GAT are the main backbones; the runner also supports additional backbones.

Run a single main-method experiment:

```bash
python run_experiment.py --dataset Cora --model GCN --seeds 0 --gpu 0 \
  --calibrator Uncal,TS,HoTS --early_stop loss --deterministic
```

Run the main benchmark (18 datasets, GCN/GAT, seeds 0–9):

```bash
python run_experiment.py --dataset all --model GCN,GAT --seeds 0-9 --gpu 0 \
  --calibrator main --save_dir results/main --early_stop loss --deterministic
```

`--calibrator all` and `--calibrator main` select Uncal, TS, VS, ETS, HTS, CaGCN,
GATS, GETS, WATS, and HoTS. Existing `model.pt` files are reused; otherwise the backbone
is trained. Use a fresh output directory for a new protocol or checkpoint set.
The default GPU is 0. GATS can run out of memory on Reddit, as in the paper;
that failure is reported rather than assigned a metric.

## Outputs

Each run writes `<output>/<dataset>/<backbone>/seed_<seed>/` containing the
checkpoint, metric JSON files, test-node predictions, and (for HoTS) predictor
weights, all-node homophily estimates, and temperature parameters.

Current main results, pooled over GCN and GAT:

| Metric | HoTS |
|---|---:|
| Mean ECE (%) | 4.79 |
| Average rank | 3.33 |
| Mean NLL | 0.92 |
| Mean degree-stratified ECE (%) | 5.90 |

These reported results use saved backbone checkpoints. Retraining a backbone
in a different environment may change numerical results.

## Layout

```text
hots/                 # data loaders, backbones, calibrators
configs/              # dataset and calibrator settings
run_experiment.py     # training, calibration, evaluation
pyproject.toml        # package metadata and dependencies
```

Only source code, configuration, and installation documentation are distributed.
MIT License; see [LICENSE](LICENSE).
