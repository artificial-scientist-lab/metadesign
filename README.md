# Meta-designing quantum experiments with language models
**Sören Arlt, Haonan Duan, Felix Li, Sang Michael Xie, Yuhuai Wu, Mario Krenn**

This repository implements a sequence-to-sequence (encoder–decoder) transformer that generates **meta-solutions**—human-readable Python programs that generalize experimental designs across *entire classes* of targets (rather than a single target instance). The implementation is inspired by the minimal and effective NanoGPT training framework, adapted here for sequence-to-sequence code generation.

---

## Paper

**Meta-designing quantum experiments with language models**  
Sören Arlt, Haonan Duan, Felix Li, Sang Michael Xie, Yuhuai Wu, Mario Krenn  
*Nature Machine Intelligence* **8**, 148–157 (2026) — **Published: 19 February 2026**  
DOI: **10.1038/s42256-025-01153-0**

Links:
- Published version (Nature Machine Intelligence): https://www.nature.com/articles/s42256-025-01153-0
- arXiv preprint: https://arxiv.org/abs/2406.02470

Artifacts:
- **Code archive (Zenodo, v1.0):** https://doi.org/10.5281/zenodo.17287889
- **Model checkpoints + example training data (Zenodo):** https://doi.org/10.5281/zenodo.14899993

---

## Overview

This project trains a sequence-to-sequence transformer to:
1. **Generate synthetic training data** from randomized program templates under hard structural constraints.
2. **Learn the inverse map**: from structured quantum targets (sequence A) → executable Python programs (sequence B).
3. **Sample and evaluate** many candidate programs, using physics-based simulation to score fidelity to the target class.

The resulting programs are interpretable and often generalize to larger system sizes than those seen during training.

---

## Quickstart

### Installation

```bash
git clone https://github.com/artificial-scientist-lab/metadesign.git
cd metadesign
pip install -r requirements.txt
````

### Download checkpoints (recommended)

Download the released checkpoints from the Zenodo record **10.5281/zenodo.14899993** and place them at:

```text
ckpt_main/ckpt_main.pt
ckpt_circuit/ckpt_circuit.pt
```

If the downloaded filenames are `ckpt_main.pt` and `ckpt_circuit.pt`, you can do:

```bash
mkdir -p ckpt_main ckpt_circuit
mv ckpt_main.pt ckpt_main/ckpt_main.pt
mv ckpt_circuit.pt ckpt_circuit/ckpt_circuit.pt
```

---

## Sampling demos

### Main task (quantum optics / graph-based setups)

Run:

```bash
python sample_main.py
```

Optional CLI flags (see `sample_main.py`):

```bash
python sample_main.py --task_id 0 --temp 0.2 --topp 0.5
```

Notes:

* `task_id` selects a target *state class* (mapped internally to a list of modes in `sample_main.py`).
* Sampling is probabilistic; running multiple times can yield different candidate meta-solutions.

### Quantum circuits example

Run:

```bash
python sample_circuit.py
```

This loads `ckpt_circuit/ckpt_circuit.pt`, generates circuit code, and prints the resulting states for several system sizes.

---

## Training

### Important: config file is a positional argument

`train.py` loads a Python config file via `sys.argv[1]`.
So you run training like this (no `--config` flag):

#### Main model

```bash
python train.py ckpt_main/config.py
```

#### Circuit model

For the provided circuit config (set up for multi-GPU DDP in our experiments), use `torchrun`, e.g.:

```bash
torchrun --standalone --nproc_per_node=4 train.py ckpt_circuit/config.py
```

To train on fewer GPUs / a single GPU:

* either adjust `ckpt_circuit/config.py` accordingly,
* or create a new config file (recommended) that matches your setup.

---

## Repository structure (high level)

* `ckpt_main/`

  * `config.py` (training config for main task)
  * place `ckpt_main.pt` here as `ckpt_main/ckpt_main.pt` for sampling
* `ckpt_circuit/`

  * `config.py` (training config for circuit task)
  * place `ckpt_circuit.pt` here as `ckpt_circuit/ckpt_circuit.pt` for sampling
* `data_main/` — data generation + utilities for the main task
* `data_circuits/` — data generation for the circuits example
* `seq2seq.py` — encoder–decoder transformer implementation
* `train.py` — training loop
* `sample_main.py` — sampling + fidelity evaluation for main task
* `sample_circuit.py` — sampling + simulation for circuit task
* `hdf5dataloader.py`, `helper.py` — utilities
* `requirements.txt`

---

## Reproducibility / data

* **Model checkpoints and example training data** are available on Zenodo:
  [https://doi.org/10.5281/zenodo.14899993](https://doi.org/10.5281/zenodo.14899993)

* **Archived code (v1.0)** is available on Zenodo:
  [https://doi.org/10.5281/zenodo.17287889](https://doi.org/10.5281/zenodo.17287889)

This repository also includes the complete code to regenerate synthetic datasets and retrain models.
(Full-scale reproduction can be computationally expensive; the Zenodo record is meant to provide convenient access to checkpoints and a small example dataset.)

---

## Average fidelities when sampling on validation data (from the original README)

A more interpretable measure of training progress is to compare the states resulting from the generated code with the states resulting from the ground-truth code (from the training/validation data).

We compute the fidelity (0 = no overlap, 1 = perfect overlap) for `N = 0, 1, 2, 3, 4` and average over the validation set.

Main model (trained from scratch):

```
[0.877±0.002, 0.661±0.003, 0.402±0.003, 0.112±0.002, 0.112±0.002]
```

Fine-tuned Llama3 8B model:

```
[0.950±0.004, 0.584±0.008, 0.316±0.008, 0.008±0.002, 0.000±0.0]
```

---

## Pseudocode / pipeline details

The original README contained a very detailed walkthrough of the synthetic data generation and topology constraints.
If you want to keep that level of detail without making the top of the README overwhelming, one option is to put it into a collapsible block:

<details>
<summary><strong>Click to expand: detailed pseudocode and data-generation pipeline</strong></summary>

(You can paste the long “Pseudocode” section from the previous README here verbatim, or move it into a separate markdown file under `docs/`.)

</details>

---

## How to cite

If you use this repository, please cite the **Nature Machine Intelligence** paper.
If you use the released checkpoints and/or want to cite the archived artifacts, you can also cite the Zenodo records.

### Paper (recommended)

```bibtex
@article{arlt2026metadesigning,
  title   = {Meta-designing quantum experiments with language models},
  author  = {Arlt, S{\"o}ren and Duan, Haonan and Li, Felix and Xie, Sang Michael and Wu, Yuhuai and Krenn, Mario},
  journal = {Nature Machine Intelligence},
  volume  = {8},
  pages   = {148--157},
  year    = {2026},
  doi     = {10.1038/s42256-025-01153-0}
}
```

### Checkpoints + example training data (Zenodo)

```bibtex
@software{arlt2025_checkpoints_metadesign,
  author    = {Arlt, S{\"o}ren},
  title     = {Model checkpoints and example training data for {``Meta-designing quantum experiments with language models''}},
  publisher = {Zenodo},
  year      = {2025},
  version   = {v1},
  doi       = {10.5281/zenodo.14899993}
}
```

### Code archive (Zenodo)

```bibtex
@software{arlt_krenn_2025_metadesign_code,
  author    = {Arlt, S{\"o}ren and Krenn, Mario},
  title     = {artificial-scientist-lab/metadesign: V1.0},
  publisher = {Zenodo},
  year      = {2025},
  version   = {V1.0},
  doi       = {10.5281/zenodo.17287889}
}
```

---

## Acknowledgements

* The overall training code structure is largely adapted from nanoGPT: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
  (This repo uses an **encoder–decoder** transformer rather than a decoder-only model.)

---

## Related links

* Nature Machine Intelligence article: [https://www.nature.com/articles/s42256-025-01153-0](https://www.nature.com/articles/s42256-025-01153-0)
* arXiv: [https://arxiv.org/abs/2406.02470](https://arxiv.org/abs/2406.02470)
* Zenodo (code archive): [https://doi.org/10.5281/zenodo.17287889](https://doi.org/10.5281/zenodo.17287889)
* Zenodo (checkpoints + example data): [https://doi.org/10.5281/zenodo.14899993](https://doi.org/10.5281/zenodo.14899993)
