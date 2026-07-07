# AMM for Traffic Signal Control

This repository contains the code for [Planning Under Observation Mismatch for Traffic Signal Control via Adaptive Modular World Models](https://arxiv.org/abs/2501.02548), which is accepted by The 36th International Conference on Automated Planning and Scheduling (ICAPS 2026).


## Requirements

The Python packages are listed in `requirements.txt`. CityFlow simulator can be installed following [official docs](https://cityflow.readthedocs.io/en/latest/install.html).

## Runs

Meta-train the shared dynamics model on two source domains:

```bash
python program/train/meta_train.py \
  --config1 Cityflow_run/cfg/config_4x4.json \
  --config2 Cityflow_run/cfg/config_28x7.json \
  --save_model_dir outputs/models/meta_dynamics/4x4_and_28x7/
```

Adapt AMM to a target domain:

```bash
python program/train/target_adapt.py \
  --config Cityflow_run/cfg/config_16x3.json \
  --dynamics_model_dir outputs/models/meta_dynamics/4x4_and_28x7/<RUN_DIR>/ \
  --episode 60 \
  --save_model_freq 60 \
  --save_model_dir outputs/models/amm/16x3/
```

Evaluate a trained AMM checkpoint:

```bash
python program/eval/evaluate_amm.py \
  --config Cityflow_run/cfg/config_16x3.json \
  --adapter_model_dir outputs/models/amm/16x3/<RUN_DIR>/ \
  --dynamics_model_dir outputs/models/amm/16x3/<RUN_DIR>/
```

Run the non-modular ablation:

```bash
python program/train/train_non_modular_ablation.py \
  --mode meta_train \
  --config Cityflow_run/cfg/config_16x3.json \
  --episode 60 \
  --save_model_freq 60 \
  --save_model_dir outputs/models/non_modular/16x3/
```

Run the target-budget sweep used for the data-efficiency experiment:

```bash
python program/experiments/run_data_efficiency.py \
  --target_config Cityflow_run/cfg/config_16x3.json \
  --dynamics_model_dir outputs/models/meta_dynamics/4x4_and_28x7/<RUN_DIR>/
```

For the w/o-ML variant, sequentially call `program/train/target_adapt.py` on source domains and use the resulting `dynamics_model.pth` as the initialization for target adaptation.
