# B3D

Unofficial implementation of "Black-box Detection of Backdoor Attacks with Limited Information and Data" (https://arxiv.org/pdf/2103.13127.pdf).


## Usage

Both CIFAR-10 and GTSRB are supported. Select one with `--dataset-name cifar10` or `--dataset-name gtsrb`. GTSRB images are resized to 32x32 and use a 43-class ResNet-18 head.

Train a backdoored model:

```bash
python3 train.py --dataset-name cifar10 --backdoor 1
python3 train.py --dataset-name gtsrb --backdoor 1
```

Training and B3D progress is written to both the terminal and a timestamped
file in `logs/`. Use `--log-file path/to/run.log` to select a specific file.
Running `python3 main.py` uses one shared log for the training phase followed by
the B3D detection phase.

Run the complete GTSRB experiment (fresh clean and high-ASR models, reversal to
0% measured ASR by default, B3D after each model, and visual comparison):

```bash
python3 main.py --dataset-name gtsrb --backdoor 1
```

Training is deterministic by default with `--seed 0`, existing checkpoints are replaced,
and `--resume` explicitly opts into checkpoint continuation. The pipeline logs clean
accuracy and trigger ASR before and after reversal. B3D uses a 4.5 anomaly-index
threshold by default; override it with `--anomaly-threshold`.

Comparison images and trigger L1 values are written under
`images/gtsrb-backdoored-1-comparison/` by default.

CIFAR-10 checkpoints keep the existing `weights/{name}.pt` names. GTSRB checkpoints use `weights/gtsrb-{name}.pt` to prevent collisions.

Reverse-train a GTSRB model:

```bash
python3 reverse_train.py --dataset-name gtsrb --backdoor 1 \
  --checkpoint weights/gtsrb-backdoored-1.pt --asr-threshold 0
```

Run B3D (43 classes are scanned for GTSRB):

```bash
python3 b3d.py backdoored-1 --dataset-name cifar10
python3 b3d.py backdoored-1 --dataset-name gtsrb

# Optional faster diagnostic scan; full scans are the default.
python3 b3d.py backdoored-1 --dataset-name cifar10 --max-batches 100 --samples 10
```

B3D trigger application is vectorized for lower runtime without changing the default
scan depth. Trigger setups are defined in `masks.py`. B3D triggers are saved beside the selected checkpoint with a `-TRIGGERS.pt` suffix.
