# Claudin

# TCRM and SIDM

This repository provides two pathology image classification modules based on the UNI pathology foundation model:

- `TCRM.py`: tumor cell recognition module.
- `SIDM.py`: staining/intensity recognition module.

Both modules use patch-level histopathology images organized with `torchvision.datasets.ImageFolder`, load a local UNI checkpoint, and train task-specific classifiers on top of the UNI visual encoder.

## Repository Structure

```text
.
├── TCRM.py              # Tumor recognition module
├── SIDM.py              # Intensity recognition module
├── lora.py              # LoRA implementation for ViT/timm models
├── requirements.txt     # pip dependencies
└── env.yml              # conda environment file
```

## Environment Setup

We recommend using conda. The provided environment file creates a Python 3.8 environment and installs the required packages.

```bash
conda env create -f env.yml
conda activate claudin
```

Alternatively, create your own environment and install dependencies with pip:

```bash
conda create -n claudin python=3.8 -y
conda activate claudin
pip install -r requirements.txt
```

Please install a PyTorch version compatible with your CUDA driver if the default installation does not match your GPU environment. See the official PyTorch installation guide for CUDA-specific commands.

## Download UNI Weights

This code uses the UNI ViT-L/16 checkpoint. Please download UNI from the official repository:

https://github.com/mahmoodlab/UNI

The UNI model weights are hosted on Hugging Face and may require access approval and authentication. After obtaining access, download `pytorch_model.bin` and place it in a local directory, for example:

```text
/path/to/UNI/
└── pytorch_model.bin
```

One possible download method is:

```python
from huggingface_hub import login, hf_hub_download

login()
hf_hub_download(
    repo_id="MahmoodLab/UNI",
    filename="pytorch_model.bin",
    local_dir="/path/to/UNI",
    force_download=True,
)
```

After downloading, update the UNI path in the scripts:

```python
# TCRM.py
visual_model.load_state_dict(
    torch.load(os.path.join("PATH TO UNI MODEL", "pytorch_model.bin"), map_location="cuda"),
    strict=True,
)

# SIDM.py
visual_model.load_state_dict(
    torch.load(os.path.join("PATH TO UNI", "pytorch_model.bin"), map_location="cuda"),
    strict=True,
)
```

Replace `PATH TO UNI MODEL` and `PATH TO UNI` with the actual directory containing `pytorch_model.bin`.

## Dataset Preparation

The training and testing images should be organized in ImageFolder format. Each class should have its own subdirectory.

```text
/path/to/dataset/
├── train/
│   ├── class_0/
│   │   ├── image_001.png
│   │   └── ...
│   └── class_1/
│       ├── image_001.png
│       └── ...
└── test/
    ├── class_0/
    │   ├── image_001.png
    │   └── ...
    └── class_1/
        ├── image_001.png
        └── ...
```

For `SIDM.py`, use the corresponding intensity classes. The current script is configured with:

```python
num_classes = 3
```

For `TCRM.py`, the current script is configured with:

```python
num_classes = 2
```

Before running, update the dataset path in each script:

```python
dataset = "PATH TO DATASET"
```

Replace `PATH TO DATASET` with the actual dataset directory.

## Configuration

Before training, check the following settings in `TCRM.py` or `SIDM.py`:

- `CUDA_VISIBLE_DEVICES`: GPU index used for training.
- `dataset`: path to the dataset root.
- `num_classes`: number of target classes.
- `batch_size`: batch size, adjust according to GPU memory.
- `num_epochs`: number of training epochs.
- UNI checkpoint path: directory containing `pytorch_model.bin`.
- Output path prefixes: `./MODEL_SAVE` in `TCRM.py` and `./SAVE` in `SIDM.py`.

Example:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset = "/data/my_dataset"
batch_size = 16
num_classes = 2
num_epochs = 20
```

## Run Training

Run the tumor recognition module:

```bash
python TCRM.py
```

Run the intensity recognition module:

```bash
python SIDM.py
```

Training logs will print loss, accuracy, macro-AUC, and micro-AUC for each epoch.

## Outputs

`TCRM.py` saves LoRA parameters and training history:

```text
MODEL_SAVE_model_1.safetensors
MODEL_SAVE_model_2.safetensors
...
MODEL_SAVE_history.pt
lora_loss_curve.png
lora_accuracy_curve.png
```

`SIDM.py` saves the trained model, training history, and curves:

```text
SAVE_model_1.pt
SAVE_model_2.pt
...
SAVE_history.pt
adapter_loss_curve.png
adapter_accuracy_curve.png
```

## Notes

- The scripts assume CUDA is available. If you want to run on CPU, update the device-related code accordingly.
- Make sure the class folders under `train/` and `test/` have consistent names and ordering.
- If you change the number of classes, update all related class-count settings in the script.
- The paths in the code are placeholders and must be modified according to your local file system.

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{your_citation_key,
  title   = {Your Paper Title},
  author  = {Your Author List},
  journal = {Your Journal or Conference},
  year    = {2026}
}
```

This project also uses the UNI pathology foundation model. Please cite the UNI paper if you use UNI weights:

```bibtex
@article{chen2024uni,
  title   = {Towards a General-Purpose Foundation Model for Computational Pathology},
  author  = {Chen, Richard J. and Ding, Tong and Lu, Ming Y. and Williamson, Drew F. K. and others},
  journal = {Nature Medicine},
  year    = {2024}
}
```

## Acknowledgements

We thank the authors of UNI and the open-source libraries used in this repository, including PyTorch, timm, torchvision, scikit-learn, and Hugging Face.
