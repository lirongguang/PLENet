# PLENet
## High-Precision Prediction of Protein-Protein Interaction Sites
PLENet is a novel deep learning framework designed to accurately identify protein-protein interaction (PPI) sites. By integrating a pre-trained protein language model with geometric deep learning, PLENet addresses the limitations of traditional computational methods in both accuracy and generalization.
### Core Dependencies
* **Python**: 3.7
* **TensorFlow**: 2.6.0
* **PyTorch**: 1.12.1 (CUDA 11.3)
* **Transformers**: 4.21.1
* **DGL**: 0.7.0 (CUDA 10.1 version based on `dgl-cu101`)
* **Biopython**: 1.79
## Dataset Download
> **Important:** You must download and configure the dataset before running the model.
1. Download the dataset package here: [https://zenodo.org/records/18443340](https://zenodo.org/records/18443340)
2. Extract the contents fully into the `inputs/` folder.
3. Ensure the `inputs` folder contains the required subdirectories (e.g., `dbd5`, `dockground`) to avoid file-not-found errors.
## Key Files Description
* **`config.py`**: Stores all experimental configurations including dataset paths, hyperparameters, and feature toggles.
* **`utils.py`**: Provides helper functions for calculating performance metrics (e.g., AUROC, F1) and setting random seeds for reproducibility.
* **`run_plenet.py`**: The main execution script used to initialize the model and start the training or testing process.
