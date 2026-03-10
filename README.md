# A Cost-Sensitive Transformer Model for Prognostics Under Highly Imbalanced Industrial Data

This repository contains the implementation code for the paper:

**"A Cost-Sensitive Transformer Model for Prognostics Under Highly Imbalanced Industrial Data"**  
*Published in Cluster Computing (Springer)*

## Overview

This work proposes a cost-sensitive Transformer-based approach for predictive maintenance using the APS (Air Pressure System) Failure dataset from Scania trucks. The model addresses the severe class imbalance problem and incorporates asymmetric misclassification costs through Binary Focal Loss optimization.

### Key Features

- **Transformer Architecture**: Multi-head self-attention mechanism for capturing complex patterns in sensor data
- **Cost-Sensitive Learning**: Binary Focal Loss with configurable α and γ parameters
- **Data Preprocessing Pipeline**: Iterative imputation (BayesianRidge), SVMSMOTE resampling, and MinMax scaling
- **Custom Cost Metric**: Asymmetric cost function (FN: $500, FP: $10) aligned with industry standards

## Dataset

The APS Failure dataset from Scania trucks is used, which is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks).

- **Training set**: 60,000 samples (59,000 negative, 1,000 positive)
- **Test set**: 16,000 samples (15,625 negative, 375 positive)
- **Features**: 170 anonymized sensor attributes

## Requirements

```
tensorflow>=2.x
keras
scikit-learn
imbalanced-learn
pandas
numpy
scipy
xgboost
lightgbm
focal-loss
scikitplot
seaborn
matplotlib
plotly
transformers
```

Install dependencies:
```bash
pip install tensorflow scikit-learn imbalanced-learn pandas numpy scipy xgboost lightgbm focal-loss scikit-plot seaborn matplotlib plotly transformers
```

## Repository Structure

```
├── DataPreprocessingCode.py     # Data preprocessing pipeline
├── TrainingCode.py              # Transformer model training script
├── script example.txt           # Example commands for running the code
├── aps_failure_training_set.csv # Training dataset
├── aps_failure_test_set.csv     # Test dataset
├── CSV/                         # Training logs (metrics per epoch)
└── Model/                       # Saved trained models
```

## Usage

### Step 1: Data Preprocessing

Run the preprocessing script to handle missing values, perform imputation, and apply resampling:

```bash
python DataPreprocessingCode.py
```

This will generate the following intermediate files:
- `aps_failure_train_valid_set_cut_imputed.csv`
- `aps_failure_train_valid_set_cut_imputed_scaled.csv`
- `aps_failure_train_valid_set_cut_imputed_resampled_scaled.csv`
- `aps_failure_test_set_cut_imputed_scaled.csv`
- Label files for train/test sets

### Step 2: Model Training

Train the Transformer model with customizable hyperparameters:

```bash
python TrainingCode.py --seed_number 0 \
                       --learning_rate 0.0005 \
                       --pos_weight 0.95 \
                       --gamma 1.5 \
                       --num_epochs 8000 \
                       --batch_size 72 \
                       --head_size 256 \
                       --num_heads 4 \
                       --ff_dim 4 \
                       --num_transformer_blocks 4 \
                       --mlp_units 128_64 \
                       --mlp_dropout 0.4 \
                       --dropout 0.25
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed_number` | 0 | Random seed for reproducibility |
| `--learning_rate` | 0.0005 | Learning rate for Adam optimizer |
| `--pos_weight` | 0.95 | α parameter for Binary Focal Loss |
| `--gamma` | 1.5 | γ parameter for Binary Focal Loss |
| `--num_epochs` | 8000 | Number of training epochs |
| `--batch_size` | 72 | Training batch size |
| `--head_size` | 256 | Attention head dimension |
| `--num_heads` | 4 | Number of attention heads |
| `--ff_dim` | 4 | Feed-forward dimension multiplier |
| `--num_transformer_blocks` | 4 | Number of Transformer encoder blocks |
| `--mlp_units` | 128_64 | MLP hidden layer sizes (underscore-separated) |
| `--mlp_dropout` | 0.4 | Dropout rate for MLP layers |
| `--dropout` | 0.25 | Dropout rate for attention layers |

## Model Architecture

The Transformer model consists of:
1. **Input Layer**: Accepts preprocessed sensor features
2. **Transformer Encoder Blocks**: Multiple blocks with:
   - Layer Normalization
   - Multi-Head Self-Attention
   - Residual Connections
   - Feed-Forward Network (Conv1D layers)
3. **Global Average Pooling**
4. **MLP Classification Head**: Dense layers with dropout
5. **Sigmoid Output**: Binary classification

## Cost-Sensitive Metric

The model uses an asymmetric cost function following the Scania competition guidelines:

```
Total Cost = 500 × FN + 10 × FP
```

Where:
- **FN (False Negative)**: Missed failure prediction — $500 cost (breakdown, service interruption)
- **FP (False Positive)**: Unnecessary maintenance — $10 cost (inspection cost)

## Output Files

After training, the following outputs are generated:

- **Model checkpoints**: `Model/<model_name>/` — Saved Keras models
- **Training logs**: `CSV/<model_name>.csv` — Epoch-wise metrics (loss, cost, etc.)
- **TensorBoard logs**: `TB/<model_name>/` — Visualization logs

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Cost-SensitiveTransformerModel2026,
  title={A Cost-Sensitive Transformer Model for Prognostics Under Highly Imbalanced Industrial Data},
  author={Beikmohammadi, Ali and Hamian, Mohammad Hosein and Khoeyniha, Neda and Lindgren, Tony and Steinert, Olof and Magn{\'u}sson, Sindri},
  journal={Cluster Computing},
  year={2026},
  publisher={Springer}
}
```

## License

This project is provided for academic and research purposes.

## Contact

For questions or issues regarding this code, please contact the authors.
