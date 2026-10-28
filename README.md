# Multilayer Perceptron

A small multilayer perceptron (MLP) binary classifier implemented from scratch
with NumPy. The project is designed for datasets with the Wisconsin Diagnostic
Breast Cancer CSV layout: an identifier, an `M`/`B` diagnosis label, and
numeric features.

The network supports configurable hidden layers, activation functions,
initializers, loss functions, mini-batch gradient descent, early stopping, and
model serialization.

## Requirements

- Python 3
- NumPy
- Matplotlib (used by `Train.py` to save learning curves)

Install the Python dependencies with:

```bash
python3 -m pip install numpy matplotlib
```

## Project files

| File | Purpose |
| --- | --- |
| `Split.py` | Creates stratified training and validation CSV files. |
| `Train.py` | Defines the MLP and trains it from CSV data. |
| `Predict.py` | Loads a saved model and evaluates it on a CSV dataset. |

## Dataset format

Each data row must contain:

```text
id,diagnosis,feature_1,feature_2,...
```

The diagnosis must be `M` (malignant) or `B` (benign). A header row is
optional. `Train.py` uses every column after the diagnosis as a numeric input
feature.

## Usage

### 1. Split a dataset

Create an 80/20 stratified split using the default random seed:

```bash
python3 Split.py --dataset data.csv
```

This writes:

- `data_training.csv`
- `data_validation.csv`

To choose a different training fraction and seed:

```bash
python3 Split.py --dataset data.csv --ratio 0.75 --seed 42
```

`--ratio` is the fraction assigned to training for each class. The remaining
rows are assigned to validation.

### 2. Train the model

With the default settings:

```bash
python3 Train.py
```

The defaults are:

- two hidden layers of 24 neurons each
- 70 epochs
- mini-batches of 8
- learning rate `0.01`
- sigmoid hidden-layer activation
- He uniform initialization
- categorical cross-entropy

Example with custom settings:

```bash
python3 Train.py \
  --train data_training.csv \
  --valid data_validation.csv \
  --layer 32 16 \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --activation relu \
  --init heUniform \
  --loss categoricalCrossentropy \
  --patience 10 \
  --seed 42 \
  --save saved_model.npy
```

Training standardizes features using statistics calculated from the training
set only. It prints training and validation loss/accuracy for each epoch and
creates:

- `saved_model.npy`: learned weights, biases, architecture, and scaling
  statistics
- `learning_curves.png`: loss and accuracy curves

Use `--patience N` to stop after `N` consecutive epochs without validation-loss
improvement. Early stopping is disabled when `--patience 0` (the default).

### 3. Evaluate a saved model

```bash
python3 Predict.py \
  --model saved_model.npy \
  --dataset data_validation.csv
```

The evaluator applies the same training-set standardization saved in the model
and reports:

- number of samples
- binary cross-entropy
- accuracy
- confusion-matrix counts (`TP`, `TN`, `FP`, and `FN`)

The positive class is `M` (malignant).

## Implementation notes

- The output layer uses softmax for the two classes.
- Hidden layers can use `sigmoid`, `tanh`, or `relu`.
- Supported initializers are `heUniform` and `xavierUniform`.
- Supported losses are `categoricalCrossentropy` and `binaryCrossentropy`.
- Backpropagation and gradient-descent updates are implemented directly in
  `Train.py`; no deep-learning framework is required.

## Case-sensitive filenames

The source files in this repository use capitalized names (`Train.py`,
`Predict.py`, and `Split.py`). Python imports are case-sensitive on Linux.
`Predict.py` currently imports a lowercase module named `train`; if prediction
fails with `ModuleNotFoundError: No module named 'train'`, rename
`Train.py` to `train.py` or update that import to match the filename before
running the prediction command.
