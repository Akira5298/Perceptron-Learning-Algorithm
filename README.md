# Perceptron Learning Algorithm (PLA)

A simple implementation of the **Perceptron Learning Algorithm** in Python. The project shows how a binary classifier learns a linear decision boundary, with animations made using Matplotlib.

## Features

- Implements PLA using NumPy
- Generates example datasets for binary classification
- Visualizes data points and the decision boundary during training
- Explores the difference between linearly and nonlinearly separable data

## Files

| File | Purpose |
| --- | --- |
| `pla_main.py` | Runs the perceptron algorithm and training animation |
| `generate_data.py` | Generates sample datasets (`pla.npz` and `pla2.npz`) |
| `visualise.py` | Plots the data and decision boundary |

## How to run

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate the datasets:

```bash
python3 generate_data.py
```

Run the algorithm:

```bash
python3 pla_main.py
```

**Note:** Rename `pla_main_final.py` to `pla_main.py` if needed. The current main script uses `pla.npz` by default.

## How it works

The perceptron predicts one of two classes (`+1` or `-1`). When it misclassifies a point, it updates its weights and tries again. This process continues until all points are classified correctly or the iteration limit is reached.

PLA can converge for linearly separable data, but it cannot perfectly separate circular patterns using a straight line.

## Tech stack

Python · NumPy · Matplotlib
