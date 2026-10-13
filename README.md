# Perceptron Learning Algorithm (PLA) — From Scratch

A Python implementation of the **Perceptron Learning Algorithm (PLA)** for binary classification, with synthetic dataset generation and Matplotlib visualizations of the evolving linear decision boundary.

This is a learning-focused project exploring how a simple linear classifier adjusts its weights in response to misclassified examples and what happens when the data is not linearly separable.

## Features

- Implements the perceptron weight update rule using NumPy, without a machine-learning training library.
- Uses two-dimensional features and a bias term to learn a straight-line decision boundary.
- Updates weights on the **first misclassified sample** at each iteration, stopping when all samples are classified correctly or the iteration limit is reached.
- Visualizes positively and negatively labeled points alongside the decision boundary, with an animated training sequence.
- Generates reproducible synthetic datasets with both linear and circular class boundaries.

## Project structure

```text
perceptron-learning-algorithm/
├── pla_main.py         # PLA training and animation (rename pla_main_final.py)
├── visualise.py        # 2D data and decision-boundary plotting
├── generate_data.py    # Creates pla.npz and pla2.npz
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes generated/local files
└── README.md
```

> **Note:** The original main script is named `pla_main_final.py`. Rename it to `pla_main.py` when preparing the repository; these instructions assume the renamed file. The `plot_3d()` helper in that script is unused and assumes three feature columns, whereas these datasets contain only two; consider removing it during cleanup.

## Requirements

- Python 3.10 or newer (the main script uses modern union type annotations)
- NumPy
- Matplotlib
- A graphical environment capable of showing Matplotlib windows for the interactive plots

## Getting started

1. Clone or download the repository, then open a terminal in its folder.
2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Generate the sample datasets:

   ```bash
   python generate_data.py
   ```

   This creates `pla.npz` (linearly separable: classes split around `y = x`) and `pla2.npz` (circular pattern: points inside versus outside a radius of 1). The generator also opens scatter plots for both datasets; close the plot windows to finish.

5. Run the perceptron visualization:

   ```bash
   python pla_main.py
   ```

   The current main function loads **`pla.npz`** and runs up to 100 weight-update iterations, visualizing the training process.

## How PLA works

For a feature vector **x**, the classifier predicts a class using the sign of the weighted sum:

```text
prediction = sign(w · x_b)
```

Here, `x_b` is the feature vector with a constant bias feature appended. When a sample with label `y ∈ {-1, +1}` is misclassified, the weights are updated:

```text
w <- w + y * x_b
```

If the training samples are linearly separable, the perceptron can converge to a separator. For nonlinearly separable data, convergence to zero training error is not guaranteed.

## Experiments and limitations

- **`pla.npz`:** A dataset whose classes are separated by a straight line, suitable for studying perceptron convergence.
- **`pla2.npz`:** A circular-label dataset that is **not linearly separable** in the original two-dimensional feature space. It demonstrates a key limitation of a single linear perceptron.
- **Current script limitation:** `pla_main.py` is hard-coded to load `pla.npz`. To experiment with `pla2.npz`, change `np.load('pla.npz')` in its `main()` function to `np.load('pla2.npz')`.
- **Accuracy display:** The supplied implementation computes the accuracy shown in each frame **before** updating weights, so the displayed metric may not precisely correspond to the boundary in that frame. This is a known improvement to make before treating the visualization as an evaluation result.
- **Evaluation scope:** The current project uses synthetic data and reports training behavior, not held-out test-set performance or generalization metrics.

## Possible improvements

- Recompute accuracy after each weight update for consistent animation labels.
- Add command-line options to choose the dataset and iteration limit.
- Add automated tests for the weight updates and convergence conditions.
- Save the animation as a GIF for easier portfolio sharing.
- Record iteration counts and final training accuracy for both datasets.

## Technologies

**Python**, **NumPy**, **Matplotlib**

## Project context

An educational implementation of a foundational machine-learning algorithm, intended to make the mechanics and limitations of linear classification easier to observe.
