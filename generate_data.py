import numpy as np
import matplotlib.pyplot as plt

def generate_pla_data(n=100, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, 2)  # 2D data
    Y = np.ones(n)
    
    # Generate a linear separator (e.g., a line y = x)
    # Points above the line will be class 1, below the line class -1
    Y[X[:, 1] < X[:, 0]] = -1  # y < x -> label -1, else label 1

    # Return data in a dictionary for saving
    return {'X': X, 'Y': Y}

def generate_pla2_data(n=100, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, 2)  # 2D data
    Y = np.ones(n)
    
    # Create a non-linearly separable dataset (circle pattern)
    # Points inside a circle will be class 1, outside class -1
    radius = 1
    distance_from_origin = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    Y[distance_from_origin > radius] = -1  # points outside the circle get label -1

    # Return data in a dictionary for saving
    return {'X': X, 'Y': Y}

# Save datasets as .npz files
pla_data = generate_pla_data()
np.savez('pla.npz', X=pla_data['X'], Y=pla_data['Y'])

pla2_data = generate_pla2_data()
np.savez('pla2.npz', X=pla2_data['X'], Y=pla2_data['Y'])

# Optionally, plot to visualize the generated datasets
def plot_data(X, Y):
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='blue', label='Class 1')
    plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], color='red', label='Class -1')
    plt.legend()
    plt.show()

# Visualize pla.npz
plot_data(pla_data['X'], pla_data['Y'])

# Visualize pla2.npz
plot_data(pla2_data['X'], pla2_data['Y'])

