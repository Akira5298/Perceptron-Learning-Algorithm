#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
from visualise import visualise

def plot_3d(X, Y):
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    ax1.scatter(X[:, 0], X[:, 1], Y, c='blue', label='train')
    ax1.scatter(X[:, 0], X[:, 1], Y, c='green', label='test')
    ax1.set_xlabel(f'$X_{0}$')
    ax1.set_ylabel(f'$X_{1}$')
    ax1.set_zlabel('$Y$')
    ax1.legend()
    ax2.scatter(X[:, 0], X[:, 2], Y, c='blue', label='train')
    ax2.scatter(X[:, 0], X[:, 2], Y, c='green', label='test')
    ax2.legend()
    ax2.set_xlabel(f'$X_{0}$')
    ax2.set_ylabel(f'$X_{2}$')
    ax2.set_zlabel('$Y$')
    return fig, (ax1, ax2)

def pla(
    data,
    plot_every: Optional[int] = 1,
    max_iteration: int = 100
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, list]:
#////////////////////////////////////////////////////////////////////from30to36
    X = data['X']
    Y = data['Y'].flatten()
    N, d = X.shape
    X_bias = np.c_[X, np.ones(N)]  # add bias term
    w = np.zeros(d + 1)
#////////////////////////////////////////////////////////////////////
    if plot_every:
        fig, ax = plt.subplots()
        artists = []

    for i in range(max_iteration):
#////////////////////////////////////////////////////////////////////from42to51
        predictions = np.sign(X_bias @ w)
        misclassified = np.where(predictions != Y)[0]
        if len(misclassified) == 0:
            print(f"Converged after {i} iterations.")
            break
        idx = misclassified[0]
        w += Y[idx] * X_bias[idx]
        acc = np.mean(predictions == Y)
#////////////////////////////////////////////////////////////////////

        if plot_every and i % plot_every == 0:
            _fig, _ax, artists_ = visualise(
                w,
                X,
                Y,
                ax=ax,
                animated=True,
                title=f'Training, accuracy: {acc:0.3f} (iter {i})')
            artists.append(artists_)

    if plot_every:
        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000)
        return w, (ani, fig)
    else:
        return w

def main():
#////////////////////////////////////////////////////////////////////from70to73
    data = np.load('pla.npz')
    w, (ani, fig) = pla(data, plot_every=1, max_iteration=100)
#////////////////////////////////////////////////////////////////////
    plt.show()

if __name__ == "__main__":
    main()

