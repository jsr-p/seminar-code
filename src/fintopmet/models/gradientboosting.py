"""
Inspiration taken from:
    https://blog.mattbowers.dev/gradient-boosting-machine-from-scratch
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

import fintopmet

# --------------------- Data --------------------- #


plt.style.use("default")


def generate_data():
    rng = np.random.default_rng()
    x = np.linspace(0, 10, 50)
    y = np.where(x < 5, x, 5) + rng.normal(0, 0.4, size=x.shape)
    x = x.reshape(-1, 1)
    return x, y


# --------------------- Iterative gradient boosting --------------------- #


def multiple():
    # hyperparameters
    learning_rate = 1
    n_trees = 8
    max_depth = 1

    x, y = generate_data()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    ax.scatter(x, y)
    ax.set(
        xlabel="x",
        ylabel="y",
        title=(
            "Data generated as:\n"
            r"$y = xI(x < 5) + 5 I(x \geq 5) + \epsilon, \epsilon \sim N(0, 1)$"
        ),
    )
    fig.savefig(fname=str(fintopmet.fp.FIGS / "gbtut" / "data_gb.png"))

    # Training
    F0 = y.mean()

    m = 0
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    fig.suptitle("m = {}".format(m))

    ax1.plot(x, y, "o", label=r"$y$")
    ax1.axhline(F0, color="r", linestyle="--", label=r"$F_0 := \bar{y}$")
    ax1.set_ylabel("y")
    ax1.set_xlabel("x")
    ax1.legend()
    ax1.set_ylim([-1, 7])

    ax2.plot(x, y, "o", label=r"$y$")
    ax2.axhline(F0, color="r", linestyle="--", label=r"$F_0 := \bar{y}$")
    resids = y - F0
    negative = resids < 0
    ax2.vlines(x[negative], ymin=y[negative], ymax=F0, linewidth=0.5)
    ax2.vlines(x[~negative], ymin=F0, ymax=y[~negative], linewidth=0.5)
    ax2.set_ylim([-1, 7])
    ax2.legend()
    ax2.set_ylabel("y")
    ax2.set_xlabel("x")

    ax3.plot(x, y - F0, "d", label=r"$y - F_{}(x)$".format(0))
    ax3.vlines(x, 0, y - F0, linewidth=0.5)
    ax3.legend()
    ax3.set_ylim([-6, 6])
    ax3.set_ylabel("residuals")
    ax3.set_xlabel("x")

    fig.savefig(fname=str(fintopmet.fp.FIGS / "gbtut" / "gbm_iterative_0.png"))

    Fm = F0
    trees = []
    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(x, y - Fm)
        preds = tree.predict(x)

        m = i + 1

        if m in [1, 2]:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
            plot_tree(tree, ax=ax)
            fig.tight_layout()
            fig.savefig(fname=str(fintopmet.fp.FIGS / "gbtut" / f"dtree{m}.png"))

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        ax1.plot(x, y - Fm, "o", label=r"$y - F_{}(x)$".format(m - 1))
        ax1.plot(x, preds, "--r", label=r"$h_{}(x)$".format(m))
        ax1.legend()
        ax1.set_ylim([-6, 6])
        ax1.set_title("m = {}".format(m))
        ax1.set_ylabel("residuals")

        Fm += learning_rate * tree.predict(x)
        trees.append(tree)

        ax2.plot(x, y, "o", label=r"$y$")
        ax2.plot(x, Fm, "-r", label=r"$F_{}(x)$".format(m))
        ax2.legend()
        ax2.set_ylim([-1, 7])
        ax2.set_title(f"m = {m}; $F_{{{m}}}(x) = F_{{{m-1}}}(x) + h_{{{m}}}(x)$")
        ax2.set_ylabel("y")

        ax1.set_xlabel("x")
        ax2.set_xlabel("x")

        fig.tight_layout()
        fig.savefig(fname=str(fintopmet.fp.FIGS / "gbtut" / f"gbm_iterative_{i+1}.png"))
        # plt.show()


if __name__ == "__main__":
    multiple()
