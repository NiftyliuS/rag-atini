import numpy as np
import umap
import matplotlib.pyplot as plt


def peak_velocity_chart(velocity, peaks):
    plt.figure(figsize=(16, 5))
    plt.plot(velocity, color="black", linewidth=1)
    valid = peaks[peaks < len(velocity)]
    plt.plot(valid, velocity[valid], "x", color="red", markersize=10, markeredgewidth=2)
    plt.title("RagAtini Semantic Velocity (Peaks)")
    plt.xlabel("Token Index")
    plt.ylabel("Velocity")
    plt.xlim(0, len(velocity))
    plt.ylim(0, velocity.max() * 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def umap_chart(vectors, peaks=None):
    centered = vectors - vectors.mean(axis=0)
    coords = umap.UMAP(n_components=3, metric="cosine").fit_transform(centered)

    clusters = np.zeros(len(coords), dtype=int)
    if peaks is not None:
        for fid, p in enumerate(peaks[peaks < len(coords)]):
            clusters[p:] = fid + 1

    plt.figure(figsize=(14, 10))
    plt.plot(coords[:, 0], coords[:, 1], color="gray", lw=0.5, alpha=0.3)
    plt.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap="tab10", s=8, alpha=0.6)

    if peaks is not None:
        valid = peaks[peaks < len(coords)]
        plt.scatter(coords[valid, 0], coords[valid, 1], c="red", marker="x", s=120, zorder=5)

    plt.title("Semantic Trajectory (frames + peaks)")
    plt.tight_layout()
    plt.show()