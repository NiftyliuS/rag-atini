import numpy as np
import umap
import matplotlib.pyplot as plt

def peak_velocity_chart(velocity, peaks):
    plt.figure(figsize=(16, 5))
    plt.plot(velocity, color="black", linewidth=1)
    valid = peaks[peaks < len(velocity)]
    plt.scatter(valid, velocity[valid], facecolors="none", edgecolors="red",
                marker="o", s=120, linewidths=2, zorder=5)
    plt.title("RagAtini Semantic Velocity (Peaks)")
    plt.xlabel("Token Index")
    plt.ylabel("Velocity")
    plt.xlim(0, len(velocity))
    plt.ylim(0, velocity.max() * 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def umap_chart_2d(vectors, peaks=None):
    print("This is going to take a while so get some coffee...")
    centered = vectors - vectors.mean(axis=0)
    coords = umap.UMAP(n_components=2, metric="cosine", random_state=42).fit_transform(centered)

    plt.figure(figsize=(14, 10))
    plt.plot(coords[:, 0], coords[:, 1], color="gray", lw=0.5, alpha=0.3)

    if peaks is not None:
        colors = np.zeros(len(coords), dtype=int)
        for fid, p in enumerate(peaks[peaks < len(coords)]):
            colors[p:] = fid + 1
        cmap = "tab20"
    else:
        colors = np.arange(len(coords))
        cmap = "viridis"

    sc = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap=cmap, s=8, alpha=0.6)

    if peaks is not None:
        valid = peaks[peaks < len(coords)]
        plt.scatter(coords[valid, 0], coords[valid, 1], facecolors="none",
                    edgecolors="red", marker="o", s=120, linewidths=2, zorder=5)

    plt.colorbar(sc, label="Frame" if peaks is not None else "Token Index")
    plt.title("Semantic Trajectory (frames + peaks)")
    plt.tight_layout()
    plt.show()