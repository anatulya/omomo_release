"""
Quick visualization of exported object surface points (.npz from export_object_surface_points.py).

Usage:
    python visualize_surface_points.py data/mesh_points/largebox_surface_points_1024.npz
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_surface_points.py <path_to_npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    data = np.load(npz_path)
    points = data["points_body_local"]  # [N, 3]

    print(f"Loaded {points.shape[0]} points from {npz_path}")
    print(f"Bounds: min={points.min(axis=0).round(4)}  max={points.max(axis=0).round(4)}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, alpha=0.7, c=points[:, 2], cmap="viridis")

    # Equal aspect ratio so the box doesn't look distorted
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{npz_path}\n{points.shape[0]} surface points")

    plt.tight_layout()

    out_path = npz_path.rsplit(".", 1)[0] + "_viz.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()