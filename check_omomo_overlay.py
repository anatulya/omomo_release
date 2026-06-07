# check_omomo_overlay.py

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


REF_ROOT_POS = slice(0, 3)
REF_ROOT_ROT = slice(3, 7)
REF_BODY_POS = slice(162, 318)  # 52 * 3
REF_OBJ_POS = slice(318, 321)
REF_OBJ_ROT = slice(321, 325)


JOINT24_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_middle",
    "right_middle",
]



OMOMO24_TO_PT52 = np.array([
    0,   # pelvis
    1,   # left_hip
    5,   # right_hip
    9,   # spine1
    2,   # left_knee
    6,   # right_knee
    10,  # spine2
    3,   # left_ankle
    7,   # right_ankle
    11,  # spine3
    4,   # left_foot
    8,   # right_foot
    12,  # neck
    14,  # left_collar
    33,  # right_collar
    13,  # head
    15,  # left_shoulder
    34,  # right_shoulder
    16,  # left_elbow
    35,  # right_elbow
    17,  # left_wrist
    36,  # right_wrist
    21,  # left_middle
    40,  # right_middle
], dtype=np.int64)

# SMPL body tree for the first 22 joints.
BODY22_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]

# OMOMO's returned joints24 = first 22 body joints + l/r middle-finger proxies.
JOINT24_EDGES = BODY22_EDGES + [
    (20, 22),
    (21, 23),
]

# Approximate SMPL-X/SMPL-H 52-joint hand chains.
# First 22 are body. 22:37 left hand, 37:52 right hand in many BodyModel conventions.
# The exact finger semantic labels are less important here; this is just for visualization.
LEFT_HAND_CHAINS = [
    [20, 22, 23, 24],
    [20, 25, 26, 27],
    [20, 28, 29, 30],
    [20, 31, 32, 33],
    [20, 34, 35, 36],
]

RIGHT_HAND_CHAINS = [
    [21, 37, 38, 39],
    [21, 40, 41, 42],
    [21, 43, 44, 45],
    [21, 46, 47, 48],
    [21, 49, 50, 51],
]


def chains_to_edges(chains):
    edges = []
    for chain in chains:
        for a, b in zip(chain[:-1], chain[1:]):
            edges.append((a, b))
    return edges


BODY52_EDGES = BODY22_EDGES + chains_to_edges(LEFT_HAND_CHAINS) + chains_to_edges(RIGHT_HAND_CHAINS)


def load_pt_array(path):
    x = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    raise TypeError(f"Expected raw tensor .pt, got {type(x)}")


def get_pt_body52(raw):
    return raw[:, REF_BODY_POS].reshape(raw.shape[0], 52, 3).astype(np.float32)


def get_pt_joints24(raw):
    body52 = raw[:, REF_BODY_POS].reshape(raw.shape[0], 52, 3).astype(np.float32)
    return body52[:, OMOMO24_TO_PT52, :]


def draw_edges(ax, points, edges, linewidth=1.5, alpha=0.8, linestyle="-", label=None):
    first = True
    for a, b in edges:
        if a >= len(points) or b >= len(points):
            continue
        p = points[[a, b]]
        ax.plot(
            p[:, 0], p[:, 1], p[:, 2],
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            label=label if first and label is not None else None,
        )
        first = False


def print_nearest_pt_joint_mapping(omomo_j24, pt_body52):
    T = min(len(omomo_j24), len(pt_body52))
    A = omomo_j24[:T]
    B = pt_body52[:T]

    # Remove each frame's root translation so global offsets do not dominate.
    A_rel = A - A[:, 0:1, :]
    B_rel = B - B[:, 0:1, :]

    print("\n--- nearest .pt body52 joint for each OMOMO replay joint24 ---")
    print("Using per-frame root-relative mean distance.")
    for j in range(A_rel.shape[1]):
        # [T, 52]
        d = np.linalg.norm(A_rel[:, j:j + 1, :] - B_rel, axis=-1)
        mean_d = d.mean(axis=0)
        best = int(mean_d.argmin())

        name = JOINT24_NAMES[j] if j < len(JOINT24_NAMES) else f"joint_{j}"
        print(f"{j:02d} {name:16s} -> pt_body52[{best:02d}]  mean_err={mean_d[best]:.4f}")


def plot_frame(
    out_path,
    omomo_j24,
    pt_j24,
    pt_body52,
    omomo_obj=None,
    pt_obj=None,
    frame=0,
    labels=True,
    links=True,
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    A = omomo_j24[frame]   # [24, 3]
    B = pt_j24[frame]      # [24, 3]
    C = pt_body52[frame]   # [52, 3]

    # Links first, so points appear on top.
    if links:
        draw_edges(
            ax,
            C,
            BODY52_EDGES,
            linewidth=1.0,
            alpha=0.25,
            linestyle="-",
            label=".pt body52 links",
        )
        draw_edges(
            ax,
            B,
            JOINT24_EDGES,
            linewidth=1.3,
            alpha=0.55,
            linestyle="--",
            label=".pt guessed joints24 links",
        )
        draw_edges(
            ax,
            A,
            JOINT24_EDGES,
            linewidth=2.0,
            alpha=0.9,
            linestyle="-",
            label="OMOMO joints24 links",
        )

    # Plot all 52 .pt joints faintly.
    ax.scatter(
        C[:, 0], C[:, 1], C[:, 2],
        s=12,
        alpha=0.25,
        marker=".",
        label="InterMimic/raw .pt all body52",
    )

    # Plot guessed 24-joint subset.
    ax.scatter(
        B[:, 0], B[:, 1], B[:, 2],
        s=28,
        marker="x",
        label="InterMimic/raw .pt guessed joints24",
    )

    # Plot OMOMO replay joints.
    ax.scatter(
        A[:, 0], A[:, 1], A[:, 2],
        s=42,
        label="OMOMO replay joints24",
    )

    if labels:
        # Label OMOMO joints with indices.
        for j, p in enumerate(A):
            ax.text(p[0], p[1], p[2], str(j), fontsize=7)

        # Label pt body52 joints with smaller indices.
        for j, p in enumerate(C):
            ax.text(p[0], p[1], p[2], f"p{j}", fontsize=5, alpha=0.45)

    if omomo_obj is not None:
        obj = omomo_obj[frame]
        if obj.ndim == 2:
            # Downsample mesh verts for plotting.
            stride = max(1, len(obj) // 1000)
            obj_ds = obj[::stride]
            ax.scatter(
                obj_ds[:, 0], obj_ds[:, 1], obj_ds[:, 2],
                s=1,
                alpha=0.12,
                label="OMOMO obj verts",
            )
        else:
            ax.scatter([obj[0]], [obj[1]], [obj[2]], s=60, label="OMOMO obj")

    if pt_obj is not None:
        obj = pt_obj[frame]
        ax.scatter(
            [obj[0]], [obj[1]], [obj[2]],
            s=90,
            marker="*",
            label=".pt obj pos",
        )

    both = np.concatenate(
        [
            A.reshape(-1, 3),
            B.reshape(-1, 3),
            C.reshape(-1, 3),
        ],
        axis=0,
    )
    center = both.mean(axis=0)
    radius = np.linalg.norm(both.max(axis=0) - both.min(axis=0)) * 0.55
    radius = max(radius, 0.5)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"frame {frame}")

    ax.legend(loc="upper right")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print("saved plot:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omomo_npz", required=True, help="OMOMO replay .npz")
    parser.add_argument("--pt", required=True, help="matching InterMimic/OMOMO_new .pt")
    parser.add_argument("--out_dir", default="overlay_debug")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 30, 60, 90])
    parser.add_argument(
        "--align_root0",
        action="store_true",
        help="Subtract each sequence's frame-0 root before plotting",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Disable joint index text labels",
    )
    parser.add_argument(
        "--no_links",
        action="store_true",
        help="Disable skeleton links",
    )
    args = parser.parse_args()

    omomo = np.load(args.omomo_npz, allow_pickle=True)
    raw = load_pt_array(args.pt)

    omomo_j24 = omomo["human_jnts"].astype(np.float32)
    pt_body52 = get_pt_body52(raw)
    pt_j24 = get_pt_joints24(raw)

    T = min(len(omomo_j24), len(pt_j24), len(pt_body52))
    omomo_j24 = omomo_j24[:T]
    pt_j24 = pt_j24[:T]
    pt_body52 = pt_body52[:T]

    pt_obj = raw[:T, REF_OBJ_POS].astype(np.float32)

    omomo_obj = None
    if "obj_verts" in omomo:
        omomo_obj = omomo["obj_verts"][:T].astype(np.float32)

    print("T compare:", T)
    print("omomo_j24:", omomo_j24.shape)
    print("pt_j24 guessed:", pt_j24.shape)
    print("pt_body52:", pt_body52.shape)

    # Direct error for current guessed mapping.
    direct_err = np.linalg.norm(omomo_j24 - pt_j24, axis=-1)
    print("\n--- direct joint error using guessed pt_j24 mapping ---")
    print("mean:", direct_err.mean())
    print("max:", direct_err.max())
    print("root mean:", np.linalg.norm(omomo_j24[:, 0] - pt_j24[:, 0], axis=-1).mean())
    print("left hand proxy mean:", np.linalg.norm(omomo_j24[:, 22] - pt_j24[:, 22], axis=-1).mean())
    print("right hand proxy mean:", np.linalg.norm(omomo_j24[:, 23] - pt_j24[:, 23], axis=-1).mean())

    # Root-relative shape error for current guessed mapping.
    omomo_rel = omomo_j24 - omomo_j24[:, 0:1]
    pt_rel = pt_j24 - pt_j24[:, 0:1]
    rel_err = np.linalg.norm(omomo_rel - pt_rel, axis=-1)

    print("\n--- per-frame root-relative joint error using guessed pt_j24 mapping ---")
    print("mean:", rel_err.mean())
    print("max:", rel_err.max())
    print("left hand proxy mean:", np.linalg.norm(omomo_rel[:, 22] - pt_rel[:, 22], axis=-1).mean())
    print("right hand proxy mean:", np.linalg.norm(omomo_rel[:, 23] - pt_rel[:, 23], axis=-1).mean())

    # Useful for discovering the real mapping.
    print_nearest_pt_joint_mapping(omomo_j24, pt_body52)

    # Optional frame-0 root alignment for plots.
    plot_omomo = omomo_j24.copy()
    plot_pt24 = pt_j24.copy()
    plot_pt52 = pt_body52.copy()
    plot_omomo_obj = omomo_obj.copy() if omomo_obj is not None else None
    plot_pt_obj = pt_obj.copy()

    if args.align_root0:
        o0 = plot_omomo[0, 0].copy()
        p0 = plot_pt52[0, 0].copy()

        plot_omomo -= o0[None, None, :]
        plot_pt24 -= p0[None, None, :]
        plot_pt52 -= p0[None, None, :]

        if plot_omomo_obj is not None:
            plot_omomo_obj -= o0[None, None, :]
        plot_pt_obj -= p0[None, :]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in args.frames:
        if f < T:
            plot_frame(
                out_dir / f"overlay_frame_{f:04d}.png",
                plot_omomo,
                plot_pt24,
                plot_pt52,
                omomo_obj=plot_omomo_obj,
                pt_obj=plot_pt_obj,
                frame=f,
                labels=not args.no_labels,
                links=not args.no_links,
            )


if __name__ == "__main__":
    main()