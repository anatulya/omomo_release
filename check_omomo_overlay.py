# check_omomo_overlay.py

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import matplotlib.animation as animation

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

JOINT24_EDGES = BODY22_EDGES + [
    (20, 22),
    (21, 23),
]

LEFT_HAND_CHAINS = [
    [17, 18, 19, 20],
    [17, 21, 22, 23],
    [17, 24, 25, 26],
    [17, 27, 28, 29],
    [17, 30, 31, 32],
]

RIGHT_HAND_CHAINS = [
    [36, 37, 38, 39],
    [36, 40, 41, 42],
    [36, 43, 44, 45],
    [36, 46, 47, 48],
    [36, 49, 50, 51],
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
    body52 = get_pt_body52(raw)
    return body52[:, OMOMO24_TO_PT52, :]

def make_overlay_video(
    out_path,
    omomo_j24,
    pt_j24,
    pt_body52,
    human_verts=None,
    left_vertex_ids=None,
    right_vertex_ids=None,
    left_points=None,
    right_points=None,
    omomo_obj=None,
    pt_obj=None,
    fps=30,
    labels=False,
    links=True,
    mesh_stride=120,
    obj_stride=1000,
):
    T = len(omomo_j24)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_for_bounds = [
        omomo_j24.reshape(-1, 3),
        pt_j24.reshape(-1, 3),
        pt_body52.reshape(-1, 3),
    ]
    if left_points is not None:
        all_for_bounds.append(left_points.reshape(-1, 3))
    if right_points is not None:
        all_for_bounds.append(right_points.reshape(-1, 3))
    if omomo_obj is not None:
        # Use a subset for global bounds to avoid huge memory if object has many verts.
        all_for_bounds.append(omomo_obj[:, ::max(1, omomo_obj.shape[1] // 1000), :].reshape(-1, 3))

    all_pts = np.concatenate(all_for_bounds, axis=0)
    center = all_pts.mean(axis=0)
    radius = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.55
    radius = max(radius, 0.5)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()

        A = omomo_j24[frame]
        B = pt_j24[frame]
        C = pt_body52[frame]

        if human_verts is not None:
            V = human_verts[frame]
            V_ds = V[::max(1, mesh_stride)]
            ax.scatter(
                V_ds[:, 0], V_ds[:, 1], V_ds[:, 2],
                s=1,
                alpha=0.03,
                label="human mesh",
            )

            if left_vertex_ids is not None:
                L = V[left_vertex_ids]
                L_ds = L[::max(1, len(L) // 400)]
                ax.scatter(
                    L_ds[:, 0], L_ds[:, 1], L_ds[:, 2],
                    s=3,
                    alpha=0.30,
                    label="left hand verts",
                )

            if right_vertex_ids is not None:
                R = V[right_vertex_ids]
                R_ds = R[::max(1, len(R) // 400)]
                ax.scatter(
                    R_ds[:, 0], R_ds[:, 1], R_ds[:, 2],
                    s=3,
                    alpha=0.30,
                    label="right hand verts",
                )

        if left_points is not None:
            P = left_points[frame]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=5, alpha=0.55, label="left sampled hand")

        if right_points is not None:
            P = right_points[frame]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=5, alpha=0.55, label="right sampled hand")

        if links:
            draw_edges(ax, C, BODY52_EDGES, linewidth=1.0, alpha=0.20, linestyle="-", label=".pt body52 links")
            draw_edges(ax, B, JOINT24_EDGES, linewidth=1.2, alpha=0.45, linestyle="--", label=".pt mapped joints24 links")
            draw_edges(ax, A, JOINT24_EDGES, linewidth=2.0, alpha=0.9, linestyle="-", label="OMOMO joints24 links")

        ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=10, alpha=0.20, marker=".", label=".pt body52")
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], s=25, marker="x", label=".pt mapped joints24")
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=36, label="OMOMO joints24")

        if labels:
            for j, p in enumerate(A):
                ax.text(p[0], p[1], p[2], str(j), fontsize=7)

        if omomo_obj is not None:
            obj = omomo_obj[frame]
            stride = max(1, len(obj) // obj_stride)
            obj_ds = obj[::stride]
            ax.scatter(
                obj_ds[:, 0], obj_ds[:, 1], obj_ds[:, 2],
                s=1,
                alpha=0.14,
                label="object verts",
            )

        if pt_obj is not None:
            obj = pt_obj[frame]
            ax.scatter([obj[0]], [obj[1]], [obj[2]], s=80, marker="*", label=".pt obj pos")

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"frame {frame}/{T - 1}")
        ax.legend(loc="upper right", fontsize=7)

        return []

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    print(f"saving video: {out_path}")
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"saved video: {out_path}")

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


def load_hand_vertex_ids(seg_path, num_verts):
    with open(seg_path, "r") as f:
        seg = json.load(f)

    print("\n--- segmentation keys ---")
    print(sorted(seg.keys()))

    def collect(keys):
        ids = []
        for k in keys:
            if k in seg:
                ids.extend(seg[k])
        ids = np.asarray(sorted(set(ids)), dtype=np.int64)
        return ids

    left_keys = [
        "leftHand",
        "leftHandIndex1",
        "leftHandIndex2",
        "leftHandIndex3",
        "leftHandMiddle1",
        "leftHandMiddle2",
        "leftHandMiddle3",
        "leftHandPinky1",
        "leftHandPinky2",
        "leftHandPinky3",
        "leftHandRing1",
        "leftHandRing2",
        "leftHandRing3",
        "leftHandThumb1",
        "leftHandThumb2",
        "leftHandThumb3",
    ]

    right_keys = [
        "rightHand",
        "rightHandIndex1",
        "rightHandIndex2",
        "rightHandIndex3",
        "rightHandMiddle1",
        "rightHandMiddle2",
        "rightHandMiddle3",
        "rightHandPinky1",
        "rightHandPinky2",
        "rightHandPinky3",
        "rightHandRing1",
        "rightHandRing2",
        "rightHandRing3",
        "rightHandThumb1",
        "rightHandThumb2",
        "rightHandThumb3",
    ]

    left_ids = collect(left_keys)
    right_ids = collect(right_keys)

    if len(left_ids) == 0:
        raise ValueError("No left-hand vertex IDs found. Check segmentation key names.")
    if len(right_ids) == 0:
        raise ValueError("No right-hand vertex IDs found. Check segmentation key names.")

    max_id = max(int(left_ids.max()), int(right_ids.max()))
    print("\n--- segmentation sanity ---")
    print("num_verts:", num_verts)
    print("left_ids:", left_ids.shape, "min/max:", int(left_ids.min()), int(left_ids.max()))
    print("right_ids:", right_ids.shape, "min/max:", int(right_ids.min()), int(right_ids.max()))
    print("max hand id:", max_id)

    if max_id >= num_verts:
        raise ValueError(
            f"Segmentation max id {max_id} >= human mesh num_verts {num_verts}. "
            "This segmentation does not match the mesh topology."
        )

    return left_ids, right_ids


def build_global_submesh_faces(vertex_ids, full_faces):
    vertex_set = set(vertex_ids.tolist())
    out_faces = []

    for tri in full_faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a in vertex_set and b in vertex_set and c in vertex_set:
            out_faces.append([a, b, c])

    out_faces = np.asarray(out_faces, dtype=np.int64)
    if len(out_faces) == 0:
        raise ValueError("No submesh faces found for this vertex set.")

    return out_faces


def sample_persistent_surface_anchors(verts0, faces, n_samples, rng):
    v0 = verts0[faces[:, 0]]
    v1 = verts0[faces[:, 1]]
    v2 = verts0[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
    total_area = areas.sum()
    if total_area <= 1e-12:
        raise ValueError("Submesh has near-zero total area.")

    probs = areas / total_area
    face_ids = rng.choice(len(faces), size=n_samples, p=probs)

    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)

    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    bary = np.stack([u, v, w], axis=-1).astype(np.float32)
    return face_ids.astype(np.int64), bary


def eval_surface_points(verts, faces, face_ids, bary):
    tri = faces[face_ids]          # [N, 3]
    tri_verts = verts[:, tri, :]   # [T, N, 3, 3]

    points = (
        bary[None, :, 0:1] * tri_verts[:, :, 0, :]
        + bary[None, :, 1:2] * tri_verts[:, :, 1, :]
        + bary[None, :, 2:3] * tri_verts[:, :, 2, :]
    )

    return points.astype(np.float32)


def compute_hand_samples(human_verts, human_faces, seg_path, n_samples, seed):
    num_verts = human_verts.shape[1]
    left_ids, right_ids = load_hand_vertex_ids(seg_path, num_verts)

    left_faces = build_global_submesh_faces(left_ids, human_faces)
    right_faces = build_global_submesh_faces(right_ids, human_faces)

    print("\n--- hand submesh faces ---")
    print("left_faces:", left_faces.shape)
    print("right_faces:", right_faces.shape)

    rng = np.random.default_rng(seed)

    left_face_ids, left_bary = sample_persistent_surface_anchors(
        human_verts[0],
        left_faces,
        n_samples,
        rng,
    )
    right_face_ids, right_bary = sample_persistent_surface_anchors(
        human_verts[0],
        right_faces,
        n_samples,
        rng,
    )

    left_points = eval_surface_points(human_verts, left_faces, left_face_ids, left_bary)
    right_points = eval_surface_points(human_verts, right_faces, right_face_ids, right_bary)

    return {
        "left_ids": left_ids,
        "right_ids": right_ids,
        "left_faces": left_faces,
        "right_faces": right_faces,
        "left_face_ids": left_face_ids,
        "right_face_ids": right_face_ids,
        "left_bary": left_bary,
        "right_bary": right_bary,
        "left_points": left_points,
        "right_points": right_points,
    }


def print_nearest_pt_joint_mapping(omomo_j24, pt_body52):
    T = min(len(omomo_j24), len(pt_body52))
    A = omomo_j24[:T]
    B = pt_body52[:T]

    A_rel = A - A[:, 0:1, :]
    B_rel = B - B[:, 0:1, :]

    print("\n--- nearest .pt body52 joint for each OMOMO replay joint24 ---")
    print("Using per-frame root-relative mean distance.")
    for j in range(A_rel.shape[1]):
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
    human_verts=None,
    left_vertex_ids=None,
    right_vertex_ids=None,
    left_points=None,
    right_points=None,
    omomo_obj=None,
    pt_obj=None,
    frame=0,
    labels=True,
    links=True,
    mesh_stride=80,
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    A = omomo_j24[frame]
    B = pt_j24[frame]
    C = pt_body52[frame]

    if human_verts is not None:
        V = human_verts[frame]
        V_ds = V[::max(1, mesh_stride)]
        ax.scatter(
            V_ds[:, 0], V_ds[:, 1], V_ds[:, 2],
            s=1,
            alpha=0.035,
            label="OMOMO human mesh verts downsampled",
        )

        if left_vertex_ids is not None:
            L = V[left_vertex_ids]
            L_ds = L[::max(1, len(L) // 500)]
            ax.scatter(
                L_ds[:, 0], L_ds[:, 1], L_ds[:, 2],
                s=3,
                alpha=0.35,
                label="left hand mesh verts",
            )

        if right_vertex_ids is not None:
            R = V[right_vertex_ids]
            R_ds = R[::max(1, len(R) // 500)]
            ax.scatter(
                R_ds[:, 0], R_ds[:, 1], R_ds[:, 2],
                s=3,
                alpha=0.35,
                label="right hand mesh verts",
            )

    if left_points is not None:
        P = left_points[frame]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=6, alpha=0.55, label="left sampled hand points")

    if right_points is not None:
        P = right_points[frame]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=6, alpha=0.55, label="right sampled hand points")

    if links:
        draw_edges(ax, C, BODY52_EDGES, linewidth=1.0, alpha=0.25, linestyle="-", label=".pt body52 links")
        draw_edges(ax, B, JOINT24_EDGES, linewidth=1.3, alpha=0.55, linestyle="--", label=".pt joints24 links")
        draw_edges(ax, A, JOINT24_EDGES, linewidth=2.0, alpha=0.9, linestyle="-", label="OMOMO joints24 links")

    ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=12, alpha=0.25, marker=".", label=".pt all body52")
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], s=28, marker="x", label=".pt mapped joints24")
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=42, label="OMOMO replay joints24")

    if labels:
        for j, p in enumerate(A):
            ax.text(p[0], p[1], p[2], str(j), fontsize=7)
        for j, p in enumerate(C):
            ax.text(p[0], p[1], p[2], f"p{j}", fontsize=5, alpha=0.45)

    if omomo_obj is not None:
        obj = omomo_obj[frame]
        if obj.ndim == 2:
            stride = max(1, len(obj) // 1000)
            obj_ds = obj[::stride]
            ax.scatter(obj_ds[:, 0], obj_ds[:, 1], obj_ds[:, 2], s=1, alpha=0.12, label="OMOMO obj verts")
        else:
            ax.scatter([obj[0]], [obj[1]], [obj[2]], s=60, label="OMOMO obj")

    if pt_obj is not None:
        obj = pt_obj[frame]
        ax.scatter([obj[0]], [obj[1]], [obj[2]], s=90, marker="*", label=".pt obj pos")

    all_for_bounds = [A.reshape(-1, 3), B.reshape(-1, 3), C.reshape(-1, 3)]
    if left_points is not None:
        all_for_bounds.append(left_points[frame].reshape(-1, 3))
    if right_points is not None:
        all_for_bounds.append(right_points[frame].reshape(-1, 3))

    both = np.concatenate(all_for_bounds, axis=0)
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
    parser.add_argument("--seg_path", default="data/smplx_vert_segmentation.json")
    parser.add_argument("--out_dir", default="overlay_debug")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 30, 60, 90])
    parser.add_argument("--align_root0", action="store_true")
    parser.add_argument("--no_labels", action="store_true")
    parser.add_argument("--no_links", action="store_true")
    parser.add_argument("--plot_mesh", action="store_true")
    parser.add_argument("--plot_hands", action="store_true")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mesh_stride", type=int, default=80)
    parser.add_argument("--hand_out", type=str, default="")
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--video_out", type=str, default="overlay_debug/overlay.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    omomo = np.load(args.omomo_npz, allow_pickle=True)
    raw = load_pt_array(args.pt)

    omomo_j24 = omomo["human_jnts"].astype(np.float32)
    human_verts = omomo["human_verts"].astype(np.float32)
    human_faces = omomo["human_faces"].astype(np.int64)

    pt_body52 = get_pt_body52(raw)
    pt_j24 = get_pt_joints24(raw)

    T = min(len(omomo_j24), len(pt_j24), len(pt_body52), len(human_verts))
    omomo_j24 = omomo_j24[:T]
    human_verts = human_verts[:T]
    pt_j24 = pt_j24[:T]
    pt_body52 = pt_body52[:T]

    pt_obj = raw[:T, REF_OBJ_POS].astype(np.float32)

    omomo_obj = None
    if "obj_verts" in omomo:
        omomo_obj = omomo["obj_verts"][:T].astype(np.float32)

    print("T compare:", T)
    print("omomo_j24:", omomo_j24.shape)
    print("human_verts:", human_verts.shape)
    print("human_faces:", human_faces.shape)
    print("pt_j24 mapped:", pt_j24.shape)
    print("pt_body52:", pt_body52.shape)

    direct_err = np.linalg.norm(omomo_j24 - pt_j24, axis=-1)
    print("\n--- direct joint error using corrected mapping ---")
    print("mean:", direct_err.mean())
    print("max:", direct_err.max())
    print("root mean:", np.linalg.norm(omomo_j24[:, 0] - pt_j24[:, 0], axis=-1).mean())
    print("left hand proxy mean:", np.linalg.norm(omomo_j24[:, 22] - pt_j24[:, 22], axis=-1).mean())
    print("right hand proxy mean:", np.linalg.norm(omomo_j24[:, 23] - pt_j24[:, 23], axis=-1).mean())

    omomo_rel = omomo_j24 - omomo_j24[:, 0:1]
    pt_rel = pt_j24 - pt_j24[:, 0:1]
    rel_err = np.linalg.norm(omomo_rel - pt_rel, axis=-1)

    print("\n--- per-frame root-relative joint error using corrected mapping ---")
    print("mean:", rel_err.mean())
    print("max:", rel_err.max())
    print("left hand proxy mean:", np.linalg.norm(omomo_rel[:, 22] - pt_rel[:, 22], axis=-1).mean())
    print("right hand proxy mean:", np.linalg.norm(omomo_rel[:, 23] - pt_rel[:, 23], axis=-1).mean())

    print_nearest_pt_joint_mapping(omomo_j24, pt_body52)

    hand_data = None
    if args.plot_hands or args.hand_out:
        hand_data = compute_hand_samples(
            human_verts=human_verts,
            human_faces=human_faces,
            seg_path=args.seg_path,
            n_samples=args.n_samples,
            seed=args.seed,
        )

    plot_omomo = omomo_j24.copy()
    plot_human_verts = human_verts.copy()
    plot_pt24 = pt_j24.copy()
    plot_pt52 = pt_body52.copy()
    plot_omomo_obj = omomo_obj.copy() if omomo_obj is not None else None
    plot_pt_obj = pt_obj.copy()

    plot_left_points = hand_data["left_points"].copy() if hand_data is not None else None
    plot_right_points = hand_data["right_points"].copy() if hand_data is not None else None

    if args.align_root0:
        o0 = plot_omomo[0, 0].copy()
        p0 = plot_pt52[0, 0].copy()

        plot_omomo -= o0[None, None, :]
        plot_human_verts -= o0[None, None, :]
        plot_pt24 -= p0[None, None, :]
        plot_pt52 -= p0[None, None, :]

        if plot_omomo_obj is not None:
            plot_omomo_obj -= o0[None, None, :]
        plot_pt_obj -= p0[None, :]

        if plot_left_points is not None:
            plot_left_points -= o0[None, None, :]
        if plot_right_points is not None:
            plot_right_points -= o0[None, None, :]

    if args.hand_out and hand_data is not None:
        hand_out = Path(args.hand_out)
        hand_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            hand_out,
            left_hand_points=hand_data["left_points"],
            right_hand_points=hand_data["right_points"],
            left_vertex_ids=hand_data["left_ids"],
            right_vertex_ids=hand_data["right_ids"],
            left_faces=hand_data["left_faces"],
            right_faces=hand_data["right_faces"],
            left_face_ids=hand_data["left_face_ids"],
            right_face_ids=hand_data["right_face_ids"],
            left_bary=hand_data["left_bary"],
            right_bary=hand_data["right_bary"],
            human_jnts=omomo_j24,
            obj_verts=omomo_obj if omomo_obj is not None else np.empty((0,)),
            human_faces=human_faces,
            obj_faces=omomo["obj_faces"] if "obj_faces" in omomo else np.empty((0,)),
        )
        print("saved hand samples:", hand_out)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in args.frames:
        if f < T:
            plot_frame(
                out_dir / f"overlay_frame_{f:04d}.png",
                plot_omomo,
                plot_pt24,
                plot_pt52,
                human_verts=plot_human_verts if args.plot_mesh else None,
                left_vertex_ids=hand_data["left_ids"] if hand_data is not None else None,
                right_vertex_ids=hand_data["right_ids"] if hand_data is not None else None,
                left_points=plot_left_points if args.plot_hands else None,
                right_points=plot_right_points if args.plot_hands else None,
                omomo_obj=plot_omomo_obj,
                pt_obj=plot_pt_obj,
                frame=f,
                labels=not args.no_labels,
                links=not args.no_links,
                mesh_stride=args.mesh_stride,
            )

    if args.make_video:
        make_overlay_video(
            args.video_out,
            plot_omomo,
            plot_pt24,
            plot_pt52,
            human_verts=plot_human_verts if args.plot_mesh else None,
            left_vertex_ids=hand_data["left_ids"] if hand_data is not None else None,
            right_vertex_ids=hand_data["right_ids"] if hand_data is not None else None,
            left_points=plot_left_points if args.plot_hands else None,
            right_points=plot_right_points if args.plot_hands else None,
            omomo_obj=plot_omomo_obj,
            pt_obj=plot_pt_obj,
            fps=args.fps,
            labels=not args.no_labels,
            links=not args.no_links,
            mesh_stride=args.mesh_stride,
        )


if __name__ == "__main__":
    main()