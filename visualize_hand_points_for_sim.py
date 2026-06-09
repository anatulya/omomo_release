# visualize_hand_points_for_sim.py

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R


REF_ROOT_POS = slice(0, 3)
REF_ROOT_ROT = slice(3, 7)
REF_BODY_POS = slice(162, 318)  # 52 * 3
REF_OBJ_POS = slice(318, 321)
REF_OBJ_ROT = slice(321, 325)   # processed object quat is wxyz in your preprocess


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

JOINT24_EDGES = BODY22_EDGES + [(20, 22), (21, 23)]


BOX_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (3, 1), (3, 2), (3, 7),
    (5, 1), (5, 4), (5, 7),
    (6, 2), (6, 4), (6, 7),
]


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


def get_traj_body52(arr):
    return arr[:, REF_BODY_POS].reshape(arr.shape[0], 52, 3).astype(np.float32)


def get_traj_j24(arr):
    body52 = get_traj_body52(arr)
    return body52[:, OMOMO24_TO_PT52, :]


def get_traj_obj_rot_mats(traj):
    q_wxyz = traj[:, REF_OBJ_ROT].astype(np.float32)
    norms = np.linalg.norm(q_wxyz, axis=-1)

    # If a bad/empty quat appears, avoid crashing.
    bad = norms < 1e-8
    if np.any(bad):
        q_wxyz = q_wxyz.copy()
        q_wxyz[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        norms = np.linalg.norm(q_wxyz, axis=-1)

    q_wxyz = q_wxyz / norms[:, None]

    # scipy wants xyzw
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    return R.from_quat(q_xyzw).as_matrix().astype(np.float32)


def box_corners(center, halfsize, rot_mat=None):
    signs = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ], dtype=np.float32)

    local = signs * halfsize[None, :]

    if rot_mat is not None:
        local = local @ rot_mat.T

    return center[None, :] + local


def draw_box(ax, center, halfsize, rot_mat=None, label=None):
    corners = box_corners(center, halfsize, rot_mat=rot_mat)
    draw_edges(ax, corners, BOX_EDGES, linewidth=1.8, alpha=0.9, linestyle="-", label=label)
    ax.scatter([center[0]], [center[1]], [center[2]], s=60, marker="*", label="box center")


def load_halfsize(path, seq_name):
    if not path:
        return None

    with open(path, "r") as f:
        d = json.load(f)

    if seq_name not in d:
        print(f"[warn] {seq_name} not found in halfsize json")
        return None

    return np.asarray(d[seq_name], dtype=np.float32)


def set_equal_bounds(ax, arrays, min_radius=0.5):
    pts = []
    for x in arrays:
        if x is not None:
            pts.append(x.reshape(-1, 3))

    pts = np.concatenate(pts, axis=0)
    center = pts.mean(axis=0)
    radius = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) * 0.55
    radius = max(float(radius), min_radius)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_frame(
    out_path,
    frame,
    left_pts,
    right_pts,
    hand_j24=None,
    traj_j24=None,
    traj_body52=None,
    obj_verts=None,
    traj_obj_pos=None,
    traj_obj_rot_mats=None,
    halfsize=None,
    obj_stride=1000,
    labels=False,
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    L = left_pts[frame]
    Rpts = right_pts[frame]

    ax.scatter(L[:, 0], L[:, 1], L[:, 2], s=6, alpha=0.65, label="processed left hand points")
    ax.scatter(Rpts[:, 0], Rpts[:, 1], Rpts[:, 2], s=6, alpha=0.65, label="processed right hand points")

    if hand_j24 is not None:
        H = hand_j24[frame]
        draw_edges(ax, H, JOINT24_EDGES, linewidth=2.0, alpha=0.85, label="hand npz human_jnts")
        ax.scatter(H[:, 0], H[:, 1], H[:, 2], s=32, label="hand npz joints24")

    if traj_j24 is not None:
        J = traj_j24[frame]
        draw_edges(ax, J, JOINT24_EDGES, linewidth=1.5, alpha=0.6, linestyle="--", label="processed traj joints24")
        ax.scatter(J[:, 0], J[:, 1], J[:, 2], s=24, marker="x", label="processed traj joints24")

    if traj_body52 is not None:
        B = traj_body52[frame]
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], s=9, alpha=0.20, marker=".", label="processed traj body52")

    if obj_verts is not None:
        O = obj_verts[frame]
        stride = max(1, len(O) // obj_stride)
        O_ds = O[::stride]
        ax.scatter(O_ds[:, 0], O_ds[:, 1], O_ds[:, 2], s=1, alpha=0.18, label="processed obj verts")

    if traj_obj_pos is not None:
        c = traj_obj_pos[frame]
        ax.scatter([c[0]], [c[1]], [c[2]], s=80, marker="*", label="processed traj obj pos")

        if halfsize is not None:
            rot_mat = traj_obj_rot_mats[frame] if traj_obj_rot_mats is not None else None
            draw_box(ax, c, halfsize, rot_mat=rot_mat, label="rotated sim box halfsize")

    if labels and hand_j24 is not None:
        for j, p in enumerate(hand_j24[frame]):
            ax.text(p[0], p[1], p[2], str(j), fontsize=7)

    bounds = [L, Rpts]

    if hand_j24 is not None:
        bounds.append(hand_j24[frame])
    if traj_j24 is not None:
        bounds.append(traj_j24[frame])
    if traj_body52 is not None:
        bounds.append(traj_body52[frame])
    if obj_verts is not None:
        bounds.append(obj_verts[frame][::max(1, len(obj_verts[frame]) // 1000)])
    if traj_obj_pos is not None:
        bounds.append(traj_obj_pos[frame:frame + 1])
    if traj_obj_pos is not None and halfsize is not None:
        rot_mat = traj_obj_rot_mats[frame] if traj_obj_rot_mats is not None else None
        bounds.append(box_corners(traj_obj_pos[frame], halfsize, rot_mat=rot_mat))

    set_equal_bounds(ax, bounds)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"processed/sim frame {frame}")
    ax.legend(loc="upper right", fontsize=7)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print("saved:", out_path)


def make_video(
    video_out,
    left_pts,
    right_pts,
    hand_j24=None,
    traj_j24=None,
    traj_body52=None,
    obj_verts=None,
    traj_obj_pos=None,
    traj_obj_rot_mats=None,
    halfsize=None,
    fps=30,
    obj_stride=1000,
):
    T = len(left_pts)
    video_out = Path(video_out)
    video_out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    global_bounds = [
        left_pts.reshape(-1, 3),
        right_pts.reshape(-1, 3),
    ]

    if hand_j24 is not None:
        global_bounds.append(hand_j24.reshape(-1, 3))
    if traj_j24 is not None:
        global_bounds.append(traj_j24.reshape(-1, 3))
    if traj_body52 is not None:
        global_bounds.append(traj_body52.reshape(-1, 3))
    if obj_verts is not None:
        global_bounds.append(obj_verts[:, ::max(1, obj_verts.shape[1] // 1000), :].reshape(-1, 3))
    if traj_obj_pos is not None:
        global_bounds.append(traj_obj_pos.reshape(-1, 3))

    all_pts = np.concatenate(global_bounds, axis=0)
    center = all_pts.mean(axis=0)
    radius = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.55
    radius = max(float(radius), 0.5)

    def update(frame):
        ax.clear()

        L = left_pts[frame]
        Rpts = right_pts[frame]

        ax.scatter(L[:, 0], L[:, 1], L[:, 2], s=5, alpha=0.65, label="left hand points")
        ax.scatter(Rpts[:, 0], Rpts[:, 1], Rpts[:, 2], s=5, alpha=0.65, label="right hand points")

        if hand_j24 is not None:
            H = hand_j24[frame]
            draw_edges(ax, H, JOINT24_EDGES, linewidth=2.0, alpha=0.85, label="hand npz joints")
            ax.scatter(H[:, 0], H[:, 1], H[:, 2], s=28)

        if traj_j24 is not None:
            J = traj_j24[frame]
            draw_edges(ax, J, JOINT24_EDGES, linewidth=1.4, alpha=0.55, linestyle="--", label="processed traj joints")
            ax.scatter(J[:, 0], J[:, 1], J[:, 2], s=22, marker="x")

        if traj_body52 is not None:
            B = traj_body52[frame]
            ax.scatter(B[:, 0], B[:, 1], B[:, 2], s=9, alpha=0.18, marker=".", label="processed traj body52")

        if obj_verts is not None:
            O = obj_verts[frame]
            stride = max(1, len(O) // obj_stride)
            O_ds = O[::stride]
            ax.scatter(O_ds[:, 0], O_ds[:, 1], O_ds[:, 2], s=1, alpha=0.18, label="obj verts")

        if traj_obj_pos is not None:
            c = traj_obj_pos[frame]
            ax.scatter([c[0]], [c[1]], [c[2]], s=80, marker="*", label="traj obj pos")

            if halfsize is not None:
                rot_mat = traj_obj_rot_mats[frame] if traj_obj_rot_mats is not None else None
                draw_box(ax, c, halfsize, rot_mat=rot_mat, label="rotated sim box")

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"processed/sim frame {frame}/{T - 1}")
        ax.legend(loc="upper right", fontsize=7)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    print("saving video:", video_out)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(video_out, writer=writer)
    plt.close(fig)
    print("saved video:", video_out)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hand_npz", required=True)
    parser.add_argument("--traj_npy", default="", help="Optional processed trajectory .npy used by sim")
    parser.add_argument("--halfsize_json", default="", help="Optional halfsize_lookup.json")
    parser.add_argument("--seq_name", default="", help="Needed only for halfsize_json lookup")

    parser.add_argument("--out_dir", default="sim_hand_viz")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 30, 60, 90])

    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--video_out", default="sim_hand_viz/hand_points_sim.mp4")
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--obj_stride", type=int, default=1000)
    parser.add_argument("--labels", action="store_true")

    args = parser.parse_args()

    data = np.load(args.hand_npz, allow_pickle=True)

    if "left_hand_points" not in data or "right_hand_points" not in data:
        raise KeyError(
            "hand_npz must contain processed fields 'left_hand_points' and 'right_hand_points'. "
            "Rerun process_replay_hand_points.py without --raw_only."
        )

    left_pts = data["left_hand_points"].astype(np.float32)
    right_pts = data["right_hand_points"].astype(np.float32)

    hand_j24 = data["human_jnts"].astype(np.float32) if "human_jnts" in data else None
    obj_verts = data["obj_verts"].astype(np.float32) if "obj_verts" in data else None

    T = len(left_pts)

    traj_j24 = None
    traj_body52 = None
    traj_obj_pos = None
    traj_obj_rot_mats = None
    halfsize = None

    if args.traj_npy:
        traj = np.load(args.traj_npy).astype(np.float32)
        T = min(T, len(traj))

        left_pts = left_pts[:T]
        right_pts = right_pts[:T]

        if hand_j24 is not None:
            hand_j24 = hand_j24[:T]
        if obj_verts is not None:
            obj_verts = obj_verts[:T]

        traj = traj[:T]
        traj_body52 = get_traj_body52(traj)
        traj_j24 = get_traj_j24(traj)
        traj_obj_pos = traj[:, REF_OBJ_POS].astype(np.float32)
        traj_obj_rot_mats = get_traj_obj_rot_mats(traj)

    if args.halfsize_json:
        seq_name = args.seq_name
        if not seq_name and "seq_name" in data:
            seq_name = str(data["seq_name"])
        halfsize = load_halfsize(args.halfsize_json, seq_name)

    print("T:", T)
    print("left_pts:", left_pts.shape)
    print("right_pts:", right_pts.shape)

    if hand_j24 is not None:
        print("hand_j24:", hand_j24.shape)
    if traj_j24 is not None:
        print("traj_j24:", traj_j24.shape)
    if traj_body52 is not None:
        print("traj_body52:", traj_body52.shape)
    if obj_verts is not None:
        print("obj_verts:", obj_verts.shape)
    if traj_obj_pos is not None:
        print("traj_obj_pos:", traj_obj_pos.shape)
    if traj_obj_rot_mats is not None:
        print("traj_obj_rot_mats:", traj_obj_rot_mats.shape)
    if halfsize is not None:
        print("halfsize:", halfsize)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in args.frames:
        if f < T:
            plot_frame(
                out_dir / f"sim_overlay_frame_{f:04d}.png",
                f,
                left_pts,
                right_pts,
                hand_j24=hand_j24,
                traj_j24=traj_j24,
                traj_body52=traj_body52,
                obj_verts=obj_verts,
                traj_obj_pos=traj_obj_pos,
                traj_obj_rot_mats=traj_obj_rot_mats,
                halfsize=halfsize,
                obj_stride=args.obj_stride,
                labels=args.labels,
            )

    if args.make_video:
        make_video(
            args.video_out,
            left_pts,
            right_pts,
            hand_j24=hand_j24,
            traj_j24=traj_j24,
            traj_body52=traj_body52,
            obj_verts=obj_verts,
            traj_obj_pos=traj_obj_pos,
            traj_obj_rot_mats=traj_obj_rot_mats,
            halfsize=halfsize,
            fps=args.fps,
            obj_stride=args.obj_stride,
        )


if __name__ == "__main__":
    main()