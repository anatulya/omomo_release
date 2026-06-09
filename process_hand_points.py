# process_replay_hand_points.py

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


ROBOT_PELVIS_Z = 0.665
HUMAN_PELVIS_Z = 1.1039201
DEFAULT_SCALE = ROBOT_PELVIS_Z / HUMAN_PELVIS_Z

REF_ROOT_POS = slice(0, 3)
REF_ROOT_ROT = slice(3, 7)  # xyzw


def yaw_only_rotation(rot: R) -> R:
    mat = rot.as_matrix()
    forward = mat @ np.array([1.0, 0.0, 0.0])
    yaw = np.arctan2(forward[1], forward[0])
    return R.from_euler("z", yaw)


def load_raw_pt(path: str) -> np.ndarray:
    x = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor .pt, got {type(x)}")
    return x.detach().cpu().numpy().astype(np.float32)


def load_hand_vertex_ids(seg_path: str, num_verts: int):
    with open(seg_path, "r") as f:
        seg = json.load(f)

    def collect(keys):
        ids = []
        for k in keys:
            if k in seg:
                ids.extend(seg[k])
        return np.asarray(sorted(set(ids)), dtype=np.int64)

    left_keys = [
        "leftHand",
        "leftHandIndex1", "leftHandIndex2", "leftHandIndex3",
        "leftHandMiddle1", "leftHandMiddle2", "leftHandMiddle3",
        "leftHandPinky1", "leftHandPinky2", "leftHandPinky3",
        "leftHandRing1", "leftHandRing2", "leftHandRing3",
        "leftHandThumb1", "leftHandThumb2", "leftHandThumb3",
    ]

    right_keys = [
        "rightHand",
        "rightHandIndex1", "rightHandIndex2", "rightHandIndex3",
        "rightHandMiddle1", "rightHandMiddle2", "rightHandMiddle3",
        "rightHandPinky1", "rightHandPinky2", "rightHandPinky3",
        "rightHandRing1", "rightHandRing2", "rightHandRing3",
        "rightHandThumb1", "rightHandThumb2", "rightHandThumb3",
    ]

    left_ids = collect(left_keys)
    right_ids = collect(right_keys)

    if len(left_ids) == 0:
        raise ValueError("No left-hand IDs found in segmentation.")
    if len(right_ids) == 0:
        raise ValueError("No right-hand IDs found in segmentation.")

    max_id = max(int(left_ids.max()), int(right_ids.max()))
    if max_id >= num_verts:
        raise ValueError(
            f"Segmentation max id {max_id} >= mesh num_verts {num_verts}. "
            "This segmentation JSON does not match the replay mesh topology."
        )

    print("segmentation:", seg_path)
    print("num_verts:", num_verts)
    print("left_ids:", left_ids.shape, "min/max:", int(left_ids.min()), int(left_ids.max()))
    print("right_ids:", right_ids.shape, "min/max:", int(right_ids.min()), int(right_ids.max()))

    return left_ids, right_ids


def build_global_submesh_faces(vertex_ids: np.ndarray, full_faces: np.ndarray):
    vertex_set = set(vertex_ids.tolist())
    out_faces = []

    for tri in full_faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a in vertex_set and b in vertex_set and c in vertex_set:
            out_faces.append([a, b, c])

    out_faces = np.asarray(out_faces, dtype=np.int64)

    if len(out_faces) == 0:
        raise ValueError("No submesh faces found for this hand vertex set.")

    return out_faces


def sample_persistent_surface_anchors(
    verts0: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
):
    v0 = verts0[faces[:, 0]]
    v1 = verts0[faces[:, 1]]
    v2 = verts0[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
    total_area = areas.sum()

    if total_area <= 1e-12:
        raise ValueError("Hand submesh has near-zero total area.")

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


def eval_surface_points(
    verts: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    bary: np.ndarray,
):
    tri = faces[face_ids]          # [N, 3]
    tri_verts = verts[:, tri, :]   # [T, N, 3, 3]

    points = (
        bary[None, :, 0:1] * tri_verts[:, :, 0, :]
        + bary[None, :, 1:2] * tri_verts[:, :, 1, :]
        + bary[None, :, 2:3] * tri_verts[:, :, 2, :]
    )

    return points.astype(np.float32)


def transform_points_translation_only(
    points: np.ndarray,
    origin: np.ndarray,
    scale: float,
    offset: np.ndarray,
):
    if points.ndim == 3:
        return ((points - origin.reshape(1, 1, 3)) * scale + offset.reshape(1, 1, 3)).astype(np.float32)
    if points.ndim == 2:
        return ((points - origin.reshape(1, 3)) * scale + offset.reshape(1, 3)).astype(np.float32)
    raise ValueError(f"Expected points ndim 2 or 3, got {points.ndim}")


def transform_points_strip_yaw(
    points: np.ndarray,
    root0: np.ndarray,
    root_head_0_inv: R,
    scale: float,
    offset: np.ndarray,
):
    out = np.empty_like(points, dtype=np.float32)

    if points.ndim == 3:
        for t in range(points.shape[0]):
            centered = points[t] - root0[None, :]
            rotated = root_head_0_inv.apply(centered)
            out[t] = rotated * scale + offset[None, :]
        return out.astype(np.float32)

    if points.ndim == 2:
        for t in range(points.shape[0]):
            centered = points[t] - root0
            rotated = root_head_0_inv.apply(centered)
            out[t] = rotated * scale + offset
        return out.astype(np.float32)

    raise ValueError(f"Expected points ndim 2 or 3, got {points.ndim}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--replay_npz", required=True)
    parser.add_argument("--seg_path", type=str, default="data/smplx_vert_segmentation.json")
    parser.add_argument("--out", required=True)

    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--pt",
        type=str,
        default="",
        help="Matching raw InterMimic/OMOMO_new .pt. Required if --strip_yaw is used.",
    )
    parser.add_argument(
        "--strip_yaw",
        action="store_true",
        help="Apply frame-0 yaw removal like transform_sequence_global.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Scene scale. Default is ROBOT_PELVIS_Z / HUMAN_PELVIS_Z.",
    )
    parser.add_argument(
        "--no_scale",
        action="store_true",
        help="Use scale=1.0.",
    )
    parser.add_argument(
        "--offset_xyz",
        type=float,
        nargs=3,
        default=[0.0, 0.0, ROBOT_PELVIS_Z],
        help="Offset added after root subtraction/yaw stripping/scaling.",
    )
    parser.add_argument(
        "--no_offset",
        action="store_true",
        help="Use zero offset.",
    )
    parser.add_argument(
        "--raw_only",
        action="store_true",
        help="Only save raw hand points, no processed hand points.",
    )

    args = parser.parse_args()

    replay = np.load(args.replay_npz, allow_pickle=True)

    human_verts = replay["human_verts"].astype(np.float32)
    human_jnts = replay["human_jnts"].astype(np.float32)
    human_faces = replay["human_faces"].astype(np.int64)

    obj_verts = replay["obj_verts"].astype(np.float32) if "obj_verts" in replay else None
    obj_faces = replay["obj_faces"].astype(np.int64) if "obj_faces" in replay else None

    seq_name = str(replay["seq_name"]) if "seq_name" in replay else Path(args.replay_npz).stem
    actual_len = int(replay["actual_len"]) if "actual_len" in replay else human_verts.shape[0]

    human_verts = human_verts[:actual_len]
    human_jnts = human_jnts[:actual_len]
    if obj_verts is not None:
        obj_verts = obj_verts[:actual_len]

    T, num_verts, _ = human_verts.shape

    print("seq_name:", seq_name)
    print("T:", T)
    print("human_verts:", human_verts.shape)
    print("human_jnts:", human_jnts.shape)
    print("human_faces:", human_faces.shape)

    left_ids, right_ids = load_hand_vertex_ids(args.seg_path, num_verts)

    left_faces = build_global_submesh_faces(left_ids, human_faces)
    right_faces = build_global_submesh_faces(right_ids, human_faces)

    print("left_faces:", left_faces.shape)
    print("right_faces:", right_faces.shape)

    rng = np.random.default_rng(args.seed)

    left_face_ids, left_bary = sample_persistent_surface_anchors(
        human_verts[0], left_faces, args.n_samples, rng
    )
    right_face_ids, right_bary = sample_persistent_surface_anchors(
        human_verts[0], right_faces, args.n_samples, rng
    )

    left_hand_points_raw = eval_surface_points(
        human_verts, left_faces, left_face_ids, left_bary
    )
    right_hand_points_raw = eval_surface_points(
        human_verts, right_faces, right_face_ids, right_bary
    )

    print("left_hand_points_raw:", left_hand_points_raw.shape)
    print("right_hand_points_raw:", right_hand_points_raw.shape)

    scale = 1.0 if args.no_scale else float(args.scale)
    offset = np.zeros(3, dtype=np.float32) if args.no_offset else np.asarray(args.offset_xyz, dtype=np.float32)

    save_dict = {
        "seq_name": np.asarray(seq_name),
        "actual_len": np.asarray(actual_len),

        "scale": np.asarray(scale, dtype=np.float32),
        "offset": offset,
        "strip_yaw": np.asarray(args.strip_yaw),

        "left_hand_points_raw": left_hand_points_raw,
        "right_hand_points_raw": right_hand_points_raw,

        "left_vertex_ids": left_ids,
        "right_vertex_ids": right_ids,
        "left_faces": left_faces,
        "right_faces": right_faces,
        "left_face_ids": left_face_ids,
        "right_face_ids": right_face_ids,
        "left_bary": left_bary,
        "right_bary": right_bary,

        "human_jnts_raw": human_jnts,
        "human_faces": human_faces,
    }

    if obj_verts is not None:
        save_dict["obj_verts_raw"] = obj_verts
    if obj_faces is not None:
        save_dict["obj_faces"] = obj_faces

    if args.strip_yaw:
        if not args.pt:
            raise ValueError("--strip_yaw requires --pt so we can read frame-0 root rotation.")

        raw_pt = load_raw_pt(args.pt)
        raw_pt = raw_pt[:actual_len]

        pt_root0 = raw_pt[0, REF_ROOT_POS].astype(np.float32)
        replay_root0 = human_jnts[0, 0].astype(np.float32)

        root_rot0 = R.from_quat(raw_pt[0, REF_ROOT_ROT])
        root_head_0_inv = yaw_only_rotation(root_rot0).inv()

        root0 = replay_root0


        print("processing mode: strip_yaw")
        print("root0 from .pt:", root0)
        print("root_rot0 xyzw from .pt:", raw_pt[0, REF_ROOT_ROT])
        print("scale:", scale)
        print("offset:", offset)

        save_dict["origin_root0"] = root0
        save_dict["root_rot0_xyzw"] = raw_pt[0, REF_ROOT_ROT].astype(np.float32)
        save_dict["pt_origin_root0"] = pt_root0
        save_dict["replay_origin_root0"] = replay_root0
        save_dict["raw_frame_delta_replay_minus_pt"] = replay_root0 - pt_root0

        if not args.raw_only:
            save_dict["left_hand_points"] = transform_points_strip_yaw(
                left_hand_points_raw, root0, root_head_0_inv, scale, offset
            )
            save_dict["right_hand_points"] = transform_points_strip_yaw(
                right_hand_points_raw, root0, root_head_0_inv, scale, offset
            )
            save_dict["human_jnts"] = transform_points_strip_yaw(
                human_jnts, root0, root_head_0_inv, scale, offset
            )

            if obj_verts is not None:
                save_dict["obj_verts"] = transform_points_strip_yaw(
                    obj_verts, root0, root_head_0_inv, scale, offset
                )

    else:
        root0 = human_jnts[0, 0].astype(np.float32)

        print("processing mode: translation_only")
        print("root0 from human_jnts[0,0]:", root0)
        print("scale:", scale)
        print("offset:", offset)

        save_dict["origin_root0"] = root0

        if not args.raw_only:
            save_dict["left_hand_points"] = transform_points_translation_only(
                left_hand_points_raw, root0, scale, offset
            )
            save_dict["right_hand_points"] = transform_points_translation_only(
                right_hand_points_raw, root0, scale, offset
            )
            save_dict["human_jnts"] = transform_points_translation_only(
                human_jnts, root0, scale, offset
            )

            if obj_verts is not None:
                save_dict["obj_verts"] = transform_points_translation_only(
                    obj_verts, root0, scale, offset
                )

    if not args.raw_only:
        print("left_hand_points processed:", save_dict["left_hand_points"].shape)
        print("right_hand_points processed:", save_dict["right_hand_points"].shape)
        print("human_jnts processed:", save_dict["human_jnts"].shape)
        if "obj_verts" in save_dict:
            print("obj_verts processed:", save_dict["obj_verts"].shape)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save_dict)

    print("saved:", out_path)


if __name__ == "__main__":
    main()