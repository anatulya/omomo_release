import argparse
from pathlib import Path

import numpy as np


def load_obj_vertices_faces(path: Path):
    verts = []
    faces = []

    with path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]

                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    if verts.size == 0:
        raise ValueError(f"No vertices found in {path}")
    if faces.size == 0:
        raise ValueError(f"No faces found in {path}")

    return verts, faces


def mesh_center(verts: np.ndarray, mode: str):
    if mode == "aabb":
        return ((verts.min(axis=0) + verts.max(axis=0)) * 0.5).astype(np.float32)
    if mode == "mean":
        return verts.mean(axis=0).astype(np.float32)
    if mode == "none":
        return np.zeros(3, dtype=np.float32)
    raise ValueError(f"Unknown center mode: {mode}")


def sample_mesh_surface_points(
    verts: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
    total_area = areas.sum()
    if total_area <= 1e-12:
        raise ValueError("Mesh has near-zero total surface area.")

    probs = areas / total_area
    face_ids = rng.choice(len(faces), size=n_samples, p=probs)

    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)

    sqrt_r1 = np.sqrt(r1)
    bary = np.stack(
        [
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ],
        axis=-1,
    ).astype(np.float32)

    tri = faces[face_ids]
    tri_verts = verts[tri]
    points = (
        bary[:, 0:1] * tri_verts[:, 0]
        + bary[:, 1:2] * tri_verts[:, 1]
        + bary[:, 2:3] * tri_verts[:, 2]
    )

    return points.astype(np.float32), face_ids.astype(np.int64), bary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample persistent surface points on an OMOMO object mesh."
    )
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Final scale used by the sim/XML mesh asset for this object.",
    )
    parser.add_argument(
        "--center",
        choices=["aabb", "mean", "none"],
        default="aabb",
        help="Center to subtract before scaling. Use aabb for MuJoCo-centered mesh body frames.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    verts, faces = load_obj_vertices_faces(args.mesh)
    center = mesh_center(verts, args.center)

    points_obj_local, face_ids, bary = sample_mesh_surface_points(
        verts=verts,
        faces=faces,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    points_body_local = ((points_obj_local - center[None, :]) * args.scale).astype(
        np.float32
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        mesh_path=np.asarray(str(args.mesh)),
        n_samples=np.asarray(args.n_samples, dtype=np.int32),
        seed=np.asarray(args.seed, dtype=np.int32),
        scale=np.asarray(args.scale, dtype=np.float32),
        center_mode=np.asarray(args.center),
        mesh_center=center,
        points_obj_local=points_obj_local,
        points_body_local=points_body_local,
        face_ids=face_ids,
        bary=bary,
        verts_min=verts.min(axis=0).astype(np.float32),
        verts_max=verts.max(axis=0).astype(np.float32),
    )

    print("mesh:", args.mesh)
    print("verts:", verts.shape, "faces:", faces.shape)
    print("center_mode:", args.center)
    print("mesh_center:", center)
    print("scale:", args.scale)
    print("points_body_local:", points_body_local.shape)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
