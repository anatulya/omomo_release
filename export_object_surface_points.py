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


def sample_mesh_surface_points_area_weighted(
    verts: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    seed: int,
):
    """Area-weighted random sampling. Unbiased w.r.t. surface area, but can
    leave clumps/gaps at small n_samples since draws are independent."""
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


def farthest_point_sampling(points: np.ndarray, n_select: int, seed: int) -> np.ndarray:
    """Greedily select n_select indices from points that maximize the
    minimum pairwise distance (Eldar et al. style FPS). Produces much more
    even spatial coverage than independent random draws."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    if n_select >= n:
        return np.arange(n, dtype=np.int64)

    selected = np.zeros(n_select, dtype=np.int64)
    selected[0] = rng.integers(n)

    # running minimum distance from every candidate point to the selected set
    min_dist = np.linalg.norm(points - points[selected[0]], axis=-1)

    for i in range(1, n_select):
        next_idx = int(np.argmax(min_dist))
        selected[i] = next_idx
        new_dist = np.linalg.norm(points - points[next_idx], axis=-1)
        min_dist = np.minimum(min_dist, new_dist)

    return selected


def sample_mesh_surface_points(
    verts: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    seed: int,
    method: str,
    oversample_factor: int,
):
    if method == "area":
        return sample_mesh_surface_points_area_weighted(verts, faces, n_samples, seed)

    if method == "fps":
        # Oversample with area-weighted sampling, then greedily thin down to
        # n_samples points that are maximally spread out across the surface.
        n_candidates = n_samples * oversample_factor
        cand_points, cand_face_ids, cand_bary = sample_mesh_surface_points_area_weighted(
            verts, faces, n_candidates, seed
        )
        keep_idx = farthest_point_sampling(cand_points, n_samples, seed)
        return cand_points[keep_idx], cand_face_ids[keep_idx], cand_bary[keep_idx]

    raise ValueError(f"Unknown sampling method: {method}")


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
    parser.add_argument(
        "--method",
        choices=["area", "fps"],
        default="area",
        help="area: pure area-weighted random sampling (original behavior). "
             "fps: oversample with area-weighted sampling then farthest-point-sample "
             "down to n_samples for more even spatial coverage.",
    )
    parser.add_argument(
        "--oversample_factor",
        type=int,
        default=8,
        help="Only used with --method fps. Number of candidate points sampled "
             "per final point before farthest point sampling thins them down.",
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
        method=args.method,
        oversample_factor=args.oversample_factor,
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
        method=np.asarray(args.method),
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
    print("method:", args.method, f"(oversample_factor={args.oversample_factor})" if args.method == "fps" else "")
    print("mesh_center:", center)
    print("scale:", args.scale)
    print("points_body_local:", points_body_local.shape)
    print("saved:", args.out)


if __name__ == "__main__":
    main()