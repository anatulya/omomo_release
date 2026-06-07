# export_omomo_mesh.py

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from manip.data.hand_foot_dataset import HandFootManipDataset
from trainer_full_body_manip_diffusion import Trainer


class ReplayOnly:
    pass


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def find_item_by_seq_name(ds, seq_name):
    matches = []

    for i in range(len(ds)):
        item = ds[i]
        if item["seq_name"] == seq_name:
            matches.append((i, item))

    if not matches:
        partial = []
        for i in range(len(ds)):
            name = ds[i]["seq_name"]
            if seq_name in name or name in seq_name:
                partial.append((i, name))

        msg = f"Could not find seq_name={seq_name}"
        if partial:
            msg += "\nPartial matches:\n"
            msg += "\n".join([f"  idx={i}: {name}" for i, name in partial[:50]])
        raise ValueError(msg)

    print(f"found {len(matches)} window(s) for seq_name={seq_name}")
    print(f"using dataset idx={matches[0][0]}")
    return matches[0]


def list_matching_seq_names(ds, pattern):
    matches = []

    for i in range(len(ds)):
        name = ds[i]["seq_name"]
        if pattern in name:
            matches.append((i, name))

    print(f"found {len(matches)} matches for pattern={pattern}")
    for i, name in matches[:100]:
        print(f"  idx={i}: {name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root_folder", type=str, default="data")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--list_pattern", type=str, default="")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--use_object_split", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="debug_exports/omomo_replay.npz")

    # Default is no rendering. Pass --render to save meshes/render video.
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--vis_folder", type=str, default="debug_vis")
    parser.add_argument("--vis_tag", type=str, default="gt_replay")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ds = HandFootManipDataset(
        train=args.train,
        data_root_folder=args.data_root_folder,
        window=args.window,
        use_object_splits=args.use_object_split,
    )

    if args.list_pattern:
        list_matching_seq_names(ds, args.list_pattern)
        return

    if args.seq_name:
        idx, item = find_item_by_seq_name(ds, args.seq_name)
    else:
        idx = args.idx
        item = ds[idx]

    batch = default_collate([item])
    batch_dev = move_batch_to_device(batch, device)

    print("using idx:", idx)
    print("seq_name:", batch["seq_name"][0])
    print("gender:", batch["gender"][0])
    print("motion:", tuple(batch["motion"].shape))
    print("seq_len:", int(batch["seq_len"][0]))
    print("render:", args.render)

    replay = ReplayOnly()
    replay.ds = ds
    replay.vis_folder = args.vis_folder

    # OMOMO gen_vis_res uses this flag internally.
    # True avoids mesh/blender export. False enables the render/export branch.
    replay.for_quant_eval = not args.render

    # Reuse OMOMO's existing replay method without constructing the full Trainer.
    replay.gen_vis_res = Trainer.gen_vis_res.__get__(replay, ReplayOnly)

    with torch.no_grad():
        (
            human_trans,
            human_rot,
            human_jnts,
            human_verts,
            human_faces,
            obj_verts,
            obj_faces,
            actual_len,
        ) = replay.gen_vis_res(
            batch_dev["motion"],
            batch_dev,
            step=0,

            # If render=False:
            #   vis_gt=True + for_quant_eval=True prevents Blender rendering.
            #
            # If render=True:
            #   vis_gt=False + for_quant_eval=False triggers OMOMO mesh export/render.
            vis_gt=not args.render,
            vis_tag=args.vis_tag,
            for_quant_eval=not args.render,
        )

    print("\n--- replay result ---")
    print("actual_len:", actual_len)
    print("human_trans:", tuple(human_trans.shape))
    print("human_rot:", tuple(human_rot.shape))
    print("human_jnts:", tuple(human_jnts.shape))
    print("human_verts:", tuple(human_verts.shape))
    print("human_faces:", human_faces.shape)
    print("obj_verts:", tuple(obj_verts.shape))
    print("obj_faces:", obj_faces.shape)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        seq_name=batch["seq_name"][0],
        gender=batch["gender"][0],
        actual_len=actual_len,

        human_trans=human_trans.detach().cpu().numpy(),
        human_rot=human_rot.detach().cpu().numpy(),
        human_jnts=human_jnts.detach().cpu().numpy(),
        human_verts=human_verts.detach().cpu().numpy(),
        human_faces=human_faces,

        obj_verts=obj_verts.detach().cpu().numpy(),
        obj_faces=obj_faces,
    )

    print("\nSaved replay arrays to:")
    print(f"  {out_path}")

    if args.render:
        print("\nRendered output should be under:")
        print(f"  {args.vis_folder}/{args.vis_tag}/0/")
    else:
        print("\nRender disabled; saved arrays only.")


if __name__ == "__main__":
    main()