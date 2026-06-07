# scripts/debug_export_step1.py

import argparse
import torch
from torch.utils.data._utils.collate import default_collate

from manip.data.hand_foot_dataset import HandFootManipDataset


def describe_value(name, value):
    if torch.is_tensor(value):
        msg = f"{name:24s} tensor shape={tuple(value.shape)} dtype={value.dtype}"
        if value.numel() > 0 and value.is_floating_point():
            msg += f" min={value.min().item():.4f} max={value.max().item():.4f}"
        print(msg)
    elif isinstance(value, (list, tuple)):
        preview = value[0] if len(value) > 0 else None
        print(f"{name:24s} {type(value).__name__} len={len(value)} first={preview}")
    else:
        print(f"{name:24s} {type(value).__name__}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_folder", type=str, default="data")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--use_object_split", action="store_true")
    args = parser.parse_args()

    ds = HandFootManipDataset(
        train=args.train,
        data_root_folder=args.data_root_folder,
        window=args.window,
        use_object_splits=args.use_object_split,
    )

    print(f"dataset length: {len(ds)}")
    print(f"dataset has bm_dict: {hasattr(ds, 'bm_dict')}")
    if hasattr(ds, "bm_dict"):
        print(f"bm_dict keys: {list(ds.bm_dict.keys())}")

    item = ds[args.idx]

    print("\n--- raw dataset item ---")
    for k in sorted(item.keys()):
        describe_value(k, item[k])

    batch = default_collate([item])

    print("\n--- batch size 1 item ---")
    for k in sorted(batch.keys()):
        describe_value(k, batch[k])

    print("\n--- motion layout sanity ---")
    motion = batch["motion"]
    print("motion shape:", tuple(motion.shape))
    print("expected OMOMO full-body dim:", 24 * 3 + 22 * 6)

    if motion.shape[-1] == 24 * 3 + 22 * 6:
        print("motion dim matches full-body diffusion format")
        print("first 24*3 global-jpos block:", tuple(motion[:, :, :24 * 3].shape))
        print("last 22*6 global-rot block:", tuple(motion[:, :, -22 * 6:].shape))
    else:
        print("motion dim does NOT match 204; inspect this before replaying")

    print("\n--- identity fields ---")
    for k in ["seq_name", "gender", "seq_len"]:
        if k in batch:
            describe_value(k, batch[k])


if __name__ == "__main__":
    main()