import argparse
import torch

from pointpillars.utils import setup_seed
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.loss import Loss


def move_batch_to_cuda(data_dict):
    for key in data_dict:
        for j, item in enumerate(data_dict[key]):
            if torch.is_tensor(item):
                data_dict[key][j] = item.cuda()
    return data_dict


def main(args):
    setup_seed()

    print("Loading KITTI train dataset...")
    train_dataset = Kitti(data_root=args.data_root, split="train")

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    print("Number of training samples:", len(train_dataset))
    print("Number of batches:", len(train_dataloader))

    print("Creating PointPillars model...")
    model = PointPillars(nclasses=args.nclasses).cuda()
    model.train()

    loss_func = Loss()

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.init_lr,
        betas=(0.95, 0.99),
        weight_decay=0.01,
    )

    print("Loading one batch...")
    data_dict = next(iter(train_dataloader))
    data_dict = move_batch_to_cuda(data_dict)

    batched_pts = data_dict["batched_pts"]
    batched_gt_bboxes = data_dict["batched_gt_bboxes"]
    batched_labels = data_dict["batched_labels"]

    print("Running forward pass...")
    bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = model(
        batched_pts=batched_pts,
        mode="train",
        batched_gt_bboxes=batched_gt_bboxes,
        batched_gt_labels=batched_labels,
    )

    print("Preparing predictions and targets for loss...")

    bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
    bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

    batched_bbox_labels = anchor_target_dict["batched_labels"].reshape(-1)
    batched_label_weights = anchor_target_dict["batched_label_weights"].reshape(-1)
    batched_bbox_reg = anchor_target_dict["batched_bbox_reg"].reshape(-1, 7)
    batched_dir_labels = anchor_target_dict["batched_dir_labels"].reshape(-1)

    pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)

    bbox_pred = bbox_pred[pos_idx]
    batched_bbox_reg = batched_bbox_reg[pos_idx]
    bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
    batched_dir_labels = batched_dir_labels[pos_idx]

    # Direction/yaw handling copied from the main train.py logic.
    bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(
        batched_bbox_reg[:, -1].clone()
    )
    batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(
        batched_bbox_reg[:, -1].clone()
    )

    num_cls_pos = (batched_bbox_labels < args.nclasses).sum()

    bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
    batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
    batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

    print("Calculating loss...")
    loss_dict = loss_func(
        bbox_cls_pred=bbox_cls_pred,
        bbox_pred=bbox_pred,
        bbox_dir_cls_pred=bbox_dir_cls_pred,
        batched_labels=batched_bbox_labels,
        num_cls_pos=num_cls_pos,
        batched_bbox_reg=batched_bbox_reg,
        batched_dir_labels=batched_dir_labels,
    )

    loss = loss_dict["total_loss"]

    print("Running backward pass and optimizer step...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print()
    print("Training sanity check completed successfully.")
    print("Loss values:")

    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.item():.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-batch PointPillars training sanity check")
    parser.add_argument("--data_root", required=True, help="KITTI root for PointPillars")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--nclasses", type=int, default=3)
    parser.add_argument("--init_lr", type=float, default=0.00025)

    args = parser.parse_args()
    main(args)
