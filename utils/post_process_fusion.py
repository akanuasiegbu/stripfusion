import torch
from utils.general import box_iou


def fuse_predictions(thermal_preds, rgb_preds, thermal_conf_thresh=0.1, rgb_conf_thresh=0.1, iou_thresh=0.75):
    # import pdb; pdb.set_trace()
    fused_preds = []

    # print("Fusing predictions.\n\n")

    # Process each batch independently
    for b in range(thermal_preds.shape[0]):  # Loop over batches
        thermal_batch = thermal_preds[b]
        rgb_batch = rgb_preds[b]

        # Step 1: Filter low-confidence predictions
        thermal_batch = thermal_batch[thermal_batch[:, 4] >= thermal_conf_thresh]
        rgb_batch = rgb_batch[rgb_batch[:, 4] >= rgb_conf_thresh]

        if thermal_batch.numel() == 0 or rgb_batch.numel() == 0:
            # If no predictions, return empty tensor for this batch
            fused_preds.append(torch.zeros((0, thermal_batch.shape[-1]), device=thermal_batch.device))
            continue

        # Step 2: Compute IoU between thermal and RGB boxes
        thermal_boxes = torch.cat([
            thermal_batch[:, :2] - thermal_batch[:, 2:4] / 2,  # x_min, y_min
            thermal_batch[:, :2] + thermal_batch[:, 2:4] / 2   # x_max, y_max
        ], dim=1)
        rgb_boxes = torch.cat([
            rgb_batch[:, :2] - rgb_batch[:, 2:4] / 2,
            rgb_batch[:, :2] + rgb_batch[:, 2:4] / 2
        ], dim=1)

        ious = box_iou(thermal_boxes, rgb_boxes)

        # Step 3: Match pairs based on IoU threshold
        matched_indices = (ious >= iou_thresh).nonzero(as_tuple=False)
        thermal_matched = thermal_batch[matched_indices[:, 0]]
        rgb_matched = rgb_batch[matched_indices[:, 1]]

        # Step 4: Fusion logic
        fused_x_min = torch.min(thermal_matched[:, 0] - thermal_matched[:, 2] / 2,
                                rgb_matched[:, 0] - rgb_matched[:, 2] / 2)
        fused_y_min = torch.min(thermal_matched[:, 1] - thermal_matched[:, 3] / 2,
                                rgb_matched[:, 1] - rgb_matched[:, 3] / 2)
        fused_x_max = torch.max(thermal_matched[:, 0] + thermal_matched[:, 2] / 2,
                                rgb_matched[:, 0] + rgb_matched[:, 2] / 2)
        fused_y_max = torch.max(thermal_matched[:, 1] + thermal_matched[:, 3] / 2,
                                rgb_matched[:, 1] + rgb_matched[:, 3] / 2)

        fused_x = (fused_x_min + fused_x_max) / 2
        fused_y = (fused_y_min + fused_y_max) / 2
        fused_width = fused_x_max - fused_x_min
        fused_height = fused_y_max - fused_y_min

        fused_conf = (thermal_matched[:, 4] + rgb_matched[:, 4]) / 2
        # fused_conf = torch.max(thermal_matched[:, 4], rgb_matched[:, 4]), 
        # not including in paper but testing out average and max with diff threholds found average 
        fused_class = torch.where(
            thermal_matched[:, 4] >= rgb_matched[:, 4],
            thermal_matched[:, 5],
            rgb_matched[:, 5]
        )

        fused_batch = torch.stack(
            [fused_x, fused_y, fused_width, fused_height, fused_conf, fused_class], dim=1
        )
        fused_preds.append(fused_batch)

    # Pad fused predictions back to match the original batch shape
    max_predictions = max(f.size(0) for f in fused_preds)  # Find max predictions in a batch
    padded_preds = [
        torch.cat([f, torch.zeros((max_predictions - f.size(0), f.size(1)), device=f.device)], dim=0)
        for f in fused_preds
    ]

    # print("torch.stack(padded_preds, dim=0): ", torch.stack(padded_preds, dim=0).shape)
    return torch.stack(padded_preds, dim=0)
