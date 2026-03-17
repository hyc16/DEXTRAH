import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.models.sam.predict import SAM2Predictor


def get_im_features_batched(predictor, im):
    with torch.inference_mode():
        backbone_out = predictor.model.forward_image(im)
        _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)

        if predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed

        B = im.shape[0]
        feats = []

        for feat, feat_size in zip(vision_feats, predictor._bb_feat_sizes):
            h, w = feat_size
            # feat: [H*W, B, C] -> [B, C, H, W]
            feat_bchw = feat.permute(1, 2, 0).contiguous().reshape(B, feat.shape[-1], h, w)
            feats.append(feat_bchw)

        return {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }


def load_image_as_tensor(path, size=1024, device="cuda:0"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    img_t = torch.from_numpy(img).to(device=device, dtype=torch.float32) / 255.0
    img_t = img_t.permute(2, 0, 1).contiguous()  # HWC -> CHW
    return img, img_t


def main():
    device = "cuda:0"
    image_path = "dog.jpeg"   # 改成你的图片路径
    B = 8
    size = 1024

    predictor = SAM2Predictor(overrides={
        "model": "sam2.1_l.pt",
        "device": device,
        "imgsz": size,
        "mode": "predict",
        "task": "segment",
    })

    predictor.setup_model(model=None, verbose=False)
    predictor.args.imgsz = size
    predictor.imgsz = (size, size)
    predictor.setup_source(None)

    print("predictor ready")
    print("imgsz:", predictor.imgsz)
    print("_bb_feat_sizes:", predictor._bb_feat_sizes)

    # 1) 读 1 张真实图片
    img_np, img_t = load_image_as_tensor(image_path, size=size, device=device)

    # 2) 复制成 batch
    imgs = img_t.unsqueeze(0).repeat(B, 1, 1, 1).contiguous()
    print("imgs:", imgs.shape, imgs.dtype, imgs.device)

    # 3) batch 特征提取
    features = get_im_features_batched(predictor, imgs)
    print("image_embed:", features["image_embed"].shape)
    for i, feat in enumerate(features["high_res_feats"]):
        print(f"high_res_feats[{i}]:", feat.shape)

    # 4) 给一个固定点 prompt（图中心）
    points = torch.tensor([[[size / 2, size / 2]]], device=device, dtype=torch.float32)  # [1,1,2]
    labels = torch.tensor([[1]], device=device, dtype=torch.int64)                        # [1,1]

    # 5) 对 batch 中每张图逐个 decoder
    masks_all = []
    scores_all = []
    for i in range(B):
        pred_masks, pred_scores = predictor._inference_features(
            features,
            points=points,
            labels=labels,
            masks=None,
            multimask_output=True,
            img_idx=i,
        )
        masks_all.append(pred_masks.detach().cpu())
        scores_all.append(pred_scores.detach().cpu())
        print(f"img_idx={i}: pred_masks={pred_masks.shape}, pred_scores={pred_scores.shape}")

    # 6) 可视化第 0 张图的最佳 mask
    scores0 = scores_all[0]
    masks0 = masks_all[0]
    best_idx = int(torch.argmax(scores0).item())
    best_mask = masks0[best_idx].numpy()  # [256, 256]

    # 上采样回 1024 方便看
    best_mask_up = cv2.resize(best_mask.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.imshow(best_mask_up, alpha=0.5)
    plt.title(f"Best mask overlay, score={scores0[best_idx].item():.4f}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()