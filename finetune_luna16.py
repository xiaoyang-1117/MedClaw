"""
finetune_luna16.py — LUNA16 数据集微调脚本
============================================

独立脚本，不接入 MedClaw Agent 系统。
用于在 LUNA16 数据集上微调 MONAI lung_nodule_ct_detection bundle
的 RetinaNet 3D 检测模型。

数据目录结构 (预期):
    Model/Data/
    ├── annotations.csv          # 结节标注 (seriesuid, coordX/Y/Z, diameter_mm)
    ├── subset0/ ... subset4/    # .mhd + .raw CT 图像文件
    └── seg-lungs-LUNA16/        # 肺分割 mask

使用方式:
    python finetune_luna16.py --data_dir Model/Data --epochs 50 --batch_size 4

技术栈:
    - MONAI: 3D 目标检测训练框架
    - PyTorch: 深度学习训练
    - SimpleITK: (由 MONAI ITKReader 内部调用) 医学图像 I/O
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import monai
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import Dataset, DataLoader
from monai.networks.nets.resnet import resnet50
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    RandCropBoxByPosNegLabeld,
    RandFlipBoxd,
    RandRotateBox90d,
    RandZoomBoxd,
)


# ============================================================
# 配置参数
# ============================================================

# MONAI Bundle 路径 (相对于项目根目录)
DEFAULT_BUNDLE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "monai_bundles",
    "lung_nodule_ct_detection",
)

# 训练超参数
PATCH_SIZE = [192, 192, 80]       # 训练 patch 尺寸
VAL_PATCH_SIZE = [512, 512, 192]  # 验证 patch 尺寸
LEARNING_RATE = 0.01
WEIGHT_DECAY = 3e-5


# ============================================================
# 数据准备: 读取 LUNA16 标注 + 构建 MONAI 格式数据列表
# ============================================================

def parse_annotations(csv_path: str) -> Dict[str, List[dict]]:
    """解析 LUNA16 annotations.csv 文件。

    CSV 格式: seriesuid, coordX, coordY, coordZ, diameter_mm
    注意: annotations.csv 中的坐标是 **世界坐标 (World Coordinate)**，
    单位为 mm。MONAI 训练时需要将其转换为体素坐标 (Image Coordinate)，
    由 AffineBoxToImageCoordinated 变换自动完成。

    Returns:
        dict: {seriesuid: [{coordX, coordY, coordZ, diameter_mm}, ...]}
    """
    annotations = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["seriesuid"]
            annotations[uid].append({
                "coordX": float(row["coordX"]),
                "coordY": float(row["coordY"]),
                "coordZ": float(row["coordZ"]),
                "diameter_mm": float(row["diameter_mm"]),
            })
    print(f"[Data] 读取标注: {len(annotations)} 个 Series, "
          f"{sum(len(v) for v in annotations.values())} 个结节")
    return dict(annotations)


def build_data_list(
    data_dir: str,
    annotations: Dict[str, List[dict]],
) -> List[dict]:
    """构建 MONAI 训练所需的数据列表。

    数据格式 (MONAI Detection):
        {
            "image": "path/to/image.mhd",
            "box": [[x1,y1,z1,x2,y2,z2], ...],   # cccwhd 格式的世界坐标框
            "label": [0, 0, ...],                   # 类别标签 (0=nodule)
        }

    注意: 我们使用 cccwhd (cx, cy, cz, w, h, d) 格式存储标注框：
        - cx, cy, cz: 结节中心的世界坐标 (mm)
        - w, h, d: 结节在三个轴上的大小，近似为 diameter_mm
    """
    data_list = []

    # 在所有 subset 目录中搜索 .mhd 文件
    mhd_files = sorted(glob.glob(os.path.join(data_dir, "subset*", "*.mhd")))
    print(f"[Data] 找到 {len(mhd_files)} 个 .mhd 文件")

    matched = 0
    for mhd_path in mhd_files:
        # 从文件名提取 SeriesUID (去掉 .mhd 后缀)
        series_uid = Path(mhd_path).stem

        if series_uid not in annotations:
            continue  # 跳过没有标注的图像

        matched += 1
        nodules = annotations[series_uid]

        # 将标注转换为 cccwhd 格式的 box 列表
        # cccwhd: [center_x, center_y, center_z, width, height, depth]
        boxes = []
        labels = []
        for nodule in nodules:
            d = nodule["diameter_mm"]
            boxes.append([
                nodule["coordX"],
                nodule["coordY"],
                nodule["coordZ"],
                d,  # width ≈ diameter
                d,  # height ≈ diameter
                d,  # depth ≈ diameter
            ])
            labels.append(0)  # 单类别: nodule = 0

        data_list.append({
            "image": mhd_path,
            "box": np.array(boxes, dtype=np.float32),
            "label": np.array(labels, dtype=np.int64),
        })

    print(f"[Data] 匹配到 {matched} 个带标注的图像")
    return data_list


# ============================================================
# 构建模型
# ============================================================

def build_detector(device: torch.device) -> Tuple[RetinaNet, RetinaNetDetector]:
    """构建 RetinaNet 3D 检测器。

    网络架构 (与 MONAI Bundle 配置一致):
        - 骨干: ResNet50 (3D, 单通道输入)
        - 特征提取: FPN (Feature Pyramid Network), 层级 [1, 2]
        - 检测头: RetinaNet, 3 个 anchor 形状 / 每个特征层级
        - Anchor 形状: [6,8,4], [8,6,5], [10,10,6] (mm)
    """
    # 骨干网络
    backbone = resnet50(
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_stride=[2, 2, 1],
        conv1_t_size=[7, 7, 7],
    )

    # 特征金字塔网络
    feature_extractor = resnet_fpn_feature_extractor(
        backbone, 3, False, [1, 2], None
    )

    # RetinaNet 检测网络
    network = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=3,
        feature_extractor=feature_extractor,
        size_divisible=[16, 16, 8],
    ).to(device)

    # Anchor 生成器
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[1, 2, 4],
        base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    )

    # 检测器 (封装网络 + anchor + 后处理)
    detector = RetinaNetDetector(
        network=network,
        anchor_generator=anchor_generator,
        debug=False,
    )

    # 训练时的匹配与采样策略
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=0.3,
        pool_size=20,
        min_neg=16,
    )
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=0.02,
        topk_candidates_per_level=1000,
        nms_thresh=0.22,
        detections_per_img=300,
    )
    detector.set_sliding_window_inferer(
        roi_size=VAL_PATCH_SIZE,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device="cpu",
    )

    return network, detector


# ============================================================
# 构建数据增强 Pipeline
# ============================================================

def build_train_transforms() -> Compose:
    """构建训练时的数据预处理与增强变换。

    Pipeline:
        1. LoadImaged        — 使用 ITK Reader 加载 .mhd 文件
        2. EnsureChannelFirst — 添加通道维度 [H,W,D] → [1,H,W,D]
        3. EnsureTyped       — 确保 tensor 类型
        4. Orientationd      — 统一为 RAS 方向
        5. ScaleIntensityRange — HU 窗值归一化 [-1024, 300] → [0, 1]
        6. ConvertBoxToStandardMode — 标准化 box 格式
        7. AffineBoxToImageCoordinated — 世界坐标 → 体素坐标
        8. RandCropBoxByPosNegLabel — 正负样本平衡裁剪
        9. RandZoomBoxd       — 随机缩放
        10. ClipBoxToImaged   — 裁剪越界框
        11. RandFlipBoxd      — 三轴随机翻转
        12. RandRotateBox90d  — 随机 90° 旋转
    """
    preprocessing = [
        # 使用 ITK Reader 加载 .mhd 文件，读取完整 affine 信息
        LoadImaged(keys="image", reader="itkreader", affine_lps_to_ras=True),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "box"]),
        EnsureTyped(keys="label", dtype=torch.long),
        Orientationd(keys="image", axcodes="RAS"),
        # HU 窗值归一化: 肺窗 [-1024, 300] → [0, 1]
        ScaleIntensityRanged(
            keys="image",
            a_min=-1024.0,
            a_max=300.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # 将 box 格式统一为 MONAI 内部标准格式 (cccwhd → 标准)
        ConvertBoxToStandardModed(box_keys="box", mode="cccwhd"),
        # 世界坐标 → 体素坐标 (关键步骤)
        # 利用图像的仿射矩阵将标注框从物理空间转换到图像空间
        AffineBoxToImageCoordinated(
            box_keys="box",
            box_ref_image_keys="image",
            affine_lps_to_ras=True,
        ),
    ]

    augmentation = [
        # 以正负样本 1:1 的比例随机裁剪 patch
        RandCropBoxByPosNegLabeld(
            image_keys="image",
            box_keys="box",
            label_keys="label",
            spatial_size=PATCH_SIZE,
            whole_box=True,
            num_samples=4,  # 每张图产生 4 个 patch
            pos=1,
            neg=1,
        ),
        # 随机缩放 (20% 概率)
        RandZoomBoxd(
            image_keys="image",
            box_keys="box",
            label_keys="label",
            box_ref_image_keys="image",
            prob=0.2,
            min_zoom=0.7,
            max_zoom=1.4,
            padding_mode="constant",
            keep_size=True,
        ),
        # 裁剪掉超出图像边界的框
        ClipBoxToImaged(
            box_keys="box",
            label_keys="label",
            box_ref_image_keys="image",
            remove_empty=True,
        ),
        # 三轴随机翻转 (各 50% 概率)
        RandFlipBoxd(
            image_keys="image",
            box_keys="box",
            box_ref_image_keys="image",
            prob=0.5,
            spatial_axis=0,
        ),
        RandFlipBoxd(
            image_keys="image",
            box_keys="box",
            box_ref_image_keys="image",
            prob=0.5,
            spatial_axis=1,
        ),
        RandFlipBoxd(
            image_keys="image",
            box_keys="box",
            box_ref_image_keys="image",
            prob=0.5,
            spatial_axis=2,
        ),
        # 随机 90° 旋转 (75% 概率)
        RandRotateBox90d(
            image_keys="image",
            box_keys="box",
            box_ref_image_keys="image",
            prob=0.75,
            max_k=3,
            spatial_axes=(0, 1),
        ),
    ]

    final = [
        EnsureTyped(keys=["image", "box"]),
        EnsureTyped(keys="label", dtype=torch.long),
    ]

    return Compose(preprocessing + augmentation + final)


def build_val_transforms() -> Compose:
    """构建验证时的预处理变换（不含数据增强）。"""
    return Compose([
        LoadImaged(keys="image", reader="itkreader", affine_lps_to_ras=True),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "box"]),
        EnsureTyped(keys="label", dtype=torch.long),
        Orientationd(keys="image", axcodes="RAS"),
        ScaleIntensityRanged(
            keys="image",
            a_min=-1024.0,
            a_max=300.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ConvertBoxToStandardModed(box_keys="box", mode="cccwhd"),
        AffineBoxToImageCoordinated(
            box_keys="box",
            box_ref_image_keys="image",
            affine_lps_to_ras=True,
        ),
        EnsureTyped(keys=["image", "box"]),
        EnsureTyped(keys="label", dtype=torch.long),
    ])


# ============================================================
# 训练循环
# ============================================================

def train(
    data_dir: str,
    bundle_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_interval: int,
    resume: bool,
) -> None:
    """执行 LUNA16 微调训练。

    Args:
        data_dir:     LUNA16 数据根目录 (包含 subset*/ 和 annotations.csv)。
        bundle_root:  MONAI bundle 根目录 (包含 models/model.pt)。
        epochs:       训练轮数。
        batch_size:   每批图像数量 (注意: 每张图会裁剪出多个 patch)。
        lr:           初始学习率。
        val_interval: 每隔多少轮验证一次。
        resume:       是否从预训练权重继续训练。
    """
    # 设备检测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"LUNA16 Fine-tuning — MedClaw")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"数据目录: {data_dir}")
    print(f"训练轮数: {epochs}")
    print(f"批大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"{'='*60}\n")

    # 确保可复现
    monai.utils.set_determinism(seed=42)

    # ---------- 1. 准备数据 ----------
    annotations_path = os.path.join(data_dir, "annotations.csv")
    if not os.path.exists(annotations_path):
        print(f"❌ 未找到标注文件: {annotations_path}")
        sys.exit(1)

    annotations = parse_annotations(annotations_path)
    data_list = build_data_list(data_dir, annotations)

    if len(data_list) == 0:
        print("❌ 没有找到匹配的数据！请检查路径和文件名。")
        sys.exit(1)

    # 划分训练集和验证集 (95% / 5%)
    split_idx = int(0.95 * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]
    print(f"[Data] 训练集: {len(train_data)}, 验证集: {len(val_data)}")

    # ---------- 2. 构建 DataLoader ----------
    train_transforms = build_train_transforms()
    val_transforms = build_val_transforms()

    train_dataset = Dataset(data=train_data, transform=train_transforms)
    val_dataset = Dataset(data=val_data, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # 由于 RandCropBox 已在内部生成 batch
        shuffle=True,
        num_workers=4,
        collate_fn=monai.data.utils.no_collation,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=monai.data.utils.no_collation,
    )

    # ---------- 3. 构建模型 ----------
    network, detector = build_detector(device)

    # 加载预训练权重
    if resume:
        pretrained_path = os.path.join(bundle_root, "models", "model.pt")
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(
                pretrained_path, map_location=device, weights_only=False
            )
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                network.load_state_dict(checkpoint["model"])
            else:
                network.load_state_dict(checkpoint)
            print(f"[Model] ✅ 加载预训练权重: {pretrained_path}")
        else:
            print(f"[Model] ⚠️ 未找到预训练权重: {pretrained_path}，从头训练")

    # ---------- 4. 优化器与学习率调度 ----------
    optimizer = torch.optim.SGD(
        detector.network.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(epochs // 3, 1), gamma=0.1
    )

    # ---------- 5. 混合精度训练 ----------
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---------- 6. 输出目录 ----------
    output_dir = os.path.join(bundle_root, "finetuned_models")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 7. 训练循环 ----------
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        network.train()
        epoch_loss = 0.0
        step_count = 0

        for batch_idx, batch_data in enumerate(train_loader):
            # batch_data 是一个列表 (因为 no_collation)
            # 每个元素是一个 dict 或 dict 列表 (来自 RandCropBox)
            if isinstance(batch_data, list) and len(batch_data) > 0:
                data_items = batch_data
            else:
                data_items = [batch_data]

            for data_item in data_items:
                # 如果 RandCropBox 产生了多个 patch，它们会是列表
                if isinstance(data_item, list):
                    patches = data_item
                else:
                    patches = [data_item]

                for patch in patches:
                    if not isinstance(patch, dict):
                        continue

                    image = patch["image"].to(device)
                    box = patch["box"].to(device)
                    label = patch["label"].to(device)

                    # 确保 image 维度正确: [C, H, W, D]
                    if image.dim() == 3:
                        image = image.unsqueeze(0)

                    optimizer.zero_grad()

                    # 前向传播 (训练模式返回 loss dict)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        losses = detector(
                            [image],
                            [{"box": box, "label": label}],
                        )

                        # losses 是 dict: {"cls_loss": ..., "box_reg_loss": ...}
                        if isinstance(losses, dict):
                            loss = sum(losses.values())
                        elif isinstance(losses, (list, tuple)):
                            loss = sum(losses)
                        else:
                            loss = losses

                    # 反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += float(loss)
                    step_count += 1

        avg_loss = epoch_loss / max(step_count, 1)
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Steps: {step_count}"
        )

        # ---- Validation ----
        if epoch % val_interval == 0 or epoch == epochs:
            network.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch_data in val_loader:
                    if isinstance(batch_data, list) and len(batch_data) > 0:
                        data_items = batch_data
                    else:
                        data_items = [batch_data]

                    for data_item in data_items:
                        if not isinstance(data_item, dict):
                            continue

                        image = data_item["image"].to(device)
                        box = data_item["box"].to(device)
                        label = data_item["label"].to(device)

                        if image.dim() == 3:
                            image = image.unsqueeze(0)

                        losses = detector(
                            [image],
                            [{"box": box, "label": label}],
                        )

                        if isinstance(losses, dict):
                            loss = sum(losses.values())
                        elif isinstance(losses, (list, tuple)):
                            loss = sum(losses)
                        else:
                            loss = losses

                        val_loss += float(loss)
                        val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            print(f"  [Validation] Loss: {avg_val_loss:.4f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(output_dir, "best_model.pt")
                torch.save({"model": network.state_dict()}, save_path)
                print(f"  [Checkpoint] 保存最佳模型 → {save_path}")

    # ---------- 8. 保存最终模型 ----------
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save({"model": network.state_dict()}, final_path)
    print(f"\n{'='*60}")
    print(f"训练完成！最终模型: {final_path}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"{'='*60}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LUNA16 微调脚本 — 基于 MONAI lung_nodule_ct_detection Bundle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python finetune_luna16.py --data_dir Model/Data --epochs 50
  python finetune_luna16.py --data_dir Model/Data --epochs 100 --lr 0.005 --resume
        """,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Model/Data",
        help="LUNA16 数据目录路径 (包含 subset*/ 和 annotations.csv)",
    )
    parser.add_argument(
        "--bundle_root",
        type=str,
        default=DEFAULT_BUNDLE_ROOT,
        help="MONAI Bundle 根目录",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数 (默认: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="每张图裁剪的 patch 数 (默认: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"初始学习率 (默认: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=5,
        help="验证间隔 (每隔几个 epoch 进行一次验证, 默认: 5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从预训练权重继续训练",
    )

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        bundle_root=args.bundle_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_interval=args.val_interval,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
