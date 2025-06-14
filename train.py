"""
Simple one-click ViT finetune script for VS Code
------------------------------------------------
▶ 모든 경로·파라미터를 **코드 상단 변수**로 설정해 두었으니, 
  *Run ▶* / F5 버튼만 눌러 바로 학습을 실행할 수 있습니다.

⚙️ 수정할 부분은 “USER CONFIG” 섹션의 값만 바꿔 주세요.
"""

# ============================================================
# USER CONFIG – 여기만 고치면 됩니다 🛠️
# ============================================================

import time

TRAIN_DIR   = "./data/train"
VAL_DIR     = "./data/val"
MODEL_PATH  = "./models/vit"      # 디렉터리 or 허브 이름

IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 100
LR          = 3e-4
NUM_WORKERS = 2
SEED        = 42
BALANCED    = True  # WeightedRandomSampler 사용
USE_HEAD    = True  # AutoModelForImageClassification 사용 여부

# ============================================================
# LIBRARIES
# ============================================================

import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
)
from tqdm.auto import tqdm
# ============================================================
# 0. 유틸: 시드 고정 & 장치 결정
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. 전처리 & 데이터로더
# ============================================================

def build_transforms(img_size: int, processor):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(processor.image_mean, processor.image_std),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(processor.image_mean, processor.image_std),
    ])
    return train_tfms, val_tfms


def make_balanced_loader(dataset, batch_size, num_workers=4):
    class_counts = np.bincount([y for _, y in dataset])
    weights = 1.0 / class_counts
    sample_weights = [weights[label] for _, label in dataset]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True)


def get_dataloaders(train_dir: str, val_dir: str, processor, img_size: int,
                    batch_size: int, num_workers: int, balanced: bool):
    train_tfms, val_tfms = build_transforms(img_size, processor)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds  = datasets.ImageFolder(val_dir,  transform=val_tfms)
    val_ds.class_to_idx = train_ds.class_to_idx
    if balanced:
        train_dl = make_balanced_loader(train_ds, batch_size, num_workers)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, len(train_ds.classes)

# ============================================================
# 2. 모델 로드
# ============================================================
from transformers import BitsAndBytesConfig
def get_model(model_path: str, num_labels: int, device, use_head: bool = True):
    if use_head:
        model = AutoModelForImageClassification.from_pretrained(
            model_path, num_labels=num_labels, ignore_mismatched_sizes=True)
    else:
        backbone  = AutoModel.from_pretrained(model_path)
        hidden    = backbone.config.hidden_size
        classifier = nn.Linear(hidden, num_labels)
        model = torch.nn.Sequential(backbone, nn.Flatten(1), classifier)
    return model

# ============================================================
# 3. 학습 & 평가 루프
# ============================================================
from torch.cuda.amp import autocast, GradScaler
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = running_correct = 0
    pbar = tqdm(loader, desc=f"[E{epoch}] Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()

        running_loss   += loss.item() * images.size(0)
        running_correct += (logits.argmax(1) == labels).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        r_mem = torch.cuda.max_memory_reserved(device) / 1024**2
        mem_info = f"│ peak {peak_mem:.0f}MiB, reserve {r_mem:.0f}MiB"
        print(mem_info)
    n = len(loader.dataset)
    return running_loss / n, running_correct / n



def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = running_correct = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[E{epoch}] Val", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)
            running_loss   += loss.item() * images.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
    n = len(loader.dataset)
    return running_loss / n, running_correct / n

# ============================================================
# 4. 메인
# ============================================================



def plan_to_pruned_idx_map(pruning_plan):
    pruned_map = {}
    for entry in pruning_plan:
        key = f"{entry['module_path']}.{entry['param_name']}"
        pruned_map[key] = entry['removed_idxs']
    return pruned_map


import torch_pruning as tp
from collections import defaultdict
import torch_pruning as tp
from collections import defaultdict
def cut(model, num):
    c = [688, 536, 384, 224, 72]
    cu = [0.1, 0.3, 0.5, 0.7, 0.9]

    imp = tp.importance.GroupMagnitudeImportance(p=2)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 101:
            ignored_layers.append(m)

    channel_groups = {}
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            channel_groups[m] = m.num_heads

    pruner = tp.pruner.BasePruner(
        model = model,
        example_inputs = torch.randn(1,3,224,224),
        importance = imp,
        pruning_ratio = cu[num],
        ignored_layers = ignored_layers, # <- 이 2가지 조건
        round_to=8,
        prune_num_heads=True,
        prune_head_dims=False,
        isomorphic=True,
    )

    pruning_plan = []
    for group in pruner.step(interactive=True):
    # group: List[(dep, idxs)]
        for dep, idxs in group:
            M = dep.target.module
            # 2-a) 우리가 관심있는 모듈만 골르기
            if not isinstance(M, (nn.Linear, nn.Conv2d)):
                continue

            # 2-b) 모듈 경로 찾기 (named_modules에서 객체 비교)
            module_path = None
            for name, m in model.named_modules():
                if m is M:
                    module_path = name
                    break

            # 2-c) param_name은 weight로 고정 (bias는 보통 프루닝 안 함)
            param_name = "weight"
            removed = idxs.tolist() if isinstance(idxs, torch.Tensor) else idxs

            pruning_plan.append({
                "module_path":  module_path or M.__class__.__name__,
                "param_name":   param_name,
                "dim":          dep.target.pruning_dim,
                "removed_idxs": removed,
            })
        # 3) 실제 프루닝 수행
        group.prune()


    cfg = model.config
    cfg.hidden_size = c[num]              # pruning 뒤 실제 hidden dim
    cfg.intermediate_size = c[num] * 4    # FFN 차원도 맞춰 줌
    old_pos = model.vit.embeddings.position_embeddings
    n_pos = old_pos.size(1)            # 보통 197 (=1+196)
    model.vit.embeddings.position_embeddings = nn.Parameter(
        old_pos[:, :, :c[num]].clone())   # 768→384 앞부분만 복사(또는 평균)

    # CLS 토큰도 동일
    old_cls = model.vit.embeddings.cls_token     # [1,1,768]
    model.vit.embeddings.cls_token = nn.Parameter(
        old_cls[:, :, :c[num]].clone())

    result = {
        f"{item['module_path']}.{item['param_name']}": item['removed_idxs']
        for item in pruning_plan
    }

    return model, result




def main():
    progress_start = time.time()
    set_seed(SEED)
    device = get_device(); print("Device =", device)

    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    train_dl, val_dl, num_labels = get_dataloaders(
        TRAIN_DIR, VAL_DIR, processor, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, BALANCED)


    model = get_model(MODEL_PATH, num_labels, device, use_head=USE_HEAD)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    progress_end = time.time() - progress_start
    print(f"{progress_end//60}min {(progress_end%60)//1}sec")

    epoch_end = 0
    val_check = 0
    check_val = 0
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device, epoch)
        va_loss, va_acc = evaluate(model, val_dl, criterion, device, epoch)
        scheduler.step()

        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        mem_info = f"│ peak {peak_mem:.0f}MiB"

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}, "
              f"val_loss={va_loss:.4f}, val_acc={va_acc:.4f}\n"
              f"{epoch_time//60}min {(epoch_time%60)//1}sec, {mem_info}")

        if val_check > va_acc:
            if check_val == 2:
                break
            check_val += 1
        else:
            val_check = va_acc
            check_val = 0

    print(f"Training finished. "
          f"{(epoch_end - progress_start)//60} min"
          f"{((epoch_end - progress_start)%60)//1} sec passed"
          f"│ peak {peak_mem:.0f}MiB")

# ============================================================
# 5. ONE-CLICK ENTRYPOINT (F5/Run ▶)
# ============================================================

if __name__ == "__main__":
    main()