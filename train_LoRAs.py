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

import loralib as lora
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

def replace_with_lora(model, r, alpha, target_modules=("query","key","value")):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            lora_mod = lora.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=r,
                lora_alpha=alpha,
                merge_weights=False
            )
            # W, bias 복사
            lora_mod.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_mod.bias.data = module.bias.data.clone()
            # 교체
            parent_path, _, attr_name = name.rpartition(".")
            parent = model
            if parent_path:
                for part in parent_path.split("."):
                    if part.isdigit():               # 숫자면 리스트 인덱싱
                        parent = parent[int(part)]
                    else:                             # 아니면 속성 접근
                        parent = getattr(parent, part)
            # 4) 실제 교체
            setattr(parent, attr_name, lora_mod)
    lora.mark_only_lora_as_trainable(model)

# ============================================================
# 3. 학습 & 평가 루프
# ============================================================

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

def train(train_dl, val_dl, num_labels):
    start_t = time.time()
    set_seed(SEED)
    device = get_device(); print("Device =", device)

    

    model = get_model(MODEL_PATH, num_labels, device, use_head=USE_HEAD)

    replace_with_lora(model, r=R, alpha = R)
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    optim_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.AdamW(optim_params, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"{(time.time()-start_t)//60} min {((time.time()-start_t)%60)//1} sec passed")
    val_check = 0
    check_val = 0
    for epoch in range(1, EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device, epoch)
        va_loss, va_acc = evaluate(model, val_dl, criterion, device, epoch)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        mem_info = f"│ peak {peak_mem:.0f}MiB"
        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}, "
              f"val_loss={va_loss:.4f}, val_acc={va_acc:.4f}\n"
              f"│ time {epoch_time//60}min {(epoch_time%60)//1}sec, {mem_info}")
        break

    print("Training finished.")

# ============================================================
# 5. ONE-CLICK ENTRYPOINT (F5/Run ▶)
# ============================================================

processor = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)
train_dl, val_dl, num_labels = get_dataloaders(
    TRAIN_DIR, VAL_DIR, processor, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, BALANCED)



R = 1
train(train_dl, val_dl, num_labels)

R = 2
train(train_dl, val_dl, num_labels)

R = 4
train(train_dl, val_dl, num_labels)

R = 8
train(train_dl, val_dl, num_labels)

R = 16
train(train_dl, val_dl, num_labels)

R = 32
train(train_dl, val_dl, num_labels)

R = 64
train(train_dl, val_dl, num_labels)