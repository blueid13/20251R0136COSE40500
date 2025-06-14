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

TRAIN_DIR   = "/testdata/train"
VAL_DIR     = "/testdata/val"
MODEL_PATH  = "./models/vit"      # 디렉터리 or 허브 이름

NUM_CLIENTS = 10
IMG_SIZE    = 224
BATCH_SIZE  = 16
LOCAL_EPOCHS = 1
ROUNDS = 20
LR          = 3e-4
NUM_WORKERS = 2
SEED        = 42
BALANCED    = False  # WeightedRandomSampler 사용
USE_HEAD    = True  # AutoModelForImageClassification 사용 여부


freeze_ratio_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
struggler_list = [0.1, 0.3, 0.5, 0.7, 0.9]



SPLIT_MODE = "iid"
DIRICHLET_ALPHA = 0.1

# ============================================================
# LIBRARIES
# ============================================================

import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
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

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── 가중 평균 함수 (변경 없음) ────────────────────────────────────────────────────
def weighted_average(states, sizes, client_ids=None):
    if client_ids is None:
        client_ids = list(range(len(states)))
    total = sum(sizes[i] for i in client_ids)
    avg = {}
    for k in states[0]:
        agg = torch.zeros_like(states[0][k], dtype=torch.float)
        for i in client_ids:
            agg += states[i][k].float() * (sizes[i] / total)
        avg[k] = agg
    return avg



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


def split_iid_img(dataset, n, seed=42):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(dataset))
    rng.shuffle(idxs)
    size = len(dataset) // n
    return [Subset(dataset, idxs[i*size:(i+1)*size]) for i in range(n)]

def split_dirichlet_img(dataset, n, alpha, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)          # ImageFolder가 갖고 있는 라벨 리스트
    idx_by_label = {}
    for idx, lab in enumerate(labels):
        idx_by_label.setdefault(lab, []).append(idx)

    client_idxs = [[] for _ in range(n)]
    for lab, idxs in idx_by_label.items():
        rng.shuffle(idxs)
        props = rng.dirichlet(alpha * np.ones(n))
        cuts = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, cuts)
        for cid, part in enumerate(splits):
            client_idxs[cid].extend(part.tolist())

    return [Subset(dataset, idxs) for idxs in client_idxs]



def get_dataloaders(train_dir: str, val_dir: str, processor, img_size: int,
                    batch_size: int, num_workers: int, balanced: bool):


    train_tfms, val_tfms = build_transforms(img_size, processor)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds  = datasets.ImageFolder(val_dir,  transform=val_tfms)
    val_ds.class_to_idx = train_ds.class_to_idx


    # 클라이언트별 데이터 분할 (train split 만 사용)
    if SPLIT_MODE == "iid":
        client_datasets = split_iid_img(train_ds, NUM_CLIENTS)
    else:
        client_datasets = split_dirichlet_img(train_ds, NUM_CLIENTS, DIRICHLET_ALPHA)

    client_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        for ds in client_datasets
    ]

    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return client_loaders, val_dl, len(train_ds.classes)

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

# ============================================================
# 3. 학습 & 평가 루프
# ============================================================

def freeze_by_param_percent(model, percent):
    total = sum(p.numel() for p in model.parameters())
    frozen = 0

    for p in model.vit.embeddings.parameters():
        if not percent == 0:
            p.requires_grad = False
            frozen += p.numel()
    for block in model.vit.encoder.layer:
        if frozen / total * 100 >= percent:
            break
        for p in block.parameters():
            p.requires_grad = False
        frozen += sum(p.numel() for p in block.parameters())


def fedavg(states, sizes, keylists):
    """states: List[state_dict], sizes: List[int]"""
    total_sizes = {}          # 키별 총 데이터 수
    agg = {}

    # ① 초기화
    for k in states[0]:
        agg[k] = torch.zeros_like(states[0][k], dtype=torch.float32)
        total_sizes[k] = 0

    # ② 기여할 때만 누적
    for s, sz, keys in zip(states, sizes, keylists):
        for k in keys:                     # 학습된 키만
            agg[k] += s[k].float() * sz
            total_sizes[k] += sz

    # ③ 가중 평균
    avg = {}
    for k in agg:
        if total_sizes[k] > 0:             # 학습된 적 있으면 평준화
            avg[k] = agg[k] / total_sizes[k]
        else:                              # 전원 프리징 → 값 그대로
            avg[k] = states[0][k]          # 혹은 아무 클라이언트 값

    return avg


import copy
def train_one_client(model, loader, criterion, optimizer, device, client):

    model.train()
    pbar = tqdm(loader, desc=f"[E{client}] Train", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()

    state = {k: v.cpu() for k, v in model.state_dict().items()}
    size  = len(loader.dataset)
    trainable_keys = [k for k, p in model.named_parameters() if p.requires_grad]

    return state, size, trainable_keys


def train_one_round(model, client_loaders, criterion, device, struggler, freeze_ratio, client = NUM_CLIENTS):

    high_limit = client - struggler
    states, sizes, keylists = [], [], []

    for cid, loader in enumerate(client_loaders):
        local = copy.deepcopy(model)
        local.to(device)
        if cid >= high_limit:
            freeze_by_param_percent(local, freeze_ratio)

        opt = optim.AdamW(filter(lambda p: p.requires_grad,
                                 local.parameters()), lr=LR)
        torch.cuda.reset_peak_memory_stats(device)
        state, size, keys = train_one_client(
            local, loader, criterion, opt, device, cid)

        states.append(state); sizes.append(size); keylists.append(keys)
        del local

        # GPU 메모리 즉시 해제


    # ④ FedAvg 집계 → 글로벌 파라미터 갱신
    new_state = fedavg(states, sizes, keylists)
    model.load_state_dict(new_state)
    return model


def evaluate(model, loader, criterion, device, rnd):
    model.eval()
    running_loss = running_correct = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[E{rnd}] Val", leave=False)
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


def main(freeze_ratio, struggler):
    device = get_device(); print("Device =", device)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    client_loaders, val_dl, num_labels = get_dataloaders(
            TRAIN_DIR, VAL_DIR, processor, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, BALANCED)

    model = get_model(MODEL_PATH, num_labels, device, use_head=USE_HEAD).to(device)

    criterion = nn.CrossEntropyLoss()

    val_check = 0
    check_val = 0

    vlist = []

    for rnd in range(1, ROUNDS + 1):

        round_start = time.time()

        model = train_one_round(
            model, client_loaders, criterion,
            device, struggler, freeze_ratio, client = NUM_CLIENTS)


        _, va_acc = evaluate(model, val_dl, criterion, device, rnd)

        round_end = time.time()
        round_time = round_end - round_start
        print(f"[ROUNDS {rnd}/{ROUNDS}] "
              f"val_acc={va_acc:.4f}, "
              f"{round_time//60}min {(round_time%60)//1}sec")
        vlist.append(va_acc)
        if val_check > va_acc:
            if check_val == 1:
                break
            check_val = 1
        else:
            val_check = va_acc
            check_val = 0
    print(f"max acc is {val_check}")

    for i in vlist:
        print(i, end=', ')
    print(f"\nTraining finished. ratio {freeze_ratio}, weak {struggler}")



if __name__ == "__main__":
    for i in range(4):
        for j in range(6):
            struggler = int(struggler_list[i] * NUM_CLIENTS)
            freeze_ratio = freeze_ratio_list[j] * 100
            main(freeze_ratio, struggler)