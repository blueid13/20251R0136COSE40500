import time

TRAIN_DIR   = "./data/train"
VAL_DIR     = "./data/val"
MODEL_PATH  = "./models/vit"      # 디렉터리 or 허브 이름

NUM_CLIENTS = 10
IMG_SIZE    = 224
BATCH_SIZE  = 16
LOCAL_EPOCHS = 1
ROUNDS = 30
LR          = 3e-4
NUM_WORKERS = 2
SEED        = 42
BALANCED    = False  # WeightedRandomSampler 사용
USE_HEAD    = True  # AutoModelForImageClassification 사용 여부

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

def get_model(model_path: str, num_labels: int, use_head: bool = True):
    if use_head:
        model = AutoModelForImageClassification.from_pretrained(
            model_path, num_labels=num_labels, ignore_mismatched_sizes=True)
    else:
        backbone  = AutoModel.from_pretrained(model_path)
        hidden    = backbone.config.hidden_size
        classifier = nn.Linear(hidden, num_labels)
        model = torch.nn.Sequential(backbone, nn.Flatten(1), classifier)
    return model

from typing import List, Dict
import torch
from typing import List, Dict

def fedavg_mixed_pruning(
    states: List[Dict[str, torch.Tensor]],
    sizes:  List[int],
    axis:   int = 0
) -> Dict[str, torch.Tensor]:
    """
    shape만 보고 full/pruned 모델을 함께 FedAvg 합니다.
    - states: List[state_dict], 첫 번째는 full model, 뒤는 pruned models
    - sizes:  List[int], 데이터 크기 비율
    - axis:   프루닝된 축 (기본 0 = output‐channel)
    """
    # 기준(full) 모델 키·shape
    full_state  = states[0]
    base_keys   = list(full_state.keys())
    base_shapes = {k: full_state[k].shape for k in base_keys}
    device      = full_state[base_keys[0]].device

    # 누적 변수
    agg         = {k: torch.zeros(base_shapes[k], device=device) for k in base_keys}
    total_sizes = {k: 0 for k in base_keys}

    # 각 클라이언트 순회
    for state, sz in zip(states, sizes):
        for k in base_keys:
            if k not in state:
                continue
            w = state[k].float().to(device)

            # 1) full model (shape 같으면 그대로)
            if w.shape == base_shapes[k]:
                w_full = w
            # 2) pruned model (shape 작으면 앞쪽 slice로 패딩)
            else:
                full_shape = base_shapes[k]
                w_full = torch.zeros(full_shape, dtype=torch.float32, device=device)

                # e.g. (out_chan, ...) 일 때: w.shape[axis] 만큼만 복사
                # slices = [slice(None)] * w.ndim
                # slices[axis] = slice(0, w.shape[axis])
                # w_full[tuple(slices)] = w

                # 위를 axis가 0일 때 단순하게 처리
                if axis == 0:
                    w_full[: w.shape[0], ...] = w
                else:
                    w_full[..., : w.shape[-1]] = w

            # 누적
            agg[k]         += w_full * sz
            total_sizes[k] += sz

    # 가중 평균
    avg_state = {}
    for k in base_keys:
        if total_sizes[k] > 0:
            avg_state[k] = agg[k] / total_sizes[k]
        else:
            avg_state[k] = full_state[k].float().to(device)

    return avg_state

def plan_to_pruned_idx_map(pruning_plan):
    pruned_map = {}
    for entry in pruning_plan:
        key = f"{entry['module_path']}.{entry['param_name']}"
        pruned_map[key] = entry['removed_idxs']
    return pruned_map

import copy
import torch_pruning as tp
from collections import defaultdict
def prune(model, example_inputs, pruner, num):

    # 1) 의존성 그래프 빌드


    DG = tp.DependencyGraph()
    DG.build_dependency(model, [example_inputs])

    pruning_plan = []
    # 2) interactive 모드로 그룹 리스트 얻기
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
            

    return model, plan_to_pruned_idx_map(pruning_plan)


def train_normal_client(model, loader, criterion, optimizer, device, client):

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

def cut(model, num):
    imp = tp.importance.GroupNormImportance(p=2)

    c = [0.1, 0.3, 0.5, 0.7, 0.9]


    cls_tok = model.vit.embeddings.cls_token
    pos_emb = model.vit.embeddings.position_embeddings
    unwrapped = [
        (cls_tok, 2),     # cls_token은 hidden 차원이 2번 축
        (pos_emb, 2),     # position_embeddings도 hidden 차원이 2번 축
    ]

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 101:
            ignored_layers.append(m)

    channel_groups = {}
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            channel_groups[m] = m.num_heads


    pruner = tp.pruner.MetaPruner(
        model=model,
        example_inputs=torch.randn(1,3,224,224).to(next(model.parameters()).device),
        importance=imp,
        global_pruning=False,
        pruning_ratio=c[num],  # 예시 값
        iterative_steps=1,
        ignored_layers=ignored_layers,
        channel_groups=channel_groups,
        head_pruning_ratio=c[num],
        unwrapped_parameters=unwrapped,
    )
    model, pruning_plan = prune(model, torch.randn(1,3,224,224), pruner, 1/c[num])
    return model, pruning_plan
def train_one_round(model, client_loaders, criterion, device, struggler, num, client = NUM_CLIENTS):

    high_limit = client - struggler
    states, sizes, pruned_idx_list = [], [], []
    model_cut, pruning_plan = cut(model, num)


    for cid, loader in enumerate(client_loaders):
        if cid >= high_limit:
            local = copy.deepcopy(model_cut)
            opt = optim.AdamW(filter(lambda p: p.requires_grad,
                                 local.parameters()), lr=LR)
            torch.cuda.reset_peak_memory_stats(device)

            state, size, keys = [], [], []

            local.to(device)
            state, size, keys = train_normal_client(
                local, loader, criterion, opt, device, cid)

            states.append(state); sizes.append(size); pruned_idx_list.append(pruning_plan)


        else:
            local = copy.deepcopy(model)

            opt = optim.AdamW(filter(lambda p: p.requires_grad,
                                    local.parameters()), lr=LR)
            torch.cuda.reset_peak_memory_stats(device)

            state, size, keys = [], [], []

            local.to(device)
            state, size, keys = train_normal_client(
                local, loader, criterion, opt, device, cid)

            states.append(state); sizes.append(size); pruned_idx_list.append({})

        del local
        
    # ④ FedAvg 집계 → 글로벌 파라미터 갱신
    new_state = fedavg_mixed_pruning(
        states=states,
        sizes=sizes,
        axis=0           # Linear.weight의 경우 항상 0축(output channel)
    )

    #다른 모델 합치기

    model.load_state_dict(new_state)
    return model


def evaluate(model, loader, criterion, device, rnd):
    model.to(device)
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
    model.to('cpu')
    return running_loss / n, running_correct / n
def main(struggler, num):
    device = get_device(); print("Device =", device)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    client_loaders, val_dl, num_labels = get_dataloaders(
            TRAIN_DIR, VAL_DIR, processor, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, BALANCED)

    model = get_model(MODEL_PATH, num_labels, use_head=USE_HEAD)

    criterion = nn.CrossEntropyLoss()

    val_check = 0
    check_val = 0

    vlist = []

    for rnd in range(1, ROUNDS + 1):

        round_start = time.time()

        model = train_one_round(
            model, client_loaders, criterion,
            device, struggler, num, client = NUM_CLIENTS)


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
    print(f"\nTraining finished. weak {struggler}, {(2*num + 1)*10}% prune")

    
if __name__ == "__main__":
    a = [1, 3, 5, 7, 9]
    b = [0, 1, 2, 3, 4]
    for i in a:
        for j in b:
            main(i, j)