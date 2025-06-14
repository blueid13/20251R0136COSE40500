#!/usr/bin/env python3
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ─── 공통 설정 ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_NAME = "distilbert-base-uncased"
NUM_CLIENTS = 10
ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 5e-5

# ─── 실험 분할 모드 설정: "iid" 또는 "noniid" ─────────────────────────────────────────────────
SPLIT_MODE = "iid"  # "iid" 또는 "noniid"
DIRICHLET_ALPHA = 0.5  # non-iid 시 Dirichlet 분포 파라미터

# ─── 데이터 준비 (MNLI) ─────────────────────────────────────────────────────────────
raw = load_dataset("glue", "mnli")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(ex):
    return tokenizer(
        ex["premise"],
        ex["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


# map across all splits: train, validation_matched, validation_mismatched
tokenized = raw.map(tokenize_fn, batched=True)
# remove 원본 텍스트 컬럼, rename label → labels, set torch 포맷
tokenized = tokenized.remove_columns(["premise", "hypothesis", "idx"]).rename_column(
    "label", "labels"
)
tokenized.set_format("torch")


def split_iid(ds, n):
    size = len(ds) // n
    rest = ds
    splits = []
    for _ in range(n - 1):
        sp = rest.train_test_split(test_size=size, seed=SEED)
        splits.append(sp["test"])
        rest = sp["train"]
    splits.append(rest)
    return splits


def split_dirichlet(ds, n, alpha):
    labels = np.array(ds["labels"])
    idx_by_label = {}
    for idx, label in enumerate(labels):
        idx_by_label.setdefault(int(label), []).append(idx)

    client_idxs = [[] for _ in range(n)]
    for label, idxs in idx_by_label.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(alpha * np.ones(n))
        cuts = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(np.array(idxs), cuts)
        for client_id, split in enumerate(splits):
            client_idxs[client_id].extend(split.tolist())

    return [ds.select(idxs) for idxs in client_idxs]


# 클라이언트별 데이터 분할 (train split 만 사용)
if SPLIT_MODE == "iid":
    client_datasets = split_iid(tokenized["train"], NUM_CLIENTS)
else:
    client_datasets = split_dirichlet(tokenized["train"], NUM_CLIENTS, DIRICHLET_ALPHA)

# 평가에는 validation_matched 사용
eval_dataset = tokenized["validation_matched"]
data_collator = DataCollatorWithPadding(tokenizer)


# ─── 평가 지표 함수 ───────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).astype(float).mean().item()}


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


# ─── Baseline (고사양만) ─────────────────────────────────────────────────────
def federated_baseline(high_ids):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    for _ in range(ROUNDS):
        states, sizes = [], []
        for cid in high_ids:
            local = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=3
            )
            local.load_state_dict(model.state_dict())
            Trainer(
                model=local,
                args=TrainingArguments(
                    output_dir=f"./tmp/base_c{cid}",
                    per_device_train_batch_size=BATCH_SIZE,
                    num_train_epochs=LOCAL_EPOCHS,
                    learning_rate=LR,
                    seed=SEED,
                ),
                train_dataset=client_datasets[cid],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            ).train()
            states.append(local.state_dict())
            sizes.append(len(client_datasets[cid]))
        model.load_state_dict(weighted_average(states, sizes, high_ids))
    return Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp/eval_base",
            per_device_eval_batch_size=BATCH_SIZE,
            disable_tqdm=True,
        ),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    ).evaluate(eval_dataset)["eval_accuracy"]


# ─── Adapter-FL ───────────────────────────────────────────────────────────────
def federated_adapter_fl(high_ids):
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    )
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model = get_peft_model(base, lora_cfg)

    for _ in range(ROUNDS):
        states, sizes = [], []
        for cid in range(NUM_CLIENTS):
            local_base = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=3
            )
            local = get_peft_model(local_base, lora_cfg)
            local.load_state_dict(model.state_dict(), strict=True)
            for n, p in local.named_parameters():
                p.requires_grad = ("lora_" in n) or (cid in high_ids)
            Trainer(
                model=local,
                args=TrainingArguments(
                    output_dir=f"./tmp/ada_c{cid}",
                    per_device_train_batch_size=BATCH_SIZE,
                    num_train_epochs=LOCAL_EPOCHS,
                    learning_rate=LR,
                    seed=SEED,
                ),
                train_dataset=client_datasets[cid],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            ).train()
            states.append(local.state_dict())
            sizes.append(len(client_datasets[cid]))
        new_state = {}
        base_avg = weighted_average(states, sizes, high_ids)
        full_avg = weighted_average(states, sizes)
        for k in model.state_dict().keys():
            if k.startswith("base_model") and "lora_" not in k:
                new_state[k] = base_avg[k]
            else:
                new_state[k] = full_avg[k]
        model.load_state_dict(new_state)
    return Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp/eval_ada",
            per_device_eval_batch_size=BATCH_SIZE,
            disable_tqdm=True,
        ),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    ).evaluate(eval_dataset)["eval_accuracy"]


# ─── LoRA-only Baseline (모든 장치 LoRA 파인튜닝) ─────────────────────────────────────
def federated_lora_baseline():
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    )
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model = get_peft_model(base, lora_cfg)
    for n, p in model.named_parameters():
        print(n)
        if "lora_" not in n:
            p.requires_grad = False

    for _ in range(ROUNDS):
        states, sizes = [], []
        for cid in range(NUM_CLIENTS):
            local_base = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=3
            )
            local = get_peft_model(local_base, lora_cfg)
            local.load_state_dict(model.state_dict(), strict=True)
            for n, p in local.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False
            Trainer(
                model=local,
                args=TrainingArguments(
                    output_dir=f"./tmp/lora_c{cid}",
                    per_device_train_batch_size=BATCH_SIZE,
                    num_train_epochs=LOCAL_EPOCHS,
                    learning_rate=LR,
                    seed=SEED,
                ),
                train_dataset=client_datasets[cid],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            ).train()
            states.append(local.state_dict())
            sizes.append(len(client_datasets[cid]))
        global_state = model.state_dict()
        loRA_avg = weighted_average(states, sizes)
        for k, v in loRA_avg.items():
            if "lora_" in k:
                global_state[k] = v
        model.load_state_dict(global_state)

    return Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp/eval_lora",
            per_device_eval_batch_size=BATCH_SIZE,
            disable_tqdm=True,
        ),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    ).evaluate(eval_dataset)["eval_accuracy"]


# ─── 실험 루프 ────────────────────────────────────────────────────────────────
results = []
acc_lora = federated_lora_baseline()

for high_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
    k = int(NUM_CLIENTS * high_ratio)
    high_ids = list(range(k))

    acc_base = federated_baseline(high_ids)
    acc_adapter = federated_adapter_fl(high_ids)

    results.append(
        {
            "high_ratio": high_ratio,
            "n_high": k,
            "acc_baseline": acc_base,
            "acc_adapter_fl": acc_adapter,
            "acc_lora_baseline": acc_lora,
        }
    )
    print(
        f"[ratio={high_ratio:.1f}] base={acc_base:.4f}, adapter={acc_adapter:.4f}, lora_base={acc_lora:.4f}"
    )

df = pd.DataFrame(results)
print(df)