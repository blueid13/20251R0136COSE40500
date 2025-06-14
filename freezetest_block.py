from transformers import AutoModelForImageClassification
import torch

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

def freeze_front_percent(model, percent: float):
    # ① ViT는 encoder.layer가 리스트라 “앞에서부터” 정의상 안전
    n_layers = len(model.vit.encoder.layer)
    k = int(n_layers * percent / 100)

    # ② 임베딩 + 앞 k개 블록만 얼림
    for p in model.vit.embeddings.parameters():
        p.requires_grad = False
    for i, block in enumerate(model.vit.encoder.layer):
        if i < k:
            for p in block.parameters():
                p.requires_grad = False

    # ③ 확인용
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen {frozen/total*100:.2f}% of parameters")

freeze_front_percent(model, percent=40)   # 예: 앞 40 % 고정