from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

def freeze_by_param_percent(model, percent: float = 40.0):
    total = sum(p.numel() for p in model.parameters())
    frozen = 0

    # 1) 먼저 임베딩을 얼림
    for p in model.vit.embeddings.parameters():
        p.requires_grad = False
        frozen += p.numel()
    # 2) 그 뒤 블록을 앞에서부터 추가로 얼림
    for block in model.vit.encoder.layer:
        if frozen / total * 100 >= percent:
            break
        for p in block.parameters():
            p.requires_grad = False
        frozen += sum(p.numel() for p in block.parameters())

    # 3) 필요하면 헤드까지 포함
    # for p in model.classifier.parameters():
    #     p.requires_grad = False
    #     frozen += p.numel()

    print(f"Frozen {frozen / total * 100:.2f}% of parameters "
          f"({frozen / 1e6:.1f} M / {total / 1e6:.1f} M)")
    return model

model = freeze_by_param_percent(model, percent=40)   # 예: 앞 40 % 고정