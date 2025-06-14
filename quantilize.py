import copy
import torch
from torch import nn, optim

from transformers import AutoModelForImageClassification

MODEL_ID = "google/vit-base-patch16-224"

model_fp32 = AutoModelForImageClassification.from_pretrained(MODEL_ID).to('cuda')

model_fp16 = copy.deepcopy(model_fp32)
model_fp16 = model_fp16.to(dtype=torch.float16).to("cuda")

model_int8  = torch.quantization.quantize_dynamic(
        model_fp32,
        dtype=torch.qint8
    )

# ── 메모리 사용량(바이트) 계산용 헬퍼 ─────────────────────────────
def get_state_dict_size(model):
    """state_dict에 저장된 모든 텐서의 실제 바이트 수 합산"""
    total_bytes = 0
    for v in model.state_dict().values():
        if torch.is_tensor(v):
            total_bytes += v.numel() * v.element_size()
        elif isinstance(v, (list, tuple)):
            # dynamic INT8 Linear의 _packed_params처럼
            # 튜플 안에 텐서가 들어 있는 경우가 있음
            for t in v:
                if torch.is_tensor(t):
                    total_bytes += t.numel() * t.element_size()
    return total_bytes

# ── FP32 / INT8 메모리 비교 ───────────────────────────────────
sz_fp32 = get_state_dict_size(model_fp32)      # 원본
sz_fp16 = get_state_dict_size(model_fp16)
sz_int8 = get_state_dict_size(model_int8)      # 동적 INT8

print(f"FP32 파라미터 메모리 : {sz_fp32/1024**2:.1f} MB")

print(f"FP16 파라미터 메모리 : {sz_fp16/1024**2:.1f} MB  "
      f"(≈ {sz_fp32/sz_fp16:.1f} × 압축)")

print(f"INT8 파라미터 메모리 : {sz_int8/1024**2:.1f} MB  "
      f"(≈ {sz_fp32/sz_int8:.1f} × 압축)")
import torch, copy, bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
from bitsandbytes.functional import quantize_blockwise

device          = "cuda"
compute_dtype   = torch.bfloat16
quant_type      = "nf4"     # or "fp4"
double_quant    = True

# ─────────────────────────────────────────
# 1) Linear → Linear4bit (가중치 패킹 포함)
# ─────────────────────────────────────────
def linear_to_4bit(linear: torch.nn.Linear) -> Linear4bit:
    new = Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        compute_dtype=compute_dtype,
        quant_type=quant_type,
        compress_statistics=double_quant,
    )

    # ① 가중치 4-bit로 블록-양자화
    qweight, qstate = quantize_blockwise(
        linear.weight.data.cpu(),      # FP32 입력
        quant_type=quant_type,
        compress_statistics=double_quant,
    )
    new.weight.data      = qweight.to(torch.int8)  # 4-bit 2개당 int8 1개
    new.weight.quant_state = qstate                # scale/zero-point 등

    # ② bias는 그대로 복사
    if linear.bias is not None:
        new.bias.data = linear.bias.data.clone()

    return new.to(device)

# ─────────────────────────────────────────
# 2) 재귀 변환
# ─────────────────────────────────────────
def convert_vit_to_int4(model_fp32: torch.nn.Module):
    for name, module in model_fp32.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model_fp32, name, linear_to_4bit(module))
        else:
            convert_vit_to_int4(module)
    return model_fp32

# 예) 이미 메모리에 있는 model_fp32 변환
model_int4 = convert_vit_to_int4(copy.deepcopy(model_fp32).cpu()).eval()

# ─────────────────────────────────────────
# 3) 메모리 계산 ─ state_dict 대신 직접
# ─────────────────────────────────────────
def bytes_linear4bit(m: Linear4bit):
    # weight(4-bit → int8) + scale/zero-point(FP16) + bias(optional)
    w = m.weight
    size = w.numel() * w.element_size()            # int8
    if hasattr(w, "quant_state"):                  # (scale, zp) 튜플
        for t in w.quant_state:
            size += t.numel() * t.element_size()   # fp16
    if m.bias is not None:
        size += m.bias.numel() * m.bias.element_size()
    return size

def model_size(model):
    total = 0
    for mod in model.modules():
        if isinstance(mod, Linear4bit):
            total += bytes_linear4bit(mod)
        else:
            for p in mod.parameters(recurse=False):
                total += p.numel() * p.element_size()
    return total

print(f"FP32 파라미터 : {model_size(model_fp32)/1e6:.1f} MB")
print(f"INT4 파라미터 : {model_size(model_int4)/1e6:.1f} MB "
      f"(≈ {model_size(model_fp32)/model_size(model_int4):.1f} × 압축)")

# ─────────────────────────────────────────
# 4) 추론 확인
# ─────────────────────────────────────────
dummy = torch.randn(1, 3, 224, 224, device=device, dtype=compute_dtype)
with torch.no_grad():
    out = model_int4(dummy)
print("4-bit 모델 로짓 shape:", out.logits.shape)
