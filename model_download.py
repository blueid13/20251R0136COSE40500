from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/vit-base-patch16-224",   # 모델 이름(or 사용자/레포)
    local_dir="./models/vit",                # 내려받을 위치(프로젝트 내부)
    revision="main",                         # 태그/브랜치/커밋‧SHA
    resume_download=True,                    # 끊겼을 때 이어받기
    local_dir_use_symlinks=False             # 실제 파일 복사(권장)
)
print("download clear")