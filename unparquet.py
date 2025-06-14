import io, base64, pathlib
from tqdm import tqdm
import pyarrow.parquet as pq
from PIL import Image

# ────────────────── 경로 설정 ──────────────────
ROOT_DIR     = pathlib.Path(__file__).resolve().parent   # 실행 파일(.py)의 위치
PARQUET_DIR  = ROOT_DIR / "parquet_t"                     # ./parquet   안에 여러 parquet 파일
OUT_DIR      = ROOT_DIR / "data/train"                        # ./data      에 이미지 풀기
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────── parquet → 이미지 변환 ──────────────────
for pq_path in sorted(PARQUET_DIR.glob("*.parquet")):
    # 1) 테이블 읽기 (필요하면 columns=['image','label'] 지정)
    table = pq.read_table(pq_path)
    df    = table.to_pandas()

    # 2) 이미지·라벨 반복
    for row_idx, (img_field, label) in tqdm(
        enumerate(zip(df["image"], df["label"])),
        total=len(df),
        desc=pq_path.stem                # 진행 막대에 파일 이름 표시
    ):
        # ── 바이트 추출 ───────────────────────────────
        if isinstance(img_field, dict) and "bytes" in img_field:      # {'bytes': b'...', ...}
            byte_data = img_field["bytes"]
        elif isinstance(img_field, str):                              # base64 문자열
            byte_data = base64.b64decode(img_field)
        else:                                                         # 이미 bytes
            byte_data = img_field

        # ── PIL 이미지로 복원 ─────────────────────────
        img = Image.open(io.BytesIO(byte_data)).convert("RGB")

        # ── 저장 ─────────────────────────────────────
        label_dir = OUT_DIR / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)

        # 파일명: parquet 스템 + 행번호 (충돌 방지)
        fn = f"{pq_path.stem}_{row_idx:06d}.jpg"
        img.save(label_dir / fn, "JPEG", quality=95)
        
        
PARQUET_DIR  = ROOT_DIR / "parquet_v"                     # ./parquet   안에 여러 parquet 파일
OUT_DIR      = ROOT_DIR / "data/val"                        # ./data      에 이미지 풀기

# ────────────────── parquet → 이미지 변환 ──────────────────
for pq_path in sorted(PARQUET_DIR.glob("*.parquet")):
    # 1) 테이블 읽기 (필요하면 columns=['image','label'] 지정)
    table = pq.read_table(pq_path)
    df    = table.to_pandas()

    # 2) 이미지·라벨 반복
    for row_idx, (img_field, label) in tqdm(
        enumerate(zip(df["image"], df["label"])),
        total=len(df),
        desc=pq_path.stem                # 진행 막대에 파일 이름 표시
    ):
        # ── 바이트 추출 ───────────────────────────────
        if isinstance(img_field, dict) and "bytes" in img_field:      # {'bytes': b'...', ...}
            byte_data = img_field["bytes"]
        elif isinstance(img_field, str):                              # base64 문자열
            byte_data = base64.b64decode(img_field)
        else:                                                         # 이미 bytes
            byte_data = img_field

        # ── PIL 이미지로 복원 ─────────────────────────
        img = Image.open(io.BytesIO(byte_data)).convert("RGB")

        # ── 저장 ─────────────────────────────────────
        label_dir = OUT_DIR / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)

        # 파일명: parquet 스템 + 행번호 (충돌 방지)
        fn = f"{pq_path.stem}_{row_idx:06d}.jpg"
        img.save(label_dir / fn, "JPEG", quality=95)