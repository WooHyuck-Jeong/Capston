"""
train_cone_seg.py
꼬깔 Semantic Segmentation - YOLOv11 학습 스크립트

사용법:
    python train_cone_seg.py --data dataset/
    python train_cone_seg.py --data dataset/ --model yolo11m-seg.pt --device cpu --batch 8 --workers 0
"""

import argparse
import json
import sys
import shutil
from pathlib import Path
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics 패키지가 설치되지 않았습니다.")
    print("        pip install ultralytics 를 실행하세요.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. RLE 디코딩 (pycocotools 없이 순수 Python)
# ══════════════════════════════════════════════════════════════════════════════

def rle_decode(rle: dict) -> list:
    """
    COCO RLE(counts 문자열) → 이진 마스크 → 폴리곤 좌표 리스트 반환
    반환: [x1,y1, x2,y2, ...] 픽셀 좌표 (정규화 전)
    """
    import re

    counts_raw = rle["counts"]
    h, w = rle["size"]  # [height, width]

    # ── RLE 문자열 디코딩 ─────────────────────────────────────────────────────
    # COCO uncompressed RLE: 숫자 배열
    # COCO compressed RLE  : LEB128 유사 문자열
    if isinstance(counts_raw, list):
        counts = counts_raw
    else:
        # compressed RLE 디코딩
        counts = []
        m = 0
        p = 0
        s = counts_raw
        while p < len(s):
            x = 0
            k = 0
            more = True
            while more:
                c = ord(s[p]) - 48
                p += 1
                more = bool(c & 32)
                x |= (c & 31) << (5 * k)
                k += 1
            if m > 2 and x <= 10:
                x += counts[-2]
            counts.append(x)
            m += 1

    # ── 마스크 생성 ───────────────────────────────────────────────────────────
    mask = []
    val = 0
    for c in counts:
        mask.extend([val] * c)
        val = 1 - val
    mask = mask[:h * w]
    if len(mask) < h * w:
        mask.extend([0] * (h * w - len(mask)))

    # ── 마스크 → 바운딩 폴리곤 (외곽 픽셀 추출) ───────────────────────────────
    # 마스크가 있는 픽셀의 x,y 좌표 수집 (column-major 순서)
    xs, ys = [], []
    for idx, v in enumerate(mask):
        if v == 1:
            col = idx // h   # x
            row = idx % h    # y
            xs.append(col)
            ys.append(row)

    if not xs:
        return []

    # 바운딩 박스 기반 간단 폴리곤 (4점)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 더 정밀한 폴리곤: 행별 좌우 끝점 수집
    row_range = defaultdict(lambda: [w, 0])
    for x, y in zip(xs, ys):
        if x < row_range[y][0]:
            row_range[y][0] = x
        if x > row_range[y][1]:
            row_range[y][1] = x

    rows = sorted(row_range.keys())
    left_pts  = [(row_range[r][0], r) for r in rows]
    right_pts = [(row_range[r][1], r) for r in reversed(rows)]
    polygon   = left_pts + right_pts

    coords = []
    for px, py in polygon:
        coords.extend([px, py])

    return coords


# ══════════════════════════════════════════════════════════════════════════════
# 2. COCO JSON → YOLO Segmentation txt 변환
# ══════════════════════════════════════════════════════════════════════════════

def coco_to_yolo_seg(dataset_root: Path, class_id_map: dict) -> None:
    for split in ("train", "valid", "test"):
        ann_path = dataset_root / split / "_annotations.coco.json"
        if not ann_path.exists():
            continue

        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        img_info = {img["id"]: img for img in coco["images"]}

        ann_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_img[ann["image_id"]].append(ann)

        # 이미지를 split/images/ 로 이동
        images_dir = dataset_root / split / "images"
        images_dir.mkdir(exist_ok=True)
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        moved = 0
        for f in list((dataset_root / split).iterdir()):
            if f.is_file() and f.suffix.lower() in img_exts:
                dst = images_dir / f.name
                if not dst.exists():
                    shutil.move(str(f), str(dst))
                    moved += 1
        if moved:
            print(f"  [{split}] 이미지 {moved}개 → {images_dir}")

        labels_dir = dataset_root / split / "labels"
        labels_dir.mkdir(exist_ok=True)

        converted = 0
        skipped   = 0

        for img_id, img in img_info.items():
            w    = img["width"]
            h    = img["height"]
            stem = Path(img["file_name"]).stem
            txt_path = labels_dir / f"{stem}.txt"

            lines = []
            for ann in ann_by_img[img_id]:
                cat_id = ann["category_id"]
                if cat_id not in class_id_map:
                    skipped += 1
                    continue

                yolo_cls = class_id_map[cat_id]
                seg      = ann.get("segmentation", [])

                # ── 폴리곤 추출 ──────────────────────────────────────────────
                if isinstance(seg, dict):
                    # RLE 형식 → 폴리곤 변환
                    coords_px = rle_decode(seg)
                elif isinstance(seg, list) and seg:
                    # 폴리곤 형식: 숫자만 추출
                    raw = seg if isinstance(seg[0], (int, float)) else seg[0]
                    coords_px = []
                    for v in raw:
                        try:
                            coords_px.append(float(v))
                        except (ValueError, TypeError):
                            continue
                    if len(coords_px) % 2 != 0:
                        coords_px = coords_px[:-1]
                else:
                    skipped += 1
                    continue

                if len(coords_px) < 6:
                    skipped += 1
                    continue

                # ── 정규화 ───────────────────────────────────────────────────
                coords_norm = []
                for i in range(0, len(coords_px), 2):
                    nx = round(float(coords_px[i])     / w, 6)
                    ny = round(float(coords_px[i + 1]) / h, 6)
                    nx = max(0.0, min(1.0, nx))
                    ny = max(0.0, min(1.0, ny))
                    coords_norm.extend([nx, ny])

                lines.append(f"{yolo_cls} " + " ".join(map(str, coords_norm)))
                converted += 1

            txt_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"  [{split}] 라벨 변환 완료: {converted}개  건너뜀: {skipped}개 → {labels_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 클래스 정보 추출
# ══════════════════════════════════════════════════════════════════════════════

def get_classes(dataset_root: Path) -> tuple:
    train_ann = dataset_root / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        print(f"[ERROR] 어노테이션 파일 없음: {train_ann}")
        sys.exit(1)

    with open(train_ann, encoding="utf-8") as f:
        coco = json.load(f)

    used_ids = {ann["category_id"] for ann in coco.get("annotations", [])}
    categories = sorted(
        [c for c in coco["categories"] if c["id"] in used_ids],
        key=lambda c: c["id"],
    )
    if not categories:
        categories = sorted(coco["categories"], key=lambda c: c["id"])

    class_names  = [c["name"] for c in categories]
    class_id_map = {c["id"]: i for i, c in enumerate(categories)}

    print(f"  클래스 수   : {len(class_names)}")
    print(f"  클래스명    : {class_names}")
    print(f"  category_id 매핑: {class_id_map}")
    return class_names, class_id_map


# ══════════════════════════════════════════════════════════════════════════════
# 4. data.yaml 생성
# ══════════════════════════════════════════════════════════════════════════════

def build_yaml(dataset_root: Path, yaml_path: Path, class_names: list) -> None:
    nc = len(class_names)
    has_test = (dataset_root / "test" / "images").exists()

    lines = [
        "# Auto-generated by train_cone_seg.py",
        f"path: {dataset_root.resolve()}",
        "train: train/images",
        "val:   valid/images",
    ]
    if has_test:
        lines.append("test:  test/images")
    lines += ["", f"nc: {nc}", f"names: {class_names}", ""]

    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  저장: {yaml_path}")
    print(f"  nc={nc}  names={class_names}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 학습
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    dataset_root = Path(args.data).resolve()

    print("\n[STEP 1] 클래스 정보 추출...")
    class_names, class_id_map = get_classes(dataset_root)

    print("\n[STEP 2] COCO JSON → YOLO 라벨 변환 (RLE 지원)...")
    labels_dir = dataset_root / "train" / "labels"
    if labels_dir.exists() and any(labels_dir.glob("*.txt")):
        # 내용이 비어있는 파일이 있으면 재변환
        empty = [f for f in labels_dir.glob("*.txt") if f.stat().st_size == 0]
        if empty:
            print(f"  빈 라벨 파일 {len(empty)}개 발견 → 전체 재변환...")
            for split in ("train", "valid", "test"):
                ld = dataset_root / split / "labels"
                if ld.exists():
                    shutil.rmtree(ld)
            coco_to_yolo_seg(dataset_root, class_id_map)
        else:
            print(f"  이미 변환된 라벨 존재 → 건너뜀")
    else:
        coco_to_yolo_seg(dataset_root, class_id_map)

    print("\n[STEP 3] data.yaml 생성...")
    yaml_path = dataset_root / "data.yaml"
    build_yaml(dataset_root, yaml_path, class_names)

    print(f"\n{'='*60}")
    print(f"  [STEP 4] 학습 시작")
    print(f"  data   : {yaml_path}")
    print(f"  model  : {args.model}")
    print(f"  epochs : {args.epochs}  |  batch: {args.batch}  |  device: {args.device}")
    print(f"{'='*60}\n")

    model   = YOLO(args.model)
    results = model.train(
        data     = str(yaml_path),
        task     = "segment",
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device,
        project  = args.project,
        name     = args.name,
        patience = args.patience,
        save     = True,
        plots    = True,
        workers  = args.workers,
        exist_ok = True,
    )

    save_dir = Path(results.save_dir)
    print(f"\n{'='*60}")
    print(f"  학습 완료!")
    print(f"  결과 폴더 : {save_dir}")
    print(f"  best.pt   : {save_dir / 'weights' / 'best.pt'}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 검증
# ══════════════════════════════════════════════════════════════════════════════

def validate(args: argparse.Namespace) -> None:
    weights   = Path(args.project) / args.name / "weights" / "best.pt"
    yaml_path = Path(args.data).resolve() / "data.yaml"
    if not weights.exists():
        print(f"[WARN] best.pt 없음: {weights}")
        return
    print(f"\n[INFO] 검증: {weights}")
    model   = YOLO(str(weights))
    metrics = model.val(data=str(yaml_path), imgsz=args.imgsz, device=args.device)
    print(f"  mAP50    : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95 : {metrics.seg.map:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="꼬깔 YOLO Segmentation 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",     "-d", default="dataset")
    parser.add_argument("--model",    "-m", default="yolo11n-seg.pt",
                        choices=["yolo11n-seg.pt","yolo11s-seg.pt",
                                 "yolo11m-seg.pt","yolo11l-seg.pt","yolo11x-seg.pt"])
    parser.add_argument("--epochs",   "-e", type=int, default=100)
    parser.add_argument("--imgsz",    "-i", type=int, default=640)
    parser.add_argument("--batch",    "-b", type=int, default=16)
    parser.add_argument("--patience", "-p", type=int, default=20)
    parser.add_argument("--workers",  "-w", type=int, default=4)
    parser.add_argument("--device",         default="0")
    parser.add_argument("--project",        default="runs/segment")
    parser.add_argument("--name",           default="cone_seg")
    parser.add_argument("--convert-only",   action="store_true")
    parser.add_argument("--val-only",       action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.convert_only:
        dataset_root = Path(args.data).resolve()
        _, class_id_map = get_classes(dataset_root)
        coco_to_yolo_seg(dataset_root, class_id_map)
        sys.exit(0)

    if args.val_only:
        validate(args)
    else:
        train(args)
        validate(args)