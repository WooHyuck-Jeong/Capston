"""
realtime_seg.py
학습된 YOLO Segmentation 모델로 USB 카메라 실시간 추론

사용법:
    python realtime_seg.py --weights best.pt
    python realtime_seg.py --weights best.pt --camera 0 --conf 0.5
    
    # 기본 실행
    python realtime_seg.py --weights runs/segment/runs/segment/cone_seg/weights/best.pt --conf 0.3

    # conf 낮추면 더 많이 검출 (기본 0.5)
    python realtime_seg.py --weights best.pt --conf 0.3

    # 카메라 인덱스 지정
    python realtime_seg.py --weights best.pt --camera 1

    # 마스크 투명도 조절 (0.0 투명 ~ 1.0 불투명)
    python realtime_seg.py --weights best.pt --alpha 0.5
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics 패키지가 설치되지 않았습니다.")
    print("        pip install ultralytics 를 실행하세요.")
    sys.exit(1)


# ── 클래스별 색상 (BGR) ───────────────────────────────────────────────────────
CLASS_COLORS = {
    "blue_cone": (255, 100,   0),   # 파란색
    "red_cone" : (  0,   0, 255),   # 빨간색
}
DEFAULT_COLOR = (0, 255, 0)         # 미등록 클래스 기본색 (초록)


def get_color(class_name: str) -> tuple:
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


# ══════════════════════════════════════════════════════════════════════════════
# 실시간 추론
# ══════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] 모델 파일 없음: {weights}")
        sys.exit(1)

    print(f"[INFO] 모델 로드: {weights}")
    model      = YOLO(str(weights))
    class_names = model.names   # {0: 'blue_cone', 1: 'red_cone'}
    print(f"[INFO] 클래스: {class_names}")

    # ── 카메라 오픈 ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 카메라 인덱스 {args.camera} 를 열 수 없습니다.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 카메라 해상도: {actual_w} x {actual_h}")
    print(f"[INFO] conf 임계값 : {args.conf}")
    print()
    print("  [Q / ESC] 종료")
    print()

    # FPS 계산용
    fps_list  = []
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패")
            break

        # ── YOLO 추론 ─────────────────────────────────────────────────────────
        results = model(
            frame,
            conf    = args.conf,
            iou     = args.iou,
            imgsz   = args.imgsz,
            verbose = False,
        )

        # ── 결과 시각화 ───────────────────────────────────────────────────────
        display = frame.copy()
        result  = results[0]

        det_count = {"blue_cone": 0, "red_cone": 0}

        if result.masks is not None:
            masks      = result.masks.data.cpu().numpy()   # (N, H, W)
            boxes      = result.boxes
            class_ids  = boxes.cls.cpu().numpy().astype(int)
            confidences= boxes.conf.cpu().numpy()
            xyxy       = boxes.xyxy.cpu().numpy().astype(int)

            for i, (mask, cls_id, conf, box) in enumerate(
                zip(masks, class_ids, confidences, xyxy)
            ):
                cls_name = class_names[cls_id]
                color    = get_color(cls_name)

                # ── 마스크 오버레이 ───────────────────────────────────────────
                mask_resized = cv2.resize(
                    mask, (actual_w, actual_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                overlay        = display.copy()
                overlay[mask_resized] = color
                display = cv2.addWeighted(display, 1 - args.alpha, overlay, args.alpha, 0)

                # ── 마스크 외곽선 ─────────────────────────────────────────────
                mask_uint8  = (mask_resized.astype(np.uint8)) * 255
                mask_uint8  = cv2.resize(mask_uint8, (actual_w, actual_h))
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(display, contours, -1, color, 2)

                # ── 바운딩 박스 + 라벨 ────────────────────────────────────────
                x1, y1, x2, y2 = box
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name} {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(display,
                              (x1, y1 - lh - 8), (x1 + lw + 4, y1),
                              color, -1)
                cv2.putText(display, label,
                            (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                if cls_name in det_count:
                    det_count[cls_name] += 1

        # ── FPS 계산 ──────────────────────────────────────────────────────────
        now      = time.time()
        fps_list.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        if len(fps_list) > 30:
            fps_list.pop(0)
        fps = sum(fps_list) / len(fps_list)

        # ── HUD 오버레이 ──────────────────────────────────────────────────────
        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Blue cone: {det_count['blue_cone']}",
            f"Red  cone: {det_count['red_cone']}",
            f"conf: {args.conf}  iou: {args.iou}",
        ]
        for j, line in enumerate(hud_lines):
            cv2.putText(display, line,
                        (10, 30 + j * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(display, line,
                        (10, 30 + j * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 1)

        cv2.putText(display, "Q/ESC: Quit",
                    (10, actual_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        cv2.imshow("Cone Segmentation", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 종료")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="실시간 꼬깔 Segmentation 추론",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", "-w", required=True,
                        help="학습된 모델 경로 (best.pt)")
    parser.add_argument("--camera",  "-c", type=int, default=0,
                        help="카메라 인덱스")
    parser.add_argument("--conf",    "-t", type=float, default=0.5,
                        help="confidence 임계값 (낮을수록 더 많이 검출)")
    parser.add_argument("--iou",     "-u", type=float, default=0.45,
                        help="NMS IoU 임계값")
    parser.add_argument("--imgsz",   "-i", type=int, default=640,
                        help="추론 이미지 크기")
    parser.add_argument("--alpha",   "-a", type=float, default=0.4,
                        help="마스크 투명도 (0.0~1.0)")
    parser.add_argument("--width",   "-W", type=int, default=1280,
                        help="카메라 해상도 가로")
    parser.add_argument("--height",  "-H", type=int, default=720,
                        help="카메라 해상도 세로")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())