"""
캘리브레이션 이미지 촬영 스크립트 (ELP-USB500W02M-BL36)
=========================================================
카메라 스펙:
  - 센서  : OV5640 5MP
  - 렌즈  : 3.6mm 고정 (HFOV 약 53도)
  - 해상도: 2592x1944 (최대), 1280x720 권장

사용법:
  python get_calib_img.py
  python get_calib_img.py --camera 1 --width 1280 --height 720 --out calib_img

조작키:
  s : 스냅샷 저장 (체커보드 코너 감지 시 초록 테두리 표시)
  q : 종료
"""

import cv2
import numpy as np
import argparse
import os


def capture(camera_index: int, width: int, height: int,
            output_dir: str, cols: int, rows: int):

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] 카메라({camera_index}) 열기 실패")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 해상도: {actual_w}x{actual_h}")
    print(f"[INFO] 체커보드: {cols}x{rows} 내부 코너")

    os.makedirs(output_dir, exist_ok=True)
    snap_count = len([f for f in os.listdir(output_dir)
                      if f.endswith(".jpg")])  # 기존 파일 이어서 저장

    cv2.namedWindow("Calibration Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Capture", 960, 540)

    print("\n" + "="*45)
    print("  s : 스냅샷 저장")
    print("  q : 종료")
    print("="*45 + "\n")

    CHECKERBOARD = (cols, rows)
    corner_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임 읽기 실패")
            break

        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 체커보드 코너 실시간 감지
        ret_cb, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        corner_found = ret_cb

        if ret_cb:
            # 코너 정밀화
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_ref = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_ref, ret_cb)

        # ── 오버레이 ──────────────────────────────────────
        # 상태 바
        status_color = (0, 200, 0) if corner_found else (0, 0, 200)
        status_text  = "READY  (s: save)" if corner_found else "SEARCHING..."
        cv2.rectangle(vis, (0, 0), (actual_w, 40), (0, 0, 0), -1)
        cv2.putText(vis, status_text, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # 저장 장 수
        count_text = f"Saved: {snap_count}  |  q: quit"
        cv2.putText(vis, count_text, (actual_w - 280, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # 가이드 (저장 수가 적을 때)
        if snap_count < 5:
            guides = [
                "Tip: 다양한 각도/위치/거리로 촬영하세요",
                "Tip: 체커보드가 초록으로 바뀔 때 저장하세요",
            ]
            for gi, g in enumerate(guides):
                cv2.putText(vis, g, (12, actual_h - 20 - gi * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 1)

        cv2.imshow("Calibration Capture", vis)
        key = cv2.waitKey(1) & 0xFF

        # s: 스냅샷 저장
        if key == ord("s"):
            path = os.path.join(output_dir, f"calib_{snap_count:03d}.jpg")
            cv2.imwrite(path, frame)
            snap_count += 1
            status = "OK (corner detected)" if corner_found else "SAVED (no corner)"
            print(f"[SNAP {snap_count:03d}] {path}  [{status}]")

        # q: 종료
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[완료] 총 {snap_count}장 저장 → {output_dir}/")
    if snap_count < 15:
        print(f"[WARNING] 캘리브레이션 권장: 15장 이상 (현재 {snap_count}장)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="캘리브레이션 이미지 촬영")
    parser.add_argument("--camera", type=int, default=0,
                        help="카메라 인덱스 (default: 0)")
    parser.add_argument("--width",  type=int, default=1280,
                        help="해상도 가로 (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="해상도 세로 (default: 720)")
    parser.add_argument("--out",    default="calib_img",
                        help="저장 폴더 (default: calib_img)")
    parser.add_argument("--cols",   type=int, default=8,
                        help="체커보드 내부 코너 열 수 (default: 8)")
    parser.add_argument("--rows",   type=int, default=5,
                        help="체커보드 내부 코너 행 수 (default: 5)")
    args = parser.parse_args()

    capture(args.camera, args.width, args.height,
            args.out, args.cols, args.rows)