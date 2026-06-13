"""
핀홀 카메라 캘리브레이션 스크립트 (ELP-USB500W02M-BL36)
=========================================================
사용법:
  python calibration_cam.py --images calib_img
  python calibration_cam.py --images calib_img --cols 8 --rows 5 --size 30

결과:
  calibration.yaml : K (카메라 행렬), D (왜곡계수 5개) 저장
  undistortion_preview.jpg : 보정 전/후 비교 이미지
"""

import cv2
import numpy as np
import yaml
import glob
import argparse
import os


def calibrate_pinhole(image_dir: str, cols: int, rows: int,
                      square_size_mm: float, output_path: str):

    CHECKERBOARD = (cols, rows)

    # 3D 코너 좌표 (Z=0 평면)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                    glob.glob(os.path.join(image_dir, "*.png")))

    if not images:
        print(f"[ERROR] 이미지 없음: {image_dir}")
        return

    print(f"[INFO] {len(images)}장 로드")
    img_size     = None
    success      = 0
    fail_list    = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (W, H)

        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            corners_ref = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_ref)
            success += 1
            print(f"  [OK]   {os.path.basename(fname)}")
        else:
            fail_list.append(os.path.basename(fname))
            print(f"  [FAIL] {os.path.basename(fname)}")

    print(f"\n[INFO] 성공: {success}/{len(images)}장")
    if fail_list:
        print(f"[INFO] 실패 목록: {fail_list}")

    if success < 5:
        print("[ERROR] 최소 5장 이상 필요합니다.")
        return

    # ── 핀홀 캘리브레이션 ─────────────────────────────────
    print("\n캘리브레이션 계산 중...")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    print(f"\n{'='*50}")
    print(f"캘리브레이션 완료! RMS 오차: {rms:.4f} px")
    print(f"  → RMS < 1.0이면 양호, < 0.5이면 우수")
    print(f"\n카메라 내부행렬 K:\n{K}")
    print(f"\n왜곡 계수 D (k1, k2, p1, p2, k3):\n{D.T}")

    # fx, fy로 HFOV 역산 (확인용)
    fx = K[0, 0]
    W  = img_size[0]
    hfov = np.degrees(2 * np.arctan(W / (2 * fx)))
    print(f"\n추정 HFOV: {hfov:.1f}도  (렌즈 3.6mm 기준 약 53도 예상)")

    # ── YAML 저장 ─────────────────────────────────────────
    # 폴더만 입력했을 경우 파일명 자동 추가
    if os.path.isdir(output_path) or not output_path.endswith(".yaml"):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "calibration.yaml")

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    calib_data = {
        "camera_model":  "pinhole",
        "image_width":   img_size[0],
        "image_height":  img_size[1],
        "rms_error":     float(rms),
        "camera_matrix": {
            "rows": 3, "cols": 3,
            "data": K.flatten().tolist()
        },
        "distortion_coefficients": {
            "rows": 1, "cols": 5,
            "data": D.flatten().tolist()
        },
        "note": f"ELP-USB500W02M-BL36, 3.6mm lens, HFOV~{hfov:.1f}deg"
    }

    with open(output_path, "w") as f:
        yaml.dump(calib_data, f, default_flow_style=False)

    print(f"\n[SAVED] {output_path}")
    print(f"{'='*50}")

    # ── Undistortion 미리보기 ──────────────────────────────
    _preview(images[0], K, D, img_size)


def _preview(img_path: str, K, D, img_size):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 최적 카메라 행렬 계산 (alpha=1: 검은 영역 없음)
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)
    undist = cv2.undistort(img, K, D, None, K_new)

    # ROI 크롭 (유효 영역만)
    x, y, rw, rh = roi
    undist_crop = undist[y:y+rh, x:x+rw]
    undist_resized = cv2.resize(undist_crop, (w, h))

    comparison = np.hstack([img, undist_resized])
    cv2.putText(comparison, "Original",    (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(comparison, "Undistorted", (w + 10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    out = "undistortion_preview.jpg"
    cv2.imwrite(out, comparison)
    print(f"[PREVIEW] {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="핀홀 카메라 캘리브레이션")
    parser.add_argument("--images", default="calib_img",
                        help="이미지 폴더 (default: calib_img)")
    parser.add_argument("--cols",   type=int, default=8,
                        help="체커보드 내부 코너 열 수 (default: 8)")
    parser.add_argument("--rows",   type=int, default=5,
                        help="체커보드 내부 코너 행 수 (default: 5)")
    parser.add_argument("--size",   type=float, default=30.0,
                        help="체커보드 한 칸 크기 mm (default: 30)")
    parser.add_argument("--output", default="calibration.yaml",
                        help="결과 저장 경로 (default: calibration.yaml)")
    args = parser.parse_args()

    calibrate_pinhole(args.images, args.cols, args.rows,
                      args.size, args.output)