"""
Step 3: 꼬깔 검출 + 인덱싱 + 쌍 매칭 + Bearing Angle 계산
=============================================================
사용법:
  # 이미지 테스트
  python cone_detector.py --source test.jpg --calib calibration.yaml --model best.pt

  # 영상 테스트
  python cone_detector.py --source test.mp4 --calib calibration.yaml --model best.pt

  # 웹캠 실시간
  python cone_detector.py --source 0 --calib calibration.yaml --model best.pt

  # 캘리브레이션 없이 근사 모드 (테스트용)
  python cone_detector.py --source test.jpg --model best.pt --no-calib --hfov 195
"""

import cv2
import numpy as np
import yaml
import argparse
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO


# ── 데이터 구조 ─────────────────────────────────────────────────────────────

@dataclass
class Cone:
    """검출된 꼬깔 하나"""
    index: int              # 클래스 내 인덱스 (x좌표 기준 좌→우)
    color: str              # "red" or "blue"
    cx: float               # 이미지 중심점 x (픽셀)
    cy: float               # 이미지 중심점 y (픽셀)
    confidence: float       # 검출 신뢰도
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # 세그멘테이션 마스크

@dataclass
class ConePair:
    """매칭된 꼬깔 쌍"""
    pair_index: int         # 쌍 인덱스
    red: Cone
    blue: Cone
    midpoint_x: float       # 두 꼬깔 중간점 x (픽셀)
    midpoint_y: float       # 두 꼬깔 중간점 y (픽셀)
    bearing_deg: float      # Bearing angle (도, 양수=오른쪽, 음수=왼쪽)


# ── 카메라 모델 ──────────────────────────────────────────────────────────────

class FisheyeCamera:
    """어안렌즈 카메라 (캘리브레이션 기반)"""

    def __init__(self, calib_path: str):
        with open(calib_path, "r") as f:
            data = yaml.safe_load(f)

        self.W = data["image_width"]
        self.H = data["image_height"]

        K_data = data["camera_matrix"]["data"]
        self.K = np.array(K_data, dtype=np.float64).reshape(3, 3)

        D_data = data["distortion_coefficients"]["data"]
        self.D = np.array(D_data, dtype=np.float64).reshape(4, 1)

        # Undistortion 맵 미리 계산
        self.K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (self.W, self.H), np.eye(3), balance=1.0
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.K_new,
            (self.W, self.H), cv2.CV_16SC2
        )

        self.fx = self.K_new[0, 0]
        self.cx = self.K_new[0, 2]
        print(f"[Camera] Fisheye calibrated | fx={self.fx:.1f}, cx={self.cx:.1f}")

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def pixel_to_bearing(self, px: float) -> float:
        """보정된 이미지의 픽셀 x → bearing angle (도)"""
        return math.degrees(math.atan2(px - self.cx, self.fx))


class ApproxCamera:
    """근사 핀홀 카메라 (캘리브레이션 없이 HFoV만 사용)"""

    def __init__(self, W: int, H: int, hfov_deg: float):
        self.W = W
        self.H = H
        self.cx = W / 2.0
        # 어안렌즈이므로 등거리 투영(equidistant) 근사 적용
        # f_eq = W / hfov_rad  (등거리 모델)
        hfov_rad = math.radians(hfov_deg)
        self.f_eq = W / hfov_rad
        print(f"[Camera] Approx equidistant | hfov={hfov_deg}°, f_eq={self.f_eq:.1f}")
        print(f"  ※ 경고: 캘리브레이션 없는 근사 모드 — 중앙 영역에서만 비교적 정확")

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return img  # 근사 모드에서는 보정 없이 그대로 사용

    def pixel_to_bearing(self, px: float) -> float:
        """등거리 투영 모델 기반 bearing angle"""
        # theta = (px - cx) / f_eq  (등거리 모델)
        theta_rad = (px - self.cx) / self.f_eq
        return math.degrees(theta_rad)


# ── 클래스 설정 ──────────────────────────────────────────────────────────────
# YOLO 모델의 클래스 이름에 맞게 수정하세요
# 예: {0: "red_cone", 1: "blue_cone"}
CLASS_COLOR_MAP = {
    "red_cone":  "red",
    "blue_cone": "blue",
    # 다른 이름을 쓸 경우 추가:
    # "red":  "red",
    # "blue": "blue",
}

DRAW_COLORS = {
    "red":  (0, 0, 255),    # BGR
    "blue": (255, 100, 0),
}


# ── 메인 검출기 ──────────────────────────────────────────────────────────────

class ConeDetector:

    def __init__(self, model_path: str, camera, conf_thresh: float = 0.4):
        self.model = YOLO(model_path)
        self.camera = camera
        self.conf_thresh = conf_thresh
        print(f"[YOLO] 모델 로드: {model_path}")
        print(f"[YOLO] 클래스: {self.model.names}")

    def detect(self, frame: np.ndarray):
        """
        Returns:
            undist_frame: 보정된 이미지
            pairs:        List[ConePair]
            all_cones:    {"red": List[Cone], "blue": List[Cone]}
        """
        # 1. Undistortion
        undist = self.camera.undistort(frame)

        # 2. YOLO 추론
        results = self.model(undist, conf=self.conf_thresh, verbose=False)[0]

        # 3. 꼬깔 분리 및 중심점 추출
        red_cones, blue_cones = [], []

        if results.masks is not None:
            masks_data = results.masks.data.cpu().numpy()
            boxes = results.boxes

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                color = CLASS_COLOR_MAP.get(cls_name)

                if color is None:
                    continue

                conf = float(box.conf[0])
                mask = masks_data[i]  # H×W binary mask

                # 마스크 무게중심 (더 안정적)
                moments = cv2.moments(mask.astype(np.uint8))
                if moments["m00"] == 0:
                    continue
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]

                # 마스크를 원본 이미지 크기로 리사이즈
                h, w = undist.shape[:2]
                mask_full = cv2.resize(mask, (w, h))

                cone = Cone(index=-1, color=color, cx=cx, cy=cy,
                            confidence=conf, mask=mask_full)

                if color == "red":
                    red_cones.append(cone)
                else:
                    blue_cones.append(cone)

        # 4. x좌표 기준 정렬 및 인덱스 부여
        red_cones.sort(key=lambda c: c.cx)
        blue_cones.sort(key=lambda c: c.cx)
        for i, c in enumerate(red_cones):
            c.index = i
        for i, c in enumerate(blue_cones):
            c.index = i

        # 5. 쌍 매칭 (인덱스 순서 매칭)
        pairs = self._match_pairs(red_cones, blue_cones)

        all_cones = {"red": red_cones, "blue": blue_cones}
        return undist, pairs, all_cones

    def _match_pairs(self, red_cones, blue_cones):
        """인덱스 순서 매칭: red[i] ↔ blue[i]"""
        pairs = []
        n = min(len(red_cones), len(blue_cones))
        for i in range(n):
            r, b = red_cones[i], blue_cones[i]
            mid_x = (r.cx + b.cx) / 2
            mid_y = (r.cy + b.cy) / 2
            bearing = self.camera.pixel_to_bearing(mid_x)
            pairs.append(ConePair(
                pair_index=i,
                red=r, blue=b,
                midpoint_x=mid_x,
                midpoint_y=mid_y,
                bearing_deg=bearing
            ))
        return pairs


# ── 시각화 ───────────────────────────────────────────────────────────────────

def draw_results(frame: np.ndarray, pairs, all_cones, img_w: int) -> np.ndarray:
    vis = frame.copy()
    cx_img = img_w / 2

    # 마스크 오버레이
    for color, cones in all_cones.items():
        bgr = DRAW_COLORS[color]
        for cone in cones:
            if cone.mask is not None:
                colored = np.zeros_like(vis)
                colored[cone.mask > 0.5] = bgr
                vis = cv2.addWeighted(vis, 1.0, colored, 0.4, 0)

    # 카메라 중심선
    cv2.line(vis, (int(cx_img), 0), (int(cx_img), vis.shape[0]),
             (200, 200, 200), 1, cv2.LINE_AA)

    # 꼬깔 중심점 및 인덱스 레이블
    for color, cones in all_cones.items():
        bgr = DRAW_COLORS[color]
        for cone in cones:
            cx, cy = int(cone.cx), int(cone.cy)
            cv2.circle(vis, (cx, cy), 6, bgr, -1)
            cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 2)
            label = f"{color[0].upper()}{cone.index}"  # e.g., R0, B1
            cv2.putText(vis, label, (cx + 10, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

    # 쌍 연결선 및 베어링 표시
    for pair in pairs:
        rx, ry = int(pair.red.cx), int(pair.red.cy)
        bx, by = int(pair.blue.cx), int(pair.blue.cy)
        mx, my = int(pair.midpoint_x), int(pair.midpoint_y)

        # 두 꼬깔 연결선
        cv2.line(vis, (rx, ry), (bx, by), (0, 255, 255), 2, cv2.LINE_AA)

        # 중심점
        cv2.circle(vis, (mx, my), 8, (0, 255, 255), -1)

        # 중심점 → 이미지 중심 오프셋 선
        cv2.line(vis, (int(cx_img), my), (mx, my),
                 (0, 255, 255), 1, cv2.LINE_AA)

        # 베어링 텍스트
        arrow = "R" if pair.bearing_deg > 0 else "L" if pair.bearing_deg < 0 else "C"
        text = f"Pair{pair.pair_index}: {pair.bearing_deg:+.1f}° {arrow}"
        cv2.putText(vis, text, (mx + 10, my + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # 상단 요약 패널
    y_off = 28
    for pair in pairs:
        summary = (f"[Pair {pair.pair_index}] "
                   f"R{pair.red.index}({pair.red.cx:.0f}px) <-> "
                   f"B{pair.blue.index}({pair.blue.cx:.0f}px) | "
                   f"mid=({pair.midpoint_x:.0f},{pair.midpoint_y:.0f}) | "
                   f"bearing={pair.bearing_deg:+.2f}°")
        cv2.putText(vis, summary, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_off += 22

    return vis


# ── 결과 출력 ────────────────────────────────────────────────────────────────

def print_results(pairs, all_cones):
    print("\n" + "="*60)
    print(f"  Red:  {len(all_cones['red'])}개  |  Blue: {len(all_cones['blue'])}개")
    print(f"  매칭 쌍: {len(pairs)}개")
    print("-"*60)
    for pair in pairs:
        direction = "RIGHT" if pair.bearing_deg > 1 else ("LEFT" if pair.bearing_deg < -1 else "CENTER")
        print(f"  Pair {pair.pair_index}: "
              f"red[{pair.red.index}]({pair.red.cx:.0f}px) ↔ "
              f"blue[{pair.blue.index}]({pair.blue.cx:.0f}px)")
        print(f"           중간점=({pair.midpoint_x:.1f}, {pair.midpoint_y:.1f}px) "
              f"| Bearing={pair.bearing_deg:+.2f}° [{direction}]")
    print("="*60)


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="꼬깔 검출 + Bearing Angle")
    parser.add_argument("--source",  required=True,
                        help="입력 소스: 이미지/영상 경로 또는 카메라 인덱스(0)")
    parser.add_argument("--model",   required=True, help="YOLO .pt 파일 경로")
    parser.add_argument("--calib",   default=None,  help="calibration.yaml 경로")
    parser.add_argument("--no-calib", action="store_true",
                        help="캘리브레이션 없이 근사 모드로 실행")
    parser.add_argument("--hfov",    type=float, default=195.0,
                        help="수평 화각 (근사 모드 전용, default=195)")
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="YOLO 신뢰도 임계값 (default=0.4)")
    parser.add_argument("--save",    action="store_true",
                        help="결과 영상/이미지 저장")
    args = parser.parse_args()

    # ── 카메라 모델 초기화
    if args.no_calib or args.calib is None:
        camera = ApproxCamera(W=1280, H=720, hfov_deg=args.hfov)
    else:
        camera = FisheyeCamera(args.calib)

    detector = ConeDetector(args.model, camera, conf_thresh=args.conf)

    # ── 소스 판별
    try:
        source_int = int(args.source)
        cap = cv2.VideoCapture(source_int)
        is_image = False
    except ValueError:
        path = Path(args.source)
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            is_image = True
        else:
            is_image = False
            cap = cv2.VideoCapture(str(path))

    # ── 저장 설정
    writer = None
    if args.save and not is_image:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, 30, (1280, 720))

    # 창 설정 (크기 조정 가능)
    cv2.namedWindow("Cone Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cone Detector", 1280, 720)

    # ── 실행
    if is_image:
        frame = cv2.imread(args.source)
        undist, pairs, all_cones = detector.detect(frame)
        print_results(pairs, all_cones)
        vis = draw_results(undist, pairs, all_cones, undist.shape[1])
        cv2.imshow("Cone Detector", vis)
        if args.save:
            cv2.imwrite("output.jpg", vis)
            print("[SAVED] output.jpg")
        cv2.waitKey(0)

    else:
        print("[INFO] 실행 중... 종료: q")
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            undist, pairs, all_cones = detector.detect(frame)
            vis = draw_results(undist, pairs, all_cones, undist.shape[1])

            # 터미널 출력 (매 30프레임)
            if frame_id % 30 == 0:
                print_results(pairs, all_cones)

            cv2.imshow("Cone Detector", vis)
            if writer:
                writer.write(vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_id += 1

        cap.release()
        if writer:
            writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()cd 