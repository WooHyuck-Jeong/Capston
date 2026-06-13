# """
# Step 3: 꼬깔 검출 + 인덱싱 + 쌍 매칭 + Bearing Angle 계산
# =============================================================
# 사용법:
#   python cone_detector.py --source 0 --calib calibration.yaml --model best.pt
#   python cone_detector.py --source test.jpg --calib calibration.yaml --model best.pt
#   python cone_detector.py --source test.mp4 --calib calibration.yaml --model best.pt --save
# """

# import cv2
# import numpy as np
# import yaml
# import argparse
# import math
# from pathlib import Path
# from dataclasses import dataclass, field
# from typing import Optional, Tuple
# from ultralytics import YOLO


# # ── 데이터 구조 ─────────────────────────────────────────────────────────────

# @dataclass
# class Cone:
#     index: int              # 클래스 내 인덱스 (x좌표 기준 좌→우)
#     color: str              # "red" or "blue"
#     cx: float               # BBOX 중앙점 x (픽셀)
#     cy: float               # BBOX 중앙점 y (픽셀)
#     bbox: Tuple             # (x1, y1, x2, y2)
#     confidence: float
#     mask: Optional[np.ndarray] = field(default=None, repr=False)

# @dataclass
# class ConePair:
#     pair_index: int
#     red: Cone
#     blue: Cone
#     midpoint_x: float       # BBOX 중앙점 연결선의 중점 x
#     midpoint_y: float
#     bearing_deg: float      # Bearing angle (양수=오른쪽, 음수=왼쪽)


# # ── 클래스 설정 ──────────────────────────────────────────────────────────────
# CLASS_COLOR_MAP = {
#     "red_cone":  "red",
#     "blue_cone": "blue",
# }

# # BGR
# MASK_COLORS = {
#     "red":  (0,   50, 255),
#     "blue": (255, 80,   0),
# }
# BBOX_COLORS = {
#     "red":  (0,   0,  220),
#     "blue": (220, 60,   0),
# }
# MID_COLOR   = (0, 255, 0)   # 중간점 / 연결선
# AXIS_COLOR  = (180, 180, 180) # 카메라 중심선


# # ── 카메라 모델 ──────────────────────────────────────────────────────────────

# class PinholeCamera:
#     """일반 핀홀 카메라 (ELP-USB500W02M-BL36, 3.6mm 렌즈)"""

#     def __init__(self, calib_path: str):
#         with open(calib_path, "r") as f:
#             data = yaml.safe_load(f)

#         self.W = data["image_width"]
#         self.H = data["image_height"]
#         self.K = np.array(data["camera_matrix"]["data"],
#                           dtype=np.float64).reshape(3, 3)
#         self.D = np.array(data["distortion_coefficients"]["data"],
#                           dtype=np.float64).reshape(1, 5)

#         # alpha=0: 유효 픽셀만 포함, 검은 영역 완전 제거
#         self.K_new, self.roi = cv2.getOptimalNewCameraMatrix(
#             self.K, self.D, (self.W, self.H), alpha=0)

#         # Undistortion 맵 미리 계산
#         self.map1, self.map2 = cv2.initUndistortRectifyMap(
#             self.K, self.D, None, self.K_new,
#             (self.W, self.H), cv2.CV_16SC2)

#         self.fx   = self.K_new[0, 0]
#         self.cx_p = self.K_new[0, 2]
#         x, y, rw, rh = self.roi
#         print(f"[Camera] Pinhole | fx={self.fx:.1f}  cx={self.cx_p:.1f}")
#         print(f"[Camera] ROI=({x},{y},{x+rw},{y+rh})")

#     def undistort(self, img: np.ndarray) -> np.ndarray:
#         # crop 없이 undistort만 적용 → 검은 화면 없음
#         return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

#     def pixel_to_bearing(self, px: float) -> float:
#         """핀홀 모델: bearing = arctan((px - cx) / fx)"""
#         return math.degrees(math.atan2(px - self.cx_p, self.fx))


# class ApproxCamera:
#     def __init__(self, W: int, H: int, hfov_deg: float):
#         self.W  = W
#         self.H  = H
#         self.cx_p = W / 2.0
#         self.roi  = (0, 0, W, H)
#         hfov_rad  = math.radians(hfov_deg)
#         self.f_eq = W / hfov_rad
#         print(f"[Camera] Approx | hfov={hfov_deg}deg  f_eq={self.f_eq:.1f}")

#     def undistort(self, img: np.ndarray) -> np.ndarray:
#         return img

#     def pixel_to_bearing(self, px: float) -> float:
#         return math.degrees((px - self.cx_p) / self.f_eq)


# # ── 검출기 ───────────────────────────────────────────────────────────────────

# class ConeDetector:
#     def __init__(self, model_path: str, camera, conf_thresh: float = 0.4,
#                  min_area_ratio: float = 0.003):
#         self.model         = YOLO(model_path)
#         self.camera        = camera
#         self.conf_thresh   = conf_thresh
#         self.min_area_ratio = min_area_ratio
#         print(f"[YOLO] {model_path}  classes={self.model.names}")

#     def detect(self, frame: np.ndarray):
#         undist  = self.camera.undistort(frame)
#         results = self.model(undist, conf=self.conf_thresh, verbose=False)[0]

#         red_cones, blue_cones = [], []

#         if results.masks is not None:
#             masks_data = results.masks.data.cpu().numpy()
#             img_h, img_w = undist.shape[:2]
#             img_area = img_h * img_w

#             for i, box in enumerate(results.boxes):
#                 cls_name = self.model.names[int(box.cls[0])]
#                 color    = CLASS_COLOR_MAP.get(cls_name)
#                 if color is None:
#                     continue

#                 # ── BBOX (xyxy, 원본 이미지 좌표계)
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                 x1 = max(0, x1); y1 = max(0, y1)
#                 x2 = min(img_w, x2); y2 = min(img_h, y2)
#                 bbox_cx = (x1 + x2) / 2.0
#                 bbox_cy = (y1 + y2) / 2.0
#                 bbox_area = (x2 - x1) * (y2 - y1)

#                 # 면적 필터 (오탐 제거)
#                 if bbox_area / img_area < self.min_area_ratio:
#                     continue

#                 # 마스크 → 원본 해상도로 리사이즈
#                 mask = masks_data[i]
#                 mask_full = cv2.resize(mask, (img_w, img_h))

#                 cone = Cone(
#                     index=-1, color=color,
#                     cx=bbox_cx, cy=bbox_cy,
#                     bbox=(x1, y1, x2, y2),
#                     confidence=float(box.conf[0]),
#                     mask=mask_full
#                 )
#                 (red_cones if color == "red" else blue_cones).append(cone)

#         # x좌표(BBOX 중앙) 기준 정렬 및 인덱스 부여
#         for lst in (red_cones, blue_cones):
#             lst.sort(key=lambda c: c.cx)
#             for idx, c in enumerate(lst):
#                 c.index = idx

#         pairs = self._match_pairs(red_cones, blue_cones)
#         return undist, pairs, {"red": red_cones, "blue": blue_cones}

#     def _match_pairs(self, reds, blues):
#         pairs = []
#         for i in range(min(len(reds), len(blues))):
#             r, b   = reds[i], blues[i]
#             mid_x  = (r.cx + b.cx) / 2
#             mid_y  = (r.cy + b.cy) / 2
#             bearing = self.camera.pixel_to_bearing(mid_x)
#             pairs.append(ConePair(i, r, b, mid_x, mid_y, bearing))
#         return pairs


# # ── 시각화 ───────────────────────────────────────────────────────────────────

# def draw_results(frame: np.ndarray, pairs, all_cones, img_w: int) -> np.ndarray:
#     vis    = frame.copy()
#     cx_img = img_w / 2

#     # ① 마스크 오버레이 (진하게)
#     for color, cones in all_cones.items():
#         bgr = MASK_COLORS[color]
#         for cone in cones:
#             if cone.mask is None:
#                 continue
#             overlay = vis.copy()
#             overlay[cone.mask > 0.5] = bgr
#             vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)

#     # ② BBOX + 중앙점 + 레이블
#     for color, cones in all_cones.items():
#         b_col = BBOX_COLORS[color]
#         for cone in cones:
#             x1, y1, x2, y2 = cone.bbox
#             cx, cy = int(cone.cx), int(cone.cy)

#             # BBOX
#             cv2.rectangle(vis, (x1, y1), (x2, y2), b_col, 2)

#             # BBOX 중앙점
#             cv2.circle(vis, (cx, cy), 5, b_col, -1)
#             cv2.circle(vis, (cx, cy), 7, (255, 255, 255), 1)

#             # 레이블 (BBOX 상단)
#             label = f"{color[0].upper()}{cone.index} {cone.confidence:.2f}"
#             lx, ly = x1, max(y1 - 8, 14)
#             cv2.putText(vis, label, (lx, ly),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_col, 2)

#     # ③ 카메라 중심선
#     cv2.line(vis, (int(cx_img), 0), (int(cx_img), vis.shape[0]),
#              AXIS_COLOR, 1, cv2.LINE_AA)

#     # ④ 쌍 연결선 + 중간점 + Bearing 표시
#     for pair in pairs:
#         rx, ry = int(pair.red.cx),  int(pair.red.cy)
#         bx, by = int(pair.blue.cx), int(pair.blue.cy)
#         mx, my = int(pair.midpoint_x), int(pair.midpoint_y)

#         # BBOX 중앙점 연결선
#         cv2.line(vis, (rx, ry), (bx, by), MID_COLOR, 2, cv2.LINE_AA)

#         # 중간점
#         cv2.circle(vis, (mx, my), 8, MID_COLOR, -1)
#         cv2.circle(vis, (mx, my), 10, (255, 255, 255), 1)

#         # Bearing 텍스트
#         d = "R" if pair.bearing_deg > 0 else ("L" if pair.bearing_deg < 0 else "C")
#         txt = f"Pair{pair.pair_index}: {pair.bearing_deg:+.1f}deg {d}"
#         cv2.putText(vis, txt, (mx + 12, my + 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, MID_COLOR, 2)

#     # ⑤ 상단 요약 패널
#     y_off = 26
#     for pair in pairs:
#         s = (f"[Pair{pair.pair_index}] "
#              f"R{pair.red.index}({pair.red.cx:.0f},{pair.red.cy:.0f}) "
#              f"<-> B{pair.blue.index}({pair.blue.cx:.0f},{pair.blue.cy:.0f}) "
#              f"| mid=({pair.midpoint_x:.0f},{pair.midpoint_y:.0f}) "
#              f"| bearing={pair.bearing_deg:+.2f}deg")
#         cv2.putText(vis, s, (8, y_off),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
#         y_off += 20

#     return vis


# def print_results(pairs, all_cones):
#     print("\n" + "="*60)
#     print(f"  Red:{len(all_cones['red'])}  Blue:{len(all_cones['blue'])}"
#           f"  Pairs:{len(pairs)}")
#     print("-"*60)
#     for p in pairs:
#         d = "RIGHT" if p.bearing_deg > 1 else ("LEFT" if p.bearing_deg < -1 else "CENTER")
#         print(f"  Pair{p.pair_index}: "
#               f"R{p.red.index}  bbox_cx={p.red.cx:.0f}px  <->  "
#               f"B{p.blue.index}  bbox_cx={p.blue.cx:.0f}px")
#         print(f"           mid=({p.midpoint_x:.1f}, {p.midpoint_y:.1f})  "
#               f"Bearing={p.bearing_deg:+.2f}deg [{d}]")
#     print("="*60)


# # ── 진입점 ───────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(description="Cone Detector + Bearing Angle")
#     parser.add_argument("--source",  required=True)
#     parser.add_argument("--model",   required=True)
#     parser.add_argument("--calib",   default=None)
#     parser.add_argument("--no-calib", action="store_true")
#     parser.add_argument("--hfov",    type=float, default=195.0)
#     parser.add_argument("--conf",    type=float, default=0.4)
#     parser.add_argument("--min-area", type=float, default=0.003,
#                         help="BBOX 면적 최소 비율 (default: 0.003 = 0.3%%)")
#     parser.add_argument("--save",    action="store_true")
#     args = parser.parse_args()

#     camera = (ApproxCamera(640, 480, args.hfov)
#               if args.no_calib or args.calib is None
#               else PinholeCamera(args.calib))

#     detector = ConeDetector(args.model, camera,
#                             conf_thresh=args.conf,
#                             min_area_ratio=args.min_area)

#     try:
#         source = int(args.source)
#         cap, is_image = cv2.VideoCapture(source), False
#     except ValueError:
#         p = Path(args.source)
#         is_image = p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
#         cap = None if is_image else cv2.VideoCapture(str(p))

#     writer = None
#     if args.save and not is_image:
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         writer = cv2.VideoWriter("output.mp4", fourcc, 30, (1280, 720))

#     cv2.namedWindow("Cone Detector", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Cone Detector", 1280, 720)

#     if is_image:
#         frame = cv2.imread(args.source)
#         undist, pairs, all_cones = detector.detect(frame)
#         print_results(pairs, all_cones)
#         vis = draw_results(undist, pairs, all_cones, undist.shape[1])
#         cv2.imshow("Cone Detector", vis)
#         if args.save:
#             cv2.imwrite("output.jpg", vis)
#         cv2.waitKey(0)
#     else:
#         print("[INFO] Running... press q to quit")
#         fid = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             undist, pairs, all_cones = detector.detect(frame)
#             vis = draw_results(undist, pairs, all_cones, undist.shape[1])
#             if fid % 30 == 0:
#                 print_results(pairs, all_cones)
#             cv2.imshow("Cone Detector", vis)
#             if writer:
#                 writer.write(vis)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#             fid += 1
#         cap.release()
#         if writer:
#             writer.release()

#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

# """
# python3 cone_detector.py --source 0 --model best.pt --conf 0.83 --calib calibration.yaml
# """

# """
# Step 3: 꼬깔 검출 + 인덱싱 + 쌍 매칭 + Bearing Angle 계산
# =============================================================
# 사용법:
#   python cone_detector.py --source 0 --calib calibration.yaml --model best.pt
#   python cone_detector.py --source test.jpg --calib calibration.yaml --model best.pt
#   python cone_detector.py --source test.mp4 --calib calibration.yaml --model best.pt --save
# """

# import cv2
# import numpy as np
# import yaml
# import argparse
# import math
# from pathlib import Path
# from dataclasses import dataclass, field
# from typing import Optional, Tuple
# from ultralytics import YOLO


# # ── 데이터 구조 ─────────────────────────────────────────────────────────────

# @dataclass
# class Cone:
#     index: int              # 클래스 내 인덱스 (x좌표 기준 좌→우)
#     color: str              # "red" or "blue"
#     cx: float               # BBOX 중앙점 x (픽셀)
#     cy: float               # BBOX 중앙점 y (픽셀)
#     bbox: Tuple             # (x1, y1, x2, y2)
#     confidence: float
#     mask: Optional[np.ndarray] = field(default=None, repr=False)

# @dataclass
# class ConePair:
#     pair_index: int
#     red: Cone
#     blue: Cone
#     midpoint_x: float       # BBOX 중앙점 연결선의 중점 x
#     midpoint_y: float
#     bearing_deg: float      # Bearing angle (양수=오른쪽, 음수=왼쪽)


# # ── 클래스 설정 ──────────────────────────────────────────────────────────────
# CLASS_COLOR_MAP = {
#     "red_cone":  "red",
#     "blue_cone": "blue",
# }

# # BGR
# MASK_COLORS = {
#     "red":  (0,   50, 255),
#     "blue": (255, 80,   0),
# }
# BBOX_COLORS = {
#     "red":  (0,   0,  220),
#     "blue": (220, 60,   0),
# }
# MID_COLOR   = (0, 255, 255)   # 중간점 / 연결선
# AXIS_COLOR  = (180, 180, 180) # 카메라 중심선


# # ── 카메라 모델 ──────────────────────────────────────────────────────────────

# class PinholeCamera:
#     """일반 핀홀 카메라 (ELP-USB500W02M-BL36, 3.6mm 렌즈)"""

#     def __init__(self, calib_path: str):
#         with open(calib_path, "r") as f:
#             data = yaml.safe_load(f)

#         self.K = np.array(data["camera_matrix"]["data"],
#                           dtype=np.float64).reshape(3, 3)
#         self.D = np.array(data["distortion_coefficients"]["data"],
#                           dtype=np.float64).reshape(1, 5)
#         self.map1 = None
#         self.map2 = None
#         self.last_size = None
#         self.fx   = self.K[0, 0]
#         self.cx_p = self.K[0, 2]
#         print(f"[Camera] Pinhole | fx={self.fx:.1f}  cx={self.cx_p:.1f}")

#     def _build_map(self, h: int, w: int):
#         """프레임 크기에 맞게 undistortion 맵 생성"""
#         if self.last_size == (w, h):
#             return
#         self.last_size = (w, h)
#         # alpha=0: 검은 영역 완전 없애고 유효 픽셀만
#         K_new, _ = cv2.getOptimalNewCameraMatrix(
#             self.K, self.D, (w, h), alpha=0)
#         self.map1, self.map2 = cv2.initUndistortRectifyMap(
#             self.K, self.D, None, K_new, (w, h), cv2.CV_16SC2)
#         self.fx   = K_new[0, 0]
#         self.cx_p = K_new[0, 2]
#         print(f"[Camera] Map built for {w}x{h} | fx={self.fx:.1f} cx={self.cx_p:.1f}")

#     def undistort(self, img: np.ndarray) -> np.ndarray:
#         h, w = img.shape[:2]
#         self._build_map(h, w)
#         return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

#     def pixel_to_bearing(self, px: float) -> float:
#         return math.degrees(math.atan2(px - self.cx_p, self.fx))


# class ApproxCamera:
#     def __init__(self, W: int, H: int, hfov_deg: float):
#         self.W  = W
#         self.H  = H
#         self.cx_p = W / 2.0
#         self.roi  = (0, 0, W, H)
#         hfov_rad  = math.radians(hfov_deg)
#         self.f_eq = W / hfov_rad
#         print(f"[Camera] Approx | hfov={hfov_deg}deg  f_eq={self.f_eq:.1f}")

#     def undistort(self, img: np.ndarray) -> np.ndarray:
#         return img

#     def pixel_to_bearing(self, px: float) -> float:
#         return math.degrees((px - self.cx_p) / self.f_eq)


# # ── 검출기 ───────────────────────────────────────────────────────────────────

# class ConeDetector:
#     def __init__(self, model_path: str, camera, conf_thresh: float = 0.4,
#                  min_area_ratio: float = 0.003):
#         self.model         = YOLO(model_path)
#         self.camera        = camera
#         self.conf_thresh   = conf_thresh
#         self.min_area_ratio = min_area_ratio
#         print(f"[YOLO] {model_path}  classes={self.model.names}")

#     def detect(self, frame: np.ndarray):
#         undist  = self.camera.undistort(frame)
#         results = self.model(undist, conf=self.conf_thresh, verbose=False)[0]

#         red_cones, blue_cones = [], []

#         if results.masks is not None:
#             masks_data = results.masks.data.cpu().numpy()
#             img_h, img_w = undist.shape[:2]
#             img_area = img_h * img_w

#             for i, box in enumerate(results.boxes):
#                 cls_name = self.model.names[int(box.cls[0])]
#                 color    = CLASS_COLOR_MAP.get(cls_name)
#                 if color is None:
#                     continue

#                 # ── BBOX (xyxy, 원본 이미지 좌표계)
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                 x1 = max(0, x1); y1 = max(0, y1)
#                 x2 = min(img_w, x2); y2 = min(img_h, y2)
#                 bbox_cx = (x1 + x2) / 2.0
#                 bbox_cy = (y1 + y2) / 2.0
#                 bbox_area = (x2 - x1) * (y2 - y1)

#                 # 면적 필터 (오탐 제거)
#                 if bbox_area / img_area < self.min_area_ratio:
#                     continue

#                 # 마스크 → 원본 해상도로 리사이즈
#                 mask = masks_data[i]
#                 mask_full = cv2.resize(mask, (img_w, img_h))

#                 cone = Cone(
#                     index=-1, color=color,
#                     cx=bbox_cx, cy=bbox_cy,
#                     bbox=(x1, y1, x2, y2),
#                     confidence=float(box.conf[0]),
#                     mask=mask_full
#                 )
#                 (red_cones if color == "red" else blue_cones).append(cone)

#         # x좌표(BBOX 중앙) 기준 정렬 및 인덱스 부여
#         for lst in (red_cones, blue_cones):
#             lst.sort(key=lambda c: c.cx)
#             for idx, c in enumerate(lst):
#                 c.index = idx

#         pairs = self._match_pairs(red_cones, blue_cones)
#         return undist, pairs, {"red": red_cones, "blue": blue_cones}

#     def _match_pairs(self, reds, blues):
#         pairs = []
#         for i in range(min(len(reds), len(blues))):
#             r, b   = reds[i], blues[i]
#             mid_x  = (r.cx + b.cx) / 2
#             mid_y  = (r.cy + b.cy) / 2
#             bearing = self.camera.pixel_to_bearing(mid_x)
#             pairs.append(ConePair(i, r, b, mid_x, mid_y, bearing))
#         return pairs


# # ── 시각화 ───────────────────────────────────────────────────────────────────

# def draw_results(frame: np.ndarray, pairs, all_cones, img_w: int) -> np.ndarray:
#     vis    = frame.copy()
#     cx_img = img_w / 2

#     # ① 마스크 오버레이 (진하게)
#     for color, cones in all_cones.items():
#         bgr = MASK_COLORS[color]
#         for cone in cones:
#             if cone.mask is None:
#                 continue
#             overlay = vis.copy()
#             overlay[cone.mask > 0.5] = bgr
#             vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)

#     # ② BBOX + 중앙점 + 레이블
#     for color, cones in all_cones.items():
#         b_col = BBOX_COLORS[color]
#         for cone in cones:
#             x1, y1, x2, y2 = cone.bbox
#             cx, cy = int(cone.cx), int(cone.cy)

#             # BBOX
#             cv2.rectangle(vis, (x1, y1), (x2, y2), b_col, 2)

#             # BBOX 중앙점
#             cv2.circle(vis, (cx, cy), 5, b_col, -1)
#             cv2.circle(vis, (cx, cy), 7, (255, 255, 255), 1)

#             # 레이블 (BBOX 상단)
#             label = f"{color[0].upper()}{cone.index} {cone.confidence:.2f}"
#             lx, ly = x1, max(y1 - 8, 14)
#             cv2.putText(vis, label, (lx, ly),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_col, 2)

#     # ③ 카메라 중심선
#     cv2.line(vis, (int(cx_img), 0), (int(cx_img), vis.shape[0]),
#              AXIS_COLOR, 1, cv2.LINE_AA)

#     # ④ 쌍 연결선 + 중간점 + Bearing 표시
#     for pair in pairs:
#         rx, ry = int(pair.red.cx),  int(pair.red.cy)
#         bx, by = int(pair.blue.cx), int(pair.blue.cy)
#         mx, my = int(pair.midpoint_x), int(pair.midpoint_y)

#         # BBOX 중앙점 연결선
#         cv2.line(vis, (rx, ry), (bx, by), MID_COLOR, 2, cv2.LINE_AA)

#         # 중간점
#         cv2.circle(vis, (mx, my), 8, MID_COLOR, -1)
#         cv2.circle(vis, (mx, my), 10, (255, 255, 255), 1)

#         # Bearing 텍스트
#         d = "R" if pair.bearing_deg > 0 else ("L" if pair.bearing_deg < 0 else "C")
#         txt = f"Pair{pair.pair_index}: {pair.bearing_deg:+.1f}deg {d}"
#         cv2.putText(vis, txt, (mx + 12, my + 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, MID_COLOR, 2)

#     # ⑤ 상단 요약 패널
#     y_off = 26
#     for pair in pairs:
#         s = (f"[Pair{pair.pair_index}] "
#              f"R{pair.red.index}({pair.red.cx:.0f},{pair.red.cy:.0f}) "
#              f"<-> B{pair.blue.index}({pair.blue.cx:.0f},{pair.blue.cy:.0f}) "
#              f"| mid=({pair.midpoint_x:.0f},{pair.midpoint_y:.0f}) "
#              f"| bearing={pair.bearing_deg:+.2f}deg")
#         cv2.putText(vis, s, (8, y_off),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
#         y_off += 20

#     return vis


# def print_results(pairs, all_cones):
#     print("\n" + "="*60)
#     print(f"  Red:{len(all_cones['red'])}  Blue:{len(all_cones['blue'])}"
#           f"  Pairs:{len(pairs)}")
#     print("-"*60)
#     for p in pairs:
#         d = "RIGHT" if p.bearing_deg > 1 else ("LEFT" if p.bearing_deg < -1 else "CENTER")
#         print(f"  Pair{p.pair_index}: "
#               f"R{p.red.index}  bbox_cx={p.red.cx:.0f}px  <->  "
#               f"B{p.blue.index}  bbox_cx={p.blue.cx:.0f}px")
#         print(f"           mid=({p.midpoint_x:.1f}, {p.midpoint_y:.1f})  "
#               f"Bearing={p.bearing_deg:+.2f}deg [{d}]")
#     print("="*60)


# # ── 진입점 ───────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(description="Cone Detector + Bearing Angle")
#     parser.add_argument("--source",  required=True)
#     parser.add_argument("--model",   required=True)
#     parser.add_argument("--calib",   default=None)
#     parser.add_argument("--no-calib", action="store_true")
#     parser.add_argument("--hfov",    type=float, default=195.0)
#     parser.add_argument("--conf",    type=float, default=0.4)
#     parser.add_argument("--min-area", type=float, default=0.003,
#                         help="BBOX 면적 최소 비율 (default: 0.003 = 0.3%%)")
#     parser.add_argument("--save",    action="store_true")
#     args = parser.parse_args()

#     camera = (ApproxCamera(1280, 720, args.hfov)
#               if args.no_calib or args.calib is None
#               else PinholeCamera(args.calib))

#     detector = ConeDetector(args.model, camera,
#                             conf_thresh=args.conf,
#                             min_area_ratio=args.min_area)

#     try:
#         source = int(args.source)
#         cap, is_image = cv2.VideoCapture(source), False
#     except ValueError:
#         p = Path(args.source)
#         is_image = p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
#         cap = None if is_image else cv2.VideoCapture(str(p))

#     writer = None
#     if args.save and not is_image:
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         writer = cv2.VideoWriter("output.mp4", fourcc, 30, (1280, 720))

#     cv2.namedWindow("Cone Detector", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Cone Detector", 1280, 720)

#     if is_image:
#         frame = cv2.imread(args.source)
#         undist, pairs, all_cones = detector.detect(frame)
#         print_results(pairs, all_cones)
#         vis = draw_results(undist, pairs, all_cones, undist.shape[1])
#         cv2.imshow("Cone Detector", vis)
#         if args.save:
#             cv2.imwrite("output.jpg", vis)
#         cv2.waitKey(0)
#     else:
#         print("[INFO] Running... press q to quit")
#         fid = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             undist, pairs, all_cones = detector.detect(frame)
#             vis = draw_results(undist, pairs, all_cones, undist.shape[1])
#             if fid % 30 == 0:
#                 print_results(pairs, all_cones)
#             cv2.imshow("Cone Detector", vis)
#             if writer:
#                 writer.write(vis)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#             fid += 1
#         cap.release()
#         if writer:
#             writer.release()

#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()


"""
Step 3: 꼬깔 검출 + 인덱싱 + 쌍 매칭 + Bearing Angle 계산
=============================================================
사용법:
  python cone_detector.py --source 0 --calib calibration.yaml --model best.pt
  python cone_detector.py --source test.jpg --calib calibration.yaml --model best.pt
  python cone_detector.py --source test.mp4 --calib calibration.yaml --model best.pt --save
"""

import cv2
import numpy as np
import yaml
import argparse
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple
from ultralytics import YOLO


# ── 데이터 구조 ─────────────────────────────────────────────────────────────

@dataclass
class Cone:
    index: int              # 클래스 내 인덱스 (x좌표 기준 좌→우)
    color: str              # "red" or "blue"
    cx: float               # BBOX 중앙점 x (픽셀)
    cy: float               # BBOX 중앙점 y (픽셀)
    bbox: Tuple             # (x1, y1, x2, y2)
    confidence: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class ConePair:
    pair_index: int
    red: Cone
    blue: Cone
    midpoint_x: float       # BBOX 중앙점 연결선의 중점 x
    midpoint_y: float
    bearing_deg: float      # Bearing angle (양수=오른쪽, 음수=왼쪽)


# ── 클래스 설정 ──────────────────────────────────────────────────────────────
CLASS_COLOR_MAP = {
    "red_cone":  "red",
    "blue_cone": "blue",
}

# BGR
MASK_COLORS = {
    "red":  (0,   50, 255),
    "blue": (255, 80,   0),
}
BBOX_COLORS = {
    "red":  (0,   0,  220),
    "blue": (220, 60,   0),
}
MID_COLOR   = (0, 255, 255)   # 중간점 / 연결선
AXIS_COLOR  = (180, 180, 180) # 카메라 중심선


# ── 카메라 모델 ──────────────────────────────────────────────────────────────

class PinholeCamera:
    """일반 핀홀 카메라 (ELP-USB500W02M-BL36, 3.6mm 렌즈)"""

    def __init__(self, calib_path: str):
        with open(calib_path, "r") as f:
            data = yaml.safe_load(f)

        self.calib_W = data["image_width"]
        self.calib_H = data["image_height"]
        self.K = np.array(data["camera_matrix"]["data"],
                          dtype=np.float64).reshape(3, 3)
        self.D = np.array(data["distortion_coefficients"]["data"],
                          dtype=np.float64).reshape(1, 5)
        self.map1 = None
        self.map2 = None
        self.last_size = None
        self.fx   = self.K[0, 0]
        self.cx_p = self.K[0, 2]
        print(f"[Camera] Pinhole | calib={self.calib_W}x{self.calib_H} fx={self.fx:.1f}")

    def _build_map(self, h: int, w: int):
        """프레임 크기에 맞게 undistortion 맵 생성 (해상도 스케일 보정)"""
        if self.last_size == (w, h):
            return
        self.last_size = (w, h)

        # 캘리브레이션 해상도 기준 스케일 비율 계산
        sx = w / self.calib_W
        sy = h / self.calib_H

        # K를 현재 해상도에 맞게 스케일
        K_scaled = self.K.copy()
        K_scaled[0, 0] *= sx  # fx
        K_scaled[1, 1] *= sy  # fy
        K_scaled[0, 2] *= sx  # cx
        K_scaled[1, 2] *= sy  # cy

        K_new, _ = cv2.getOptimalNewCameraMatrix(
            K_scaled, self.D, (w, h), alpha=0)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K_scaled, self.D, None, K_new, (w, h), cv2.CV_16SC2)
        self.fx   = K_new[0, 0]
        self.cx_p = K_new[0, 2]
        print(f"[Camera] Map built for {w}x{h} (calib={self.calib_W}x{self.calib_H})")
        print(f"[Camera] scale=({sx:.2f},{sy:.2f}) fx={self.fx:.1f} cx={self.cx_p:.1f}")

    def undistort(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        self._build_map(h, w)
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def pixel_to_bearing(self, px: float) -> float:
        return math.degrees(math.atan2(px - self.cx_p, self.fx))


class ApproxCamera:
    def __init__(self, W: int, H: int, hfov_deg: float):
        self.W  = W
        self.H  = H
        self.cx_p = W / 2.0
        self.roi  = (0, 0, W, H)
        hfov_rad  = math.radians(hfov_deg)
        self.f_eq = W / hfov_rad
        print(f"[Camera] Approx | hfov={hfov_deg}deg  f_eq={self.f_eq:.1f}")

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return img

    def pixel_to_bearing(self, px: float) -> float:
        return math.degrees((px - self.cx_p) / self.f_eq)


# ── 검출기 ───────────────────────────────────────────────────────────────────

class ConeDetector:
    def __init__(self, model_path: str, camera, conf_thresh: float = 0.4,
                 min_area_ratio: float = 0.003):
        self.model         = YOLO(model_path)
        self.camera        = camera
        self.conf_thresh   = conf_thresh
        self.min_area_ratio = min_area_ratio
        print(f"[YOLO] {model_path}  classes={self.model.names}")

    def detect(self, frame: np.ndarray):
        undist  = self.camera.undistort(frame)
        results = self.model(undist, conf=self.conf_thresh, verbose=False)[0]

        red_cones, blue_cones = [], []

        if results.masks is not None:
            masks_data = results.masks.data.cpu().numpy()
            img_h, img_w = undist.shape[:2]
            img_area = img_h * img_w

            for i, box in enumerate(results.boxes):
                cls_name = self.model.names[int(box.cls[0])]
                color    = CLASS_COLOR_MAP.get(cls_name)
                if color is None:
                    continue

                # ── BBOX (xyxy, 원본 이미지 좌표계)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
                bbox_cx = (x1 + x2) / 2.0
                bbox_cy = (y1 + y2) / 2.0
                bbox_area = (x2 - x1) * (y2 - y1)

                # 면적 필터 (오탐 제거)
                if bbox_area / img_area < self.min_area_ratio:
                    continue

                # 마스크 → 원본 해상도로 리사이즈
                mask = masks_data[i]
                mask_full = cv2.resize(mask, (img_w, img_h))

                cone = Cone(
                    index=-1, color=color,
                    cx=bbox_cx, cy=bbox_cy,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    mask=mask_full
                )
                (red_cones if color == "red" else blue_cones).append(cone)

        # x좌표(BBOX 중앙) 기준 정렬 및 인덱스 부여
        for lst in (red_cones, blue_cones):
            lst.sort(key=lambda c: c.cx)
            for idx, c in enumerate(lst):
                c.index = idx

        pairs = self._match_pairs(red_cones, blue_cones)
        return undist, pairs, {"red": red_cones, "blue": blue_cones}

    def _match_pairs(self, reds, blues):
        pairs = []
        for i in range(min(len(reds), len(blues))):
            r, b   = reds[i], blues[i]
            mid_x  = (r.cx + b.cx) / 2
            mid_y  = (r.cy + b.cy) / 2
            bearing = self.camera.pixel_to_bearing(mid_x)
            pairs.append(ConePair(i, r, b, mid_x, mid_y, bearing))
        return pairs


# ── 시각화 ───────────────────────────────────────────────────────────────────

def draw_results(frame: np.ndarray, pairs, all_cones, img_w: int) -> np.ndarray:
    vis    = frame.copy()
    cx_img = img_w / 2

    # ① 마스크 오버레이 (진하게)
    for color, cones in all_cones.items():
        bgr = MASK_COLORS[color]
        for cone in cones:
            if cone.mask is None:
                continue
            overlay = vis.copy()
            overlay[cone.mask > 0.5] = bgr
            vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)

    # ② BBOX + 중앙점 + 레이블
    for color, cones in all_cones.items():
        b_col = BBOX_COLORS[color]
        for cone in cones:
            x1, y1, x2, y2 = cone.bbox
            cx, cy = int(cone.cx), int(cone.cy)

            # BBOX
            cv2.rectangle(vis, (x1, y1), (x2, y2), b_col, 2)

            # BBOX 중앙점
            cv2.circle(vis, (cx, cy), 5, b_col, -1)
            cv2.circle(vis, (cx, cy), 7, (255, 255, 255), 1)

            # 레이블 (BBOX 상단)
            label = f"{color[0].upper()}{cone.index} {cone.confidence:.2f}"
            lx, ly = x1, max(y1 - 8, 14)
            cv2.putText(vis, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_col, 2)

    # ③ 카메라 중심선
    cv2.line(vis, (int(cx_img), 0), (int(cx_img), vis.shape[0]),
             AXIS_COLOR, 1, cv2.LINE_AA)

    # ④ 쌍 연결선 + 중간점 + Bearing 표시
    for pair in pairs:
        rx, ry = int(pair.red.cx),  int(pair.red.cy)
        bx, by = int(pair.blue.cx), int(pair.blue.cy)
        mx, my = int(pair.midpoint_x), int(pair.midpoint_y)

        # BBOX 중앙점 연결선
        cv2.line(vis, (rx, ry), (bx, by), MID_COLOR, 2, cv2.LINE_AA)

        # 중간점
        cv2.circle(vis, (mx, my), 8, MID_COLOR, -1)
        cv2.circle(vis, (mx, my), 10, (255, 255, 255), 1)

        # Bearing 텍스트
        d = "R" if pair.bearing_deg > 0 else ("L" if pair.bearing_deg < 0 else "C")
        txt = f"Pair{pair.pair_index}: {pair.bearing_deg:+.1f}deg {d}"
        cv2.putText(vis, txt, (mx + 12, my + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, MID_COLOR, 2)

    # ⑤ 상단 요약 패널
    y_off = 26
    for pair in pairs:
        s = (f"[Pair{pair.pair_index}] "
             f"R{pair.red.index}({pair.red.cx:.0f},{pair.red.cy:.0f}) "
             f"<-> B{pair.blue.index}({pair.blue.cx:.0f},{pair.blue.cy:.0f}) "
             f"| mid=({pair.midpoint_x:.0f},{pair.midpoint_y:.0f}) "
             f"| bearing={pair.bearing_deg:+.2f}deg")
        cv2.putText(vis, s, (8, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y_off += 20

    return vis


def print_results(pairs, all_cones):
    print("\n" + "="*60)
    print(f"  Red:{len(all_cones['red'])}  Blue:{len(all_cones['blue'])}"
          f"  Pairs:{len(pairs)}")
    print("-"*60)
    for p in pairs:
        d = "RIGHT" if p.bearing_deg > 1 else ("LEFT" if p.bearing_deg < -1 else "CENTER")
        print(f"  Pair{p.pair_index}: "
              f"R{p.red.index}  bbox_cx={p.red.cx:.0f}px  <->  "
              f"B{p.blue.index}  bbox_cx={p.blue.cx:.0f}px")
        print(f"           mid=({p.midpoint_x:.1f}, {p.midpoint_y:.1f})  "
              f"Bearing={p.bearing_deg:+.2f}deg [{d}]")
    print("="*60)


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cone Detector + Bearing Angle")
    parser.add_argument("--source",  required=True)
    parser.add_argument("--model",   required=True)
    parser.add_argument("--calib",   default=None)
    parser.add_argument("--no-calib", action="store_true")
    parser.add_argument("--hfov",    type=float, default=195.0)
    parser.add_argument("--conf",    type=float, default=0.4)
    parser.add_argument("--min-area", type=float, default=0.003,
                        help="BBOX 면적 최소 비율 (default: 0.003 = 0.3%%)")
    parser.add_argument("--save",    action="store_true")
    args = parser.parse_args()

    camera = (ApproxCamera(640, 480, args.hfov)
              if args.no_calib or args.calib is None
              else PinholeCamera(args.calib))

    detector = ConeDetector(args.model, camera,
                            conf_thresh=args.conf,
                            min_area_ratio=args.min_area)

    try:
        source = int(args.source)
        cap = cv2.VideoCapture(source)
        # 캘리브레이션 해상도로 강제 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAP] 실제 해상도: {actual_w}x{actual_h}")
        is_image = False
    except ValueError:
        p = Path(args.source)
        is_image = p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        cap = None if is_image else cv2.VideoCapture(str(p))

    writer = None
    if args.save and not is_image:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, 30, (640, 480))

    cv2.namedWindow("Cone Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cone Detector", 640, 480)

    if is_image:
        frame = cv2.imread(args.source)
        undist, pairs, all_cones = detector.detect(frame)
        print_results(pairs, all_cones)
        vis = draw_results(undist, pairs, all_cones, undist.shape[1])
        cv2.imshow("Cone Detector", vis)
        if args.save:
            cv2.imwrite("output.jpg", vis)
        cv2.waitKey(0)
    else:
        print("[INFO] Running... press q to quit")
        fid = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            undist, pairs, all_cones = detector.detect(frame)
            vis = draw_results(undist, pairs, all_cones, undist.shape[1])
            if fid % 30 == 0:
                print_results(pairs, all_cones)
            cv2.imshow("Cone Detector", vis)
            if writer:
                writer.write(vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            fid += 1
        cap.release()
        if writer:
            writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()