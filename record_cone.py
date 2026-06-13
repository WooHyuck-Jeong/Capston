"""
record_cone.py
USB 카메라로 꼬깔 동영상을 촬영하는 스크립트

사용법:
    python record_cone.py --output red_cone.mp4
    python record_cone.py --output blue_cone.mp4 --camera 1
"""

import cv2
import argparse
import sys
from datetime import datetime


def get_backend() -> int:
    """OS에 맞는 카메라 백엔드 반환"""
    import platform
    if platform.system() == 'Windows':
        return cv2.CAP_DSHOW
    else:
        return cv2.CAP_V4L2  # Linux/Ubuntu


def list_cameras(max_index: int = 5) -> list[int]:
    """사용 가능한 카메라 인덱스 목록 반환"""
    backend = get_backend()
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def record_video(
    output_path: str,
    camera_index: int = 0,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
) -> None:
    """
    USB 카메라로 동영상 촬영

    Parameters
    ----------
    output_path  : 저장할 mp4 파일 경로
    camera_index : 카메라 인덱스 (기본 0)
    fps          : 프레임레이트 (기본 30)
    width        : 해상도 가로 (기본 1280)
    height       : 해상도 세로 (기본 720)
    """

    # ── 카메라 오픈 ──────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index, get_backend())
    if not cap.isOpened():
        print(f"[ERROR] 카메라 인덱스 {camera_index} 를 열 수 없습니다.")
        available = list_cameras()
        if available:
            print(f"  사용 가능한 카메라 인덱스: {available}")
            print(f"  --camera 옵션으로 인덱스를 지정하세요.")
        else:
            print("  연결된 카메라를 찾을 수 없습니다. USB 연결을 확인하세요.")
        sys.exit(1)

    # 해상도 및 FPS 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 실제 적용된 값 확인
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = fps  # 카메라가 FPS를 보고하지 않으면 지정값 사용

    print(f"[INFO] 카메라 인덱스 : {camera_index}")
    print(f"[INFO] 해상도        : {actual_w} x {actual_h}")
    print(f"[INFO] FPS           : {actual_fps:.1f}")
    print(f"[INFO] 저장 파일     : {output_path}")
    print()
    print("  [SPACE] 녹화 시작/일시정지")
    print("  [Q / ESC] 녹화 종료 및 저장")
    print()

    # ── VideoWriter 설정 ─────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (actual_w, actual_h))

    recording = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임을 읽을 수 없습니다. 카메라 연결을 확인하세요.")
            break

        # 녹화 중이면 프레임 저장
        if recording:
            writer.write(frame)
            frame_count += 1

        # ── 화면 오버레이 ────────────────────────────────────────
        display = frame.copy()
        status_text = "● REC" if recording else "■ PAUSE"
        status_color = (0, 0, 255) if recording else (0, 200, 255)

        cv2.putText(display, status_text, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
        cv2.putText(display, f"Frames: {frame_count}", (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(display, f"Output: {output_path}", (15, actual_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(display, "SPACE: Start/Pause  |  Q/ESC: Save & Quit",
                    (15, actual_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Cone Recording", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):                    # 스페이스바 → 시작/일시정지
            recording = not recording
            state = "시작" if recording else "일시정지"
            print(f"[INFO] 녹화 {state} (저장 프레임: {frame_count})")
        elif key in (ord("q"), ord("Q"), 27):  # Q 또는 ESC → 종료
            break

    # ── 정리 ────────────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print()
    print(f"[INFO] 녹화 완료: 총 {frame_count} 프레임 → {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="USB 카메라로 꼬깔 동영상 촬영",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output",  "-o", default="cone.mp4",
                        help="저장할 mp4 파일 이름")
    parser.add_argument("--camera",  "-c", type=int, default=0,
                        help="카메라 인덱스 (기본 내장 카메라=0, USB=1 이상)")
    parser.add_argument("--fps",     "-f", type=float, default=30.0,
                        help="녹화 FPS")
    parser.add_argument("--width",   "-W", type=int, default=1280,
                        help="해상도 가로")
    parser.add_argument("--height",  "-H", type=int, default=720,
                        help="해상도 세로")
    parser.add_argument("--list-cameras", action="store_true",
                        help="사용 가능한 카메라 목록 출력 후 종료")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_cameras:
        cams = list_cameras()
        if cams:
            print(f"사용 가능한 카메라 인덱스: {cams}")
        else:
            print("연결된 카메라를 찾을 수 없습니다.")
        sys.exit(0)

    record_video(
        output_path=args.output,
        camera_index=args.camera,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )