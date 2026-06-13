"""
연결된 카메라 자동 탐색 스크립트
==================================
사용법:
  python find_cameras.py

동작:
  - 인덱스 0~9번까지 순서대로 카메라 연결 시도
  - 연결 가능한 카메라 목록 출력
  - 각 카메라 미리보기 창 표시 (아무 키 누르면 다음 카메라로)
"""

import cv2


def find_cameras(max_index: int = 10):
    print("연결된 카메라 탐색 중...\n")

    found = []

    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            found.append({"index": i, "w": w, "h": h, "fps": fps})
            print(f"  [발견] 카메라 {i}번 : {w}x{h} @ {fps:.0f}fps")
            cap.release()
        else:
            print(f"  [ -- ] 카메라 {i}번 : 없음")
            cap.release()

    print()

    if not found:
        print("[ERROR] 연결된 카메라가 없습니다.")
        print("  → USB 연결 상태를 확인하세요.")
        return

    print(f"총 {len(found)}개 카메라 발견: {[c['index'] for c in found]}")
    print("\n각 카메라 미리보기를 순서대로 표시합니다.")
    print("아무 키를 누르면 다음으로 넘어갑니다. (q: 종료)\n")

    for cam in found:
        idx = cam["index"]
        cap = cv2.VideoCapture(idx)

        print(f"[미리보기] 카메라 {idx}번 — 아무 키를 누르세요...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"  [WARNING] 카메라 {idx}번 프레임 읽기 실패")
                break

            # 카메라 정보 오버레이
            label = f"Camera {idx}  |  {cam['w']}x{cam['h']} @ {cam['fps']:.0f}fps  |  아무 키: 다음  q: 종료"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            cv2.imshow("Camera Finder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("\n[종료]")
                _print_summary(found)
                return

            elif key != 255:   # 아무 키나 누르면 다음 카메라
                break

        cap.release()
        cv2.destroyAllWindows()

    _print_summary(found)


def _print_summary(found: list):
    print("\n" + "="*45)
    print("  카메라 탐색 결과")
    print("="*45)
    for cam in found:
        print(f"  카메라 {cam['index']}번  {cam['w']}x{cam['h']} @ {cam['fps']:.0f}fps")
    print()
    print("  촬영 스크립트 실행 예시:")
    for cam in found:
        print(f"    python record_camera.py --camera {cam['index']}")
    print("="*45)


if __name__ == "__main__":
    find_cameras()