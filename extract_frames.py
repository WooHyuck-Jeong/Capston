"""
extract_frames.py
꼬깔 동영상을 프레임 단위로 분할해서 이미지 데이터셋을 생성하는 스크립트

사용법:
    python extract_frames.py --video red_cone.mp4
    python extract_frames.py --video blue_cone.mp4 --interval 5 --output dataset/blue
    python extract_frames.py --video red_cone.mp4 --video blue_cone.mp4  (다중 처리)
"""

import cv2
import argparse
import sys
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str | None = None,
    interval: int = 1,
    prefix: str | None = None,
    img_format: str = "jpg",
    quality: int = 95,
    resize: tuple[int, int] | None = None,
    max_frames: int | None = None,
) -> int:
    """
    동영상에서 프레임을 추출해 이미지로 저장

    Parameters
    ----------
    video_path  : 입력 동영상 파일 경로
    output_dir  : 이미지 저장 폴더 (None이면 동영상명_frames 폴더 자동 생성)
    interval    : N 프레임마다 1장 저장 (기본 1 = 전체 프레임)
    prefix      : 파일명 접두사 (None이면 동영상 파일명 사용)
    img_format  : 저장 형식 'jpg' 또는 'png'
    quality     : jpg 품질 (0~100)
    resize      : (width, height) 리사이즈, None이면 원본 크기 유지
    max_frames  : 최대 저장 프레임 수 (None이면 무제한)

    Returns
    -------
    저장된 이미지 수
    """

    video_p = Path(video_path)
    if not video_p.exists():
        print(f"[ERROR] 파일 없음: {video_path}")
        return 0

    # 출력 디렉토리 결정
    if output_dir is None:
        out_dir = video_p.parent / f"{video_p.stem}_frames"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 접두사
    file_prefix = prefix if prefix else video_p.stem

    # 이미지 저장 파라미터
    if img_format.lower() == "jpg":
        ext = ".jpg"
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        ext = ".png"
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    # ── 동영상 열기 ───────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        print(f"[ERROR] 동영상 파일을 열 수 없습니다: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"\n{'='*55}")
    print(f"  입력  : {video_path}")
    print(f"  해상도: {orig_w} x {orig_h}  |  FPS: {fps:.1f}  |  총 {total_frames} 프레임 ({duration_sec:.1f}초)")
    print(f"  출력  : {out_dir}/")
    print(f"  추출  : {interval} 프레임마다 1장  |  형식: {ext}  |  품질: {quality}")
    if resize:
        print(f"  리사이즈: {orig_w}x{orig_h} → {resize[0]}x{resize[1]}")
    if max_frames:
        print(f"  최대 저장: {max_frames} 장")
    print(f"{'='*55}\n")

    # ── 프레임 추출 ───────────────────────────────────────────────
    frame_idx   = 0   # 동영상 내 현재 프레임 번호
    saved_count = 0   # 실제 저장된 이미지 수

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # interval 마다 저장
        if frame_idx % interval == 0:
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

            filename = out_dir / f"{file_prefix}_{saved_count:06d}{ext}"
            cv2.imwrite(str(filename), frame, encode_params)
            saved_count += 1

            # 진행 상황 출력 (100장마다)
            if saved_count % 100 == 0:
                progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
                print(f"  [{progress:5.1f}%] {saved_count} 장 저장 완료...")

            # 최대 프레임 수 도달 시 중단
            if max_frames and saved_count >= max_frames:
                print(f"  최대 저장 수({max_frames}) 도달 → 중단")
                break

        frame_idx += 1

    cap.release()
    print(f"\n[완료] {saved_count} 장 저장 → {out_dir}/\n")
    return saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="꼬깔 동영상에서 프레임 이미지 추출",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video",    "-v", nargs="+", required=True,
                        help="입력 동영상 파일 (여러 개 지정 가능)")
    parser.add_argument("--output",   "-o", default=None,
                        help="이미지 저장 폴더 (미지정 시 동영상명_frames 폴더 자동 생성)")
    parser.add_argument("--interval", "-i", type=int, default=1,
                        help="N 프레임마다 1장 추출 (예: 5 → 5프레임당 1장)")
    parser.add_argument("--prefix",   "-p", default=None,
                        help="저장 파일명 접두사 (미지정 시 동영상 파일명 사용)")
    parser.add_argument("--format",   "-F", choices=["jpg", "png"], default="jpg",
                        help="저장 이미지 형식")
    parser.add_argument("--quality",  "-q", type=int, default=95,
                        help="JPG 저장 품질 (0~100)")
    parser.add_argument("--resize",   "-r", nargs=2, type=int, metavar=("W", "H"),
                        default=None,
                        help="리사이즈 (예: --resize 640 640)")
    parser.add_argument("--max-frames", "-m", type=int, default=None,
                        help="최대 저장 프레임 수")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    resize = tuple(args.resize) if args.resize else None
    total_saved = 0

    for video_file in args.video:
        count = extract_frames(
            video_path=video_file,
            output_dir=args.output,
            interval=args.interval,
            prefix=args.prefix,
            img_format=args.format,
            quality=args.quality,
            resize=resize,
            max_frames=args.max_frames,
        )
        total_saved += count

    if len(args.video) > 1:
        print(f"[전체 완료] 총 {total_saved} 장의 이미지를 저장했습니다.")