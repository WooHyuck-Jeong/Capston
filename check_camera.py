import cv2
import argparse
import sys
from datetime import datetime


def list_cameras(max_index: int = 5) -> list[int]:
    """사용 가능한 카메라 인덱스 목록 반환"""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

a = list_cameras()
print(a)