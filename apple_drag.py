"""
apple_drag.py - 자동 드래그 공통 도구 (apple_verify.py, apple_overlay.py --auto 가 함께 사용)

  setup()                 : DPI 인식 / pyautogui 안전장치 설정
  esc_pressed()           : ESC 키 감지 (Windows)
  drag(p0, p1, seconds)   : 누른 채 여러 지점을 거쳐 이동 → 먼 거리에서도 박스 유지
  to_screen(box, img)     : 캡처 이미지 좌표 → 화면 좌표 (HiDPI 배율 반영)
  box_margin(apples, px, py) : 사과 크기/간격으로 안전한 드래그 여백 계산
  member_box(apples, ...) : 지울 사과들을 완전히 감싸는 드래그 박스
  park()                  : 마우스를 게임 칸 밖(영역 모서리)으로 치워 화면 인식을 방해하지 않게 함
"""
import ctypes
import time

import numpy as np
import pyautogui

import sum10_pair_finder_v2 as v2

VK_ESCAPE = 0x1B


def setup():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)     # 오버레이와 같은 좌표계
    except Exception:
        pass
    pyautogui.FAILSAFE = True                              # 마우스를 화면 왼쪽 위 모서리로 밀면 즉시 중단
    pyautogui.PAUSE = 0


def esc_pressed():
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)
    except Exception:
        return False


def to_screen(box, img):
    kx = img.shape[1] / v2.REGION[2]
    ky = img.shape[0] / v2.REGION[3]
    x0, y0, x1, y1 = box
    return (v2.REGION[0] + x0 / kx, v2.REGION[1] + y0 / ky, v2.REGION[0] + x1 / kx, v2.REGION[1] + y1 / ky)


def drag(p0, p1, seconds=0.25):
    """누른 채 여러 지점을 거쳐 이동. 누르기 전/뗄 때 짧게 멈춰 게임이 입력을 놓치지(씹지) 않게 한다."""
    x0, y0, x1, y1 = p0[0], p0[1], p1[0], p1[1]
    pyautogui.moveTo(x0, y0)
    time.sleep(0.06)                                       # 커서가 자리 잡은 뒤 누름
    pyautogui.mouseDown()
    time.sleep(0.05)                                       # 눌림 인식 시간
    n = max(6, int(max(abs(x1 - x0), abs(y1 - y0)) / 20))
    for i in range(1, n + 1):
        pyautogui.moveTo(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
        if seconds > 0:
            time.sleep(seconds / n)
    time.sleep(0.08)                                       # 끝점에서 박스 확정
    pyautogui.mouseUp()


def park():
    pyautogui.moveTo(v2.REGION[0] + 3, v2.REGION[1] + 3)


START_FRAC = 0.45     # 시작점(왼쪽 위) 여백: 사과 사이 빈틈의 45%
END_FRAC = 0.80       # 끝점(오른쪽 아래) 여백: 사과 사이 빈틈의 80% (끝점이 흔들려도 사과를 놓치지 않게 더 넉넉히)


def box_margin(apples, pitch_x, pitch_y, frac=START_FRAC):
    """사과 사이 빈틈(칸 간격 - 사과 크기)의 frac 배, 최소 2px. 1.0 이면 이웃 사과 경계에 닿으므로 1.0 미만으로 쓴다."""
    aw = float(np.median([a.w for a in apples]))
    ah = float(np.median([a.h for a in apples]))
    mx = max(2.0, frac * (abs(pitch_x) - aw)) if pitch_x else 3.0
    my = max(2.0, frac * (abs(pitch_y) - ah)) if pitch_y else 3.0
    return mx, my


def member_box(members, start_m, end_m=None, extra=0.0):
    """지울 사과들을 감싸는 드래그 박스. start_m/end_m = (가로, 세로) 여백 (왼쪽 위 시작점 / 오른쪽 아래 끝점)."""
    end_m = end_m or start_m
    return (min(a.x for a in members) - start_m[0] - extra, min(a.y for a in members) - start_m[1] - extra,
            max(a.x + a.w for a in members) + end_m[0] + extra, max(a.y + a.h for a in members) + end_m[1] + extra)
