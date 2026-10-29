"""
sum10_pair_finder_v2.py  -  합이 10인 인접 쌍 탐색기 (색상 기반 숫자 추출 + 실시간 뷰어)

변경점 (v1 대비)
  1) 이진화 방식 교체: 셀 중앙부의 '대표색(중앙값)'과 크게 다른 픽셀 = 글자 로 판단
     → 사과/배경/글자 색이 어떻든 동작하고, 글자 영역만 타이트하게 잘라 OCR 정확도 상승
  2) 빈 칸('.')과 인식 실패('?')를 구분해서 표시
  3) --live : 인식 결과(숫자)와 쌍 하이라이트를 창에 계속 갱신 (제거 후 생긴 빈칸도 자동 반영)
  4) --debug: 셀별 크롭/마스크 이미지를 debug_cells 폴더에 저장 (인식 실패 원인 확인용)

사용:
  python sum10_pair_finder_v2.py --once --debug
  python sum10_pair_finder_v2.py --live
  python sum10_pair_finder_v2.py --live --click
  (live 창은 게임 그리드 영역과 겹치지 않는 곳에 두세요. 'q' 키로 종료)
"""

import argparse
import hashlib
import os
import time

import cv2
import numpy as np
import pyautogui
#import pytesseract

# ───────────────────────── 설정 ─────────────────────────
#_TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#if os.path.exists(_TESS):
#    pytesseract.pytesseract.tesseract_cmd = _TESS

REGION = (1137, 551, 720, 427)  # (left, top, width, height)  ← 기존 값으로 교체
ROWS, COLS = 10, 17             # ← 실제 그리드 크기
TARGET_SUM = 10
INTERVAL = 0.5                  # 갱신 주기(초)
CLICK_PAUSE = 0.15

CROP_LO, CROP_HI = 0.22, 0.78   # 셀 중앙 22%~78% 영역만 사용 (테두리/사과 바깥 배경 제외)
DIFF_THRESH = 60                # 대표색과 이 값 이상 차이나면 글자 픽셀
EMPTY_RATIO = 0.02              # 글자 픽셀 비율이 이보다 낮으면 빈 칸

pyautogui.FAILSAFE = True
_cache = {}


# ───────────────────── 1단계: 캡처 + 숫자 인식 ─────────────────────
def capture(region=REGION):
    pil = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def extract_digit_mask(cell_bgr):
    """셀에서 글자 픽셀 마스크(0/255)를 추출. 빈 칸이면 None."""
    h, w = cell_bgr.shape[:2]
    crop = cell_bgr[int(h * CROP_LO):int(h * CROP_HI), int(w * CROP_LO):int(w * CROP_HI)]
    px = crop.reshape(-1, 3).astype(np.float32)
    ref = np.median(px, axis=0)                       # 사과(또는 배경)의 대표색
    dist = np.linalg.norm(crop.astype(np.float32) - ref, axis=2)
    mask = (dist > DIFF_THRESH).astype(np.uint8) * 255

    if mask.mean() / 255 < EMPTY_RATIO:
        return None                                   # 글자 없음 = 빈 칸

    # 작은 잡음 제거: 가장 큰 덩어리의 25% 미만인 조각은 버림
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= max(10, 0.25 * areas.max())]
    clean = np.isin(labels, keep).astype(np.uint8) * 255
    return clean


def read_cell(cell_bgr, dbg_name=None):
    """반환: 1~9 정수 / None(빈 칸) / '?'(글자는 있는데 인식 실패)"""
    key = hashlib.md5(cell_bgr.tobytes()).hexdigest()
    if key in _cache and dbg_name is None:
        return _cache[key]

    mask = extract_digit_mask(cell_bgr)
    if mask is None:
        result = None
    else:
        ys, xs = np.where(mask > 0)
        tight = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        img = 255 - tight                              # 흰 배경 + 검은 글자
        scale = 48.0 / img.shape[0]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = cv2.copyMakeBorder(img, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)

        text = pytesseract.image_to_string(
            img, config="--psm 10 -c tessedit_char_whitelist=123456789").strip()
        result = int(text[0]) if text and text[0] in "123456789" else "?"

        if dbg_name:
            os.makedirs("debug_cells", exist_ok=True)
            cv2.imwrite(f"debug_cells/{dbg_name}_{result}_mask.png", img)

    if dbg_name:
        os.makedirs("debug_cells", exist_ok=True)
        cv2.imwrite(f"debug_cells/{dbg_name}_{result}_cell.png", cell_bgr)
    _cache[key] = result
    return result


def image_to_grid(img, debug=False):
    h, w = img.shape[:2]
    ch, cw = h / ROWS, w / COLS
    grid = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            cell = img[int(r * ch):int((r + 1) * ch), int(c * cw):int((c + 1) * cw)]
            row.append(read_cell(cell, f"r{r:02d}c{c:02d}" if debug else None))
        grid.append(row)
    return grid


# ───────────────────── 2단계: 합이 10인 인접 쌍 ─────────────────────
def find_pairs(grid, target=TARGET_SUM):
    pairs = []
    for r in range(ROWS):
        for c in range(COLS):
            a = grid[r][c]
            if not isinstance(a, int):
                continue
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr < ROWS and nc < COLS:
                    b = grid[nr][nc]
                    if isinstance(b, int) and a + b == target:
                        pairs.append(((r, c, a), (nr, nc, b)))
    return pairs


def pick_disjoint(pairs):
    used, chosen = set(), []
    for p in pairs:
        (r1, c1, _), (r2, c2, _) = p
        if (r1, c1) not in used and (r2, c2) not in used:
            chosen.append(p)
            used.update([(r1, c1), (r2, c2)])
    return chosen


# ───────────────────── 3단계: 출력 / 시각화 / 클릭 ─────────────────────
def render(img, grid, pairs):
    """인식된 숫자를 셀 위에 그리고, 합이 10인 쌍은 초록 박스로 표시."""
    out = img.copy()
    h, w = out.shape[:2]
    ch, cw = h / ROWS, w / COLS
    for (r1, c1, _), (r2, c2, _) in pairs:
        for r, c in ((r1, c1), (r2, c2)):
            cv2.rectangle(out, (int(c * cw) + 1, int(r * ch) + 1),
                          (int((c + 1) * cw) - 1, int((r + 1) * ch) - 1), (0, 255, 0), 3)
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r][c]
            if v is None:
                continue
            color = (0, 140, 255) if v == "?" else (255, 0, 0)   # ?는 주황, 숫자는 파랑
            cv2.putText(out, str(v), (int((c + 0.1) * cw), int((r + 0.95) * ch)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def report(grid, pairs):
    print("\n=== 인식된 그리드 ( . = 빈 칸 / ? = 인식 실패 ) ===")
    for row in grid:
        print(" ".join("." if v is None else str(v) for v in row))
    unknown = sum(v == "?" for row in grid for v in row)
    print(f"\n인식 실패 칸: {unknown}개 | 합이 {TARGET_SUM}인 쌍: {len(pairs)}개")
    for (r1, c1, v1), (r2, c2, v2) in pairs:
        print(f"  ({r1},{c1})={v1} + ({r2},{c2})={v2}")


def cell_center_screen(r, c):
    left, top, w, h = REGION
    return left + (c + 0.5) * w / COLS, top + (r + 0.5) * h / ROWS


def click_pairs(pairs):
    for (r1, c1, _), (r2, c2, _) in pairs:
        for r, c in ((r1, c1), (r2, c2)):
            pyautogui.click(*cell_center_screen(r, c))
            time.sleep(CLICK_PAUSE)


def run_once(debug=False, do_click=False):
    img = capture()
    grid = image_to_grid(img, debug)
    pairs = find_pairs(grid)
    report(grid, pairs)
    cv2.imwrite("overlay.png", render(img, grid, pairs))
    print("저장: overlay.png" + (" / debug_cells/ 폴더" if debug else ""))
    if do_click and pairs:
        click_pairs(pick_disjoint(pairs))


def run_live(do_click=False):
    win = "sum10 live (q = quit)"
    cv2.namedWindow(win)
    cv2.moveWindow(win, 20, 20)          # 게임 영역과 겹치면 캡처에 찍히므로 위치 조정 필요
    while True:
        img = capture()
        grid = image_to_grid(img)        # 캐시 덕분에 바뀐 셀만 다시 OCR
        pairs = find_pairs(grid)
        cv2.imshow(win, render(img, grid, pairs))
        if do_click and pairs:
            click_pairs(pick_disjoint(pairs))
        if cv2.waitKey(int(INTERVAL * 1000)) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def calibrate():
    input("그리드 '좌상단'에 마우스를 올리고 Enter...")
    x1, y1 = pyautogui.position()
    input("그리드 '우하단'에 마우스를 올리고 Enter...")
    x2, y2 = pyautogui.position()
    print(f"REGION = ({x1}, {y1}, {x2 - x1}, {y2 - y1})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--click", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    elif a.live:
        run_live(a.click)
    else:
        run_once(a.debug, a.click)
