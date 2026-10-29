"""
apple_detect.py  -  사과 기반 인식기: 격자 분할 없이 '붉은 사과'를 찾아 안의 흰 숫자를 읽는다.

기존 방식의 문제: 화면을 ROWS x COLS 로 균등 분할하면 REGION/칸 크기가 조금만 어긋나도
한 칸에 사과 조각 두 개가 걸리거나 배경만 잡힙니다. 이 모듈은
  1) 붉은 덩어리(사과)를 찾고            → 위치/크기는 사과 자신이 알려줌 (정렬 작업 불필요)
  2) 사과 중앙의 흰 픽셀만 숫자로 추출      → 사과 하이라이트(빛 반사)는 테두리 접촉 여부로 제외
  3) 숫자 모양을 템플릿과 비교해 인식       → 학습한 모양이 없으면 OCR로 임시 추정
  4) 사과들의 중심 좌표를 군집화해 격자 구성 → 제거되어 빈 칸이 된 자리는 자동으로 None

REGION 은 '그리드 전체를 넉넉히 포함'하기만 하면 됩니다 (정확히 맞출 필요 없음).
단, 가장자리 사과가 잘리면 인식되지 않으니 약간 여유 있게 잡으세요.

사용 (sum10_pair_finder_v2.py 와 같은 폴더, 영문/한글 입력 상태 무관 - 숫자/스페이스/Esc/Enter 키만 사용):
  python apple_detect.py --check     # 사과 탐지·인식 결과 확인 (apple_check.png 저장)
  python apple_detect.py --learn     # 숫자 모양 학습 (apple_templates.npz 저장)
"""

import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import sum10_pair_finder_v2 as v2     # REGION, capture, pick_disjoint 재사용

TPL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apple_templates.npz")

# ───────────── 색상 기준 (OpenCV HSV: H 0~179, S/V 0~255) ─────────────
RED_H, RED_S, RED_V = 12, 110, 110      # 붉은 사과: 색상 0±12, 채도/명도 일정 이상
WHITE_S, WHITE_V = 90, 160              # 흰 숫자: 채도 낮고 밝음
SIZE = 40                               # 정규화된 글자 크기
ACCEPT, MARGIN, CLUSTER_SIM = 0.80, 0.03, 0.90

_tpl_x, _tpl_y = [], []
_cache = {}


@dataclass
class Apple:
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float
    r: int = 0
    c: int = 0
    val: object = None                  # 1~9 / '?'
    glyph: object = field(default=None, repr=False)
    mask: object = field(default=None, repr=False)


# ───────────────────── 사과 찾기 ─────────────────────
def find_apples(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = ((h <= RED_H) | (h >= 180 - RED_H)) & (s >= RED_S) & (v >= RED_V)
    red = cv2.morphologyEx(red.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand = []
    for c in cnts:
        area = cv2.contourArea(c)                         # 외곽선 면적 = 안의 숫자 구멍 포함
        x, y, w, hh = cv2.boundingRect(c)
        if area > 0 and 0.7 <= w / hh <= 1.4:             # 사과는 대략 정사각형 (긴 막대 UI 제외)
            cand.append((area, (x, y, w, hh)))
    if not cand:
        return []
    amax = max(a for a, _ in cand)
    med = float(np.median([a for a, _ in cand if a >= 0.3 * amax]))
    return [Apple(x, y, w, hh, x + w / 2, y + hh / 2)
            for a, (x, y, w, hh) in cand if 0.55 * med <= a <= 1.8 * med]


# ───────────────────── 숫자 추출 / 인식 ─────────────────────
def _window(img, a):
    """사과 안에서 숫자가 있을 중앙 영역만 잘라낸다 (위쪽 잎/좌상단 하이라이트 회피)."""
    H, W = img.shape[:2]
    x0, x1 = max(0, a.x + int(0.20 * a.w)), min(W, a.x + int(0.80 * a.w))
    y0, y1 = max(0, a.y + int(0.28 * a.h)), min(H, a.y + int(0.92 * a.h))
    return img[y0:y1, x0:x1]


def _glyph_mask(crop, apple_area):
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] <= WHITE_S) & (hsv[..., 2] >= WHITE_V)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n <= 1:
        return None
    ch, cw = white.shape
    min_area = max(6, int(0.008 * apple_area))
    comps = []
    for i in range(1, n):
        sx, sy, sw, sh, area = stats[i]
        if area < min_area:
            continue
        touches = sx == 0 or sy == 0 or sx + sw >= cw or sy + sh >= ch   # 창 가장자리에 닿음 = 하이라이트
        comps.append((i, area, touches))
    if not comps:
        return None
    inner = [c for c in comps if not c[2]] or comps
    amax = max(c[1] for c in inner)
    keep = [c[0] for c in inner if c[1] >= 0.2 * amax]
    return np.isin(labels, keep).astype(np.uint8) * 255


def normalize(mask):
    ys, xs = np.where(mask > 0)
    tight = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255.0
    h, w = tight.shape
    s = (SIZE - 4) / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    resized = cv2.resize(tight, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIZE, SIZE), np.float32)
    y0, x0 = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def score(a, b):
    if a.std() < 1e-6 or b.std() < 1e-6:
        return -1.0
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def ranked_scores(glyph):
    best = {}
    for x, y in zip(_tpl_x, _tpl_y):
        s = score(glyph, x)
        if s > best.get(y, -2.0):
            best[y] = s
    return sorted(best.items(), key=lambda kv: -kv[1])


def classify(glyph):
    ranked = ranked_scores(glyph)
    if not ranked or ranked[0][1] < ACCEPT:
        return "?"
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < MARGIN:
        return "?"
    return ranked[0][0]


def ocr_guess(mask):
    """템플릿이 없을 때의 임시 추정 / 학습 창의 추천값."""
    try:
        ys, xs = np.where(mask > 0)
        tight = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        img = 255 - tight
        img = cv2.resize(img, None, fx=48.0 / img.shape[0], fy=48.0 / img.shape[0],
                         interpolation=cv2.INTER_CUBIC)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = cv2.copyMakeBorder(img, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
        t = v2.pytesseract.image_to_string(
            img, config="--psm 10 -c tessedit_char_whitelist=123456789").strip()
        return int(t[0]) if t and t[0] in "123456789" else "?"
    except Exception:
        return "?"


def recognize(img, a):
    crop = _window(img, a)
    key = hashlib.md5(crop.tobytes()).hexdigest() + str(crop.shape)
    if key in _cache:
        a.val, a.glyph, a.mask = _cache[key]
        return
    mask = _glyph_mask(crop, a.w * a.h)
    if mask is None:
        val, glyph = "?", None
    else:
        glyph = normalize(mask)
        val = classify(glyph) if _tpl_x else ocr_guess(mask)
    a.val, a.glyph, a.mask = val, glyph, mask
    _cache[key] = (val, glyph, mask)


# ───────────────────── 격자 구성 / 쌍 탐색 ─────────────────────
def estimate_axis(vals, tol):
    """좌표들을 군집화해 (칸 간격 pitch, 첫 칸 중심 origin)을 추정. 군집이 1개면 pitch=None."""
    sv = np.sort(np.asarray(vals, float))
    groups = [[sv[0]]]
    for v in sv[1:]:
        if v - np.mean(groups[-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    centers = np.array([np.mean(g) for g in groups])
    if len(centers) < 2:
        return None, float(centers[0])
    gaps = np.diff(centers)
    single = gaps[gaps <= 1.5 * gaps.min()]          # 한 칸 간격만 모아 중앙값 → pitch
    return float(np.median(single)), float(centers[0])


class GridLock:
    """사과가 가득 찬 처음 화면에서 칸 간격/기준점을 기억해 둔다.
    사과가 제거되어 듬성듬성해져도 같은 눈금으로 위치를 계산하므로, 빈자리가 정확히 유지된다.
    (사과 수가 크게 늘면 새 판으로 보고 다시 기억한다.)"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = self.y = None
        self.n = 0

    def update(self, est_x, est_y, n):
        if est_x[0] is None or est_y[0] is None:
            if self.x is None:                        # 처음부터 사과가 너무 적으면 임시값만 사용
                self.x, self.y = est_x, est_y
            return
        if self.n == 0 or n > 1.15 * self.n + 3:
            self.x, self.y, self.n = est_x, est_y, n


def _idx(v, pitch, origin):
    return 0 if pitch is None else int(round((v - origin) / pitch))


def build_grid(apples, lock=None):
    if not apples:
        return [], None, None
    med_w = float(np.median([a.w for a in apples]))
    med_h = float(np.median([a.h for a in apples]))
    est_x = estimate_axis([a.cx for a in apples], 0.35 * med_w)
    est_y = estimate_axis([a.cy for a in apples], 0.35 * med_h)
    if lock is not None:
        lock.update(est_x, est_y, len(apples))
        (px, ox), (py, oy) = lock.x, lock.y
    else:
        (px, ox), (py, oy) = est_x, est_y

    ci = [_idx(a.cx, px, ox) for a in apples]
    ri = [_idx(a.cy, py, oy) for a in apples]
    cmin, rmin = min(ci), min(ri)                    # 기준점보다 왼/위쪽이 생기면 음수 방지를 위해 평행이동
    for a, r, c in zip(apples, ri, ci):
        a.r, a.c = r - min(rmin, 0), c - min(cmin, 0)
    rows, cols = max(a.r for a in apples) + 1, max(a.c for a in apples) + 1
    grid = [[None] * cols for _ in range(rows)]
    for a in apples:
        grid[a.r][a.c] = a.val
    return grid, px, py


def find_rects(grid, target=10):
    """사과 합이 10인 '최소 사각형'(테두리 네 변에 각각 사과가 있는 것)을 모두 찾는다. 2개 이상, '?' 포함 영역 제외."""
    R = len(grid)
    C = len(grid[0]) if R else 0
    if not R or not C:
        return []
    val = np.array([[g if isinstance(g, int) else 0 for g in row] for row in grid])
    has = np.array([[1 if isinstance(g, int) else 0 for g in row] for row in grid])
    unk = np.array([[1 if g == "?" else 0 for g in row] for row in grid])

    def pref(m):
        return np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    S, H, U = pref(val), pref(has), pref(unk)

    def rect(P, r1, c1, r2, c2):
        return P[r2 + 1, c2 + 1] - P[r1, c2 + 1] - P[r2 + 1, c1] + P[r1, c1]

    out = []
    for r1 in range(R):
        for r2 in range(r1, R):
            for c1 in range(C):
                for c2 in range(c1, C):
                    if rect(S, r1, c1, r2, c2) != target or rect(H, r1, c1, r2, c2) < 2:
                        continue
                    if rect(U, r1, c1, r2, c2) > 0:
                        continue
                    if (rect(H, r1, c1, r1, c2) == 0 or rect(H, r2, c1, r2, c2) == 0 or
                            rect(H, r1, c1, r2, c1) == 0 or rect(H, r1, c2, r2, c2) == 0):
                        continue
                    out.append(tuple((r, c, grid[r][c]) for r in range(r1, r2 + 1)
                                     for c in range(c1, c2 + 1) if isinstance(grid[r][c], int)))
    return out


def pick_disjoint(groups):
    """사과를 두 번 쓰지 않도록 겹치지 않는 묶음만 선택 (사과 수가 적은 묶음 우선)."""
    used, chosen = set(), []
    for g in sorted(groups, key=len):
        cells = {(r, c) for r, c, _ in g}
        if cells & used:
            continue
        chosen.append(g)
        used |= cells
    return chosen


def analyze(img, lock=None):
    """이미지 → (apples, grid, groups)"""
    apples = find_apples(img)
    for a in apples:
        recognize(img, a)
    grid, _, _ = build_grid(apples, lock)
    return apples, grid, find_rects(grid)


# ───────────────────── 게임 영역 자동 설정 ─────────────────────
def auto_region(shot=None):
    """전체 화면에서 사과 덩어리를 찾아 게임 영역 (left, top, width, height) 을 자동으로 정한다. 못 찾으면 None.
    사과가 가득 찬 처음 화면에서 실행해야 격자 전체가 들어온다."""
    if shot is None:
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        shot = cv2.cvtColor(np.array(v2.pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    apples = find_apples(shot)
    if len(apples) < 10:
        return None
    w = float(np.median([a.w for a in apples]))
    pts = np.array([(a.cx, a.cy) for a in apples])
    parent = list(range(len(apples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(apples)):                        # 가까운 사과끼리 묶어 가장 큰 덩어리 = 게임판
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        for j in np.nonzero(d <= 2.2 * w)[0]:
            parent[find(int(j))] = find(i)
    roots = {}
    for i in range(len(apples)):
        roots.setdefault(find(i), []).append(i)
    comp = max(roots.values(), key=len)
    if len(comp) < 10:
        return None
    grp = [apples[i] for i in comp]
    pad = int(0.9 * w)
    H, W = shot.shape[:2]
    left = max(0, int(min(a.x for a in grp)) - pad)
    top = max(0, int(min(a.y for a in grp)) - pad - 26)   # 위쪽은 판정 배너 자리
    right = min(W, int(max(a.x + a.w for a in grp)) + pad)
    bottom = min(H, int(max(a.y + a.h for a in grp)) + pad)
    return (left, top, right - left, bottom - top)


def use_region(region):
    """게임 영역을 적용한다 (v2.REGION 과 v2.capture 를 바꿔 끼움 → sum10_pair_finder_v2.py 파일은 수정하지 않는다)."""
    v2.REGION = tuple(int(v) for v in region)

    def _capture(region=None):
        return cv2.cvtColor(np.array(v2.pyautogui.screenshot(region=v2.REGION)), cv2.COLOR_RGB2BGR)

    v2.capture = _capture


def ensure_region(auto=True):
    """자동 인식 → 성공하면 적용, 실패하면 sum10_pair_finder_v2.py 의 REGION(수동 값)을 그대로 사용."""
    if auto:
        r = auto_region()
        if r:
            use_region(r)
            print(f"[영역] 게임판을 자동으로 찾았습니다: REGION = {v2.REGION}")
            return True
        print(f"[영역] 화면에서 사과판을 못 찾아 기존 REGION {v2.REGION} 을 사용합니다. (사과가 가득 찬 상태인지 확인)")
    return False


# ───────────────────── 템플릿 저장/불러오기 ─────────────────────
def load_templates():
    global _tpl_x, _tpl_y
    _tpl_x, _tpl_y = [], []
    _cache.clear()
    if os.path.exists(TPL_FILE):
        d = np.load(TPL_FILE)
        _tpl_x = [x.astype(np.float32) for x in d["x"]]
        _tpl_y = [int(v) for v in d["y"]]
    return bool(_tpl_x)


def save_templates():
    np.savez(TPL_FILE, x=np.array(_tpl_x, np.float32), y=np.array(_tpl_y, np.int32))


# ───────────────────── 학습 / 확인 도구 ─────────────────────
def learn(auto=True):
    load_templates()
    print("3초 후 화면을 캡처합니다. 게임 화면이 터미널에 가려지지 않게 해 주세요...")
    time.sleep(3)
    ensure_region(auto)
    img = v2.capture()
    apples = find_apples(img)
    if not apples:
        print("사과를 찾지 못했습니다. REGION 이 그리드를 포함하는지, 색 기준(RED_*)을 확인하세요.")
        return
    for a in apples:
        recognize(img, a)

    clusters = []
    for a in apples:
        if a.glyph is None:
            continue
        ranked = ranked_scores(a.glyph)
        if ranked and ranked[0][1] >= CLUSTER_SIM:
            continue                                             # 이미 아는 모양
        for cl in clusters:
            if score(a.glyph, cl["rep"]) >= CLUSTER_SIM:
                cl["n"] += 1
                break
        else:
            clusters.append({"rep": a.glyph, "mask": a.mask, "apple": a, "n": 1})

    if not clusters:
        print(f"사과 {len(apples)}개 발견, 새로 배울 숫자 모양이 없습니다. (--check 로 확인해 보세요)")
        return

    clusters.sort(key=lambda cl: -cl["n"])
    win = "learn: 1-9 / Space=skip / Esc=quit"
    added = 0
    for i, cl in enumerate(clusters):
        a = cl["apple"]
        pad = 6
        crop = img[max(0, a.y - pad):a.y + a.h + pad, max(0, a.x - pad):a.x + a.w + pad]
        f = max(1, 220 // max(1, crop.shape[0]))
        big = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        glyph = cv2.resize((cl["rep"] * 255).astype(np.uint8), (220, 220), interpolation=cv2.INTER_NEAREST)
        glyph = cv2.cvtColor(255 - glyph, cv2.COLOR_GRAY2BGR)
        body_h = max(big.shape[0], 220)
        canvas = np.full((body_h + 96, max(big.shape[1] + 240, 640), 3), 255, np.uint8)
        canvas[:big.shape[0], :big.shape[1]] = big
        canvas[:220, big.shape[1] + 20:big.shape[1] + 240] = glyph
        guess = ocr_guess(cl["mask"])
        cv2.putText(canvas, f"[{i + 1}/{len(clusters)}] seen x{cl['n']}   which digit? press 1-9",
                    (10, body_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(canvas, f"OCR guess: {guess}" + ("   (Enter = accept)" if isinstance(guess, int) else ""),
                    (10, body_h + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 1)
        cv2.putText(canvas, "Space = skip     Esc = finish     (left: apple / right: shape to learn)",
                    (10, body_h + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        cv2.imshow(win, canvas)

        label, finish = None, False
        while True:                                              # 유효한 키가 올 때까지 대기 (엉뚱한 키는 무시)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                finish = True
                break
            if k == 32:
                break
            if ord("1") <= k <= ord("9"):
                label = k - ord("0")
                break
            if k in (13, 10) and isinstance(guess, int):
                label = guess
                break
        if finish:
            break
        if label:
            _tpl_x.append(cl["rep"])
            _tpl_y.append(label)
            added += 1
            print(f"  모양 {i + 1} → {label} 저장")
    cv2.destroyAllWindows()

    if added:
        save_templates()
        print(f"\n템플릿 {added}개 추가 (총 {len(_tpl_x)}개) → {TPL_FILE}")
        print("--learn 을 한 번 더 실행해 남은 새 모양이 있는지 확인하세요.")
    else:
        print("추가된 템플릿 없음")


def check(solve=False, auto=True):
    has = load_templates()
    ensure_region(auto)
    img = v2.capture()
    apples, grid, groups = analyze(img, GridLock())
    _, px, py = build_grid(apples)

    print(f"인식 방식: {'템플릿 매칭' if has else 'OCR 임시 추정 (--learn 으로 학습 권장)'}")
    print(f"사과 {len(apples)}개 발견 | 격자 {len(grid)}행 x {len(grid[0]) if grid else 0}열 | "
          f"칸 간격 ≈ {px and round(px, 1)} x {py and round(py, 1)} px")
    print("\n( . = 빈 자리 / ? = 인식 실패 )")
    for row in grid:
        print(" ".join("." if v is None else str(v) for v in row))
    unknown = sum(1 for a in apples if a.val == "?")
    tot = sum(v for row in grid for v in row if isinstance(v, int))
    print(f"\n사과 숫자 전체 합계: {tot} → " +
          ("10의 배수입니다 (올클리어가 가능할 수 있음)" if tot % 10 == 0
           else "10의 배수가 아닙니다 (이 판은 올클리어가 불가능)") +
          (f"  ※ 인식 실패 {unknown}칸이 있어 합계가 부정확할 수 있음" if unknown else ""))
    print(f"인식 실패 {unknown}개 | 지금 바로 지울 수 있는 묶음 {len(groups)}개")

    if solve and grid:
        import apple_solver as asv
        print(f"\n[분석] 올클리어 가능 여부와 최적 제거 계획을 계산 중... (최대 {asv.PLAN_SECONDS:.0f}초)")
        info = asv.analyze_board(grid, asv.PLAN_SECONDS)
        text, _ = asv.verdict(info)
        print(f"판정: {text}")
        norm = asv.Planner.norm(grid)
        # 비교: 순서 없이 첫 번째 가능한 묶음을 계속 지웠을 때 (보이는 대로 지우기)
        naive = asv.Solver(len(grid), len(grid[0]))
        G = np.array(norm, np.int16)[None]
        got = 0
        while True:
            si, mi = naive.moves(G)
            if len(mi) == 0:
                break
            got += int((G[0, naive.r1[mi[0]]:naive.r2[mi[0]] + 1, naive.c1[mi[0]]:naive.c2[mi[0]] + 1] > 0).sum())
            G = naive.apply(G, si[:1], mi[:1])
        print(f"보이는 대로 지우면 {got}/{info['total']}점 → 계획대로 지우면 {info['removed']}/{info['total']}점 ({len(info['path'])}단계)")
        for i, grp in enumerate(asv.moves_to_groups(norm, info["path"]), 1):
            print(f"  {i:>2}. " + " + ".join(str(v) for _, _, v in grp) + "   @ " +
                  ", ".join(f"({r + 1}행 {c + 1}열)" for r, c, _ in grp))

    if len(apples) < 10:
        print("※ 사과를 거의 못 찾았습니다. REGION 이 그리드를 포함하는지, 또는 RED_S/RED_V 값을 낮춰 보세요.")

    out = img.copy()
    for a in apples:
        color = (0, 140, 255) if a.val == "?" else (0, 220, 0)
        cv2.rectangle(out, (a.x, a.y), (a.x + a.w, a.y + a.h), color, 2)
        cv2.putText(out, str(a.val), (a.x + 2, a.y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imwrite("apple_check.png", out)
    print("저장: apple_check.png (초록=인식됨, 주황=실패, 노랑 글자=읽은 값)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--solve", action="store_true", help="올클리어 가능 여부 판정과 전체 제거 계획도 계산해 출력")
    ap.add_argument("--no-auto-region", action="store_true", help="게임 영역 자동 인식을 끄고 sum10_pair_finder_v2.py 의 REGION 사용")
    a = ap.parse_args()
    if a.learn:
        learn(not a.no_auto_region)
    else:
        check(a.solve, not a.no_auto_region)
