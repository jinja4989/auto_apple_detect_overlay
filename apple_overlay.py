"""
apple_overlay.py  -  게임 화면 위에 '지울 사과 묶음'을 박스로 안내하고, 판 상태(올클리어 가능 여부·예상 점수)를 알려 주는
                     투명 오버레이 (Windows 전용)

동작:
  1) 판을 처음 보면 올클리어 가능 여부와 전체 제거 계획을 '한 번만' 계산합니다 (정답 판은 보통 1~2초, 무작위 판은 최대 약 6초).
     화면 위쪽 배너에 판정과 예상 점수가 표시됩니다. 점수가 마음에 들지 않으면 게임을 다시 시작하세요.
       초록 = 정답 판(올클리어 가능)   주황 = 무작위 판 의심(올클리어 경로 못 찾음)   빨강 = 올클리어 불가(합계가 10의 배수 아님)
  2) 올클리어 가능한 판이면 정답 순서대로 박스를 '한 번에 1개씩'만 보여 줍니다. 지우면 다음 박스가 나옵니다.
  3) 올클리어가 아닌 판이면 지금 지울 수 있는 계획 단계를 번호 없이 박스로 모두 보여 줍니다 (아무 순서로 지워도 됩니다).
  4) 계산은 판당 한 번뿐이며, 계획과 다르게 지워도 자동으로 다시 계산하지 않습니다. 필요하면 F9로 다시 분석하세요.

실행:
  python apple_overlay.py                 # 기본 (계산 시간 6초)
  python apple_overlay.py --think 10      # 계산 시간(초). 무작위 판에서 점수를 더 높이고 싶을 때

키:  ESC = 종료   F9 = 새 판 시작/다시 분석 (칸 간격과 계획을 다시 기억)
"""

import argparse
import ctypes
import os
import sys
import threading
import time
import tkinter as tk

import cv2

import apple_detect as ad
import apple_solver as asv
import sum10_pair_finder_v2 as v2

KEY = "#010203"
COLORS = ["#00e676", "#ffea00", "#00b0ff", "#ff4081", "#ff9100", "#e040fb", "#1de9b6", "#ff5252"]
BANNER_BG = {"good": "#1b5e20", "warn": "#e65100", "bad": "#b71c1c", "info": "#37474f"}
VK_ESCAPE = 0x1B
VK_F9 = 0x78


REC_FPS = 10      # 녹화 영상 프레임 수


def _bgr(hex_color):
    h = hex_color.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _korean_font(size):
    """한글이 나오는 글꼴을 찾는다 (Windows: 맑은 고딕)."""
    from PIL import ImageFont
    for path in ("C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
                 "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                 "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate(img, apples, shown, banner):
    """캡처한 게임 화면(BGR)에 박스와 판정 배너를 직접 그려 넣는다 (녹화용: 오버레이 창은 캡처에서 숨겨지므로)."""
    from PIL import Image, ImageDraw
    out = img.copy()
    k = img.shape[1] / v2.REGION[2]                                   # 캡처 픽셀 / 논리 좌표 배율
    by = {(a.r, a.c): a for a in apples}
    for a in apples:
        if a.val == "?":
            cv2.rectangle(out, (a.x, a.y), (a.x + a.w, a.y + a.h), _bgr("#ff9100"), max(1, int(2 * k)))
    single = len(shown) == 1
    for i, group in enumerate(shown):
        m = [by[(r, c)] for r, c, _ in group if (r, c) in by]
        if not m:
            continue
        pad = int(4 * k)
        p0 = (int(min(a.x for a in m)) - pad, int(min(a.y for a in m)) - pad)
        p1 = (int(max(a.x + a.w for a in m)) + pad, int(max(a.y + a.h for a in m)) + pad)
        width = max(2, int((6 if single else 4) * k))
        cv2.rectangle(out, p0, p1, (0, 0, 0), width + int(3 * k))
        cv2.rectangle(out, p0, p1, _bgr(COLORS[0] if single else COLORS[i % len(COLORS)]), width)
    if banner[0]:
        bar_h = int(26 * k)
        cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), _bgr(BANNER_BG.get(banner[1], "#37474f")), -1)
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        font = _korean_font(max(12, int(15 * k)))
        ImageDraw.Draw(pil).text((int(8 * k), int(4 * k)), banner[0], font=font, fill=(255, 255, 255))
        out = cv2.cvtColor(__import__("numpy").array(pil), cv2.COLOR_RGB2BGR)
    return out


def record_loop(path, latest, stop):
    """별도 스레드: 게임 화면을 계속 캡처해 최신 박스/배너를 그려 넣고 mp4로 저장한다.
    캡처가 느려도 영상 길이가 실제 시간과 같도록 프레임을 채워 넣는다."""
    writer, t0, written = None, time.time(), 0
    while not stop.is_set():
        tc = time.time()
        img = v2.capture()
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, (w, h))
            if not writer.isOpened():
                print(f"[녹화 실패] {path} 를 열 수 없습니다.")
                return
        frame = annotate(img, *latest["v"])
        due = int((time.time() - t0) * REC_FPS) + 1
        for _ in range(max(1, due - written)):
            writer.write(frame)
            written += 1
        time.sleep(max(0.0, 1.0 / REC_FPS - (time.time() - tc)))
    if writer is not None:
        writer.release()
        print(f"[녹화 저장] {path} ({written / REC_FPS:.0f}초)")


def make_overlay():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    left, top, w, h = v2.REGION
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", KEY)
    root.configure(bg=KEY)
    root.geometry(f"{w}x{h}+{left}+{top}")
    canvas = tk.Canvas(root, width=w, height=h, bg=KEY, highlightthickness=0)
    canvas.pack()
    root.update_idletasks()
    root.update()

    user32 = ctypes.windll.user32
    hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
    style = user32.GetWindowLongW(hwnd, -20)
    user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20 | 0x80)   # layered | click-through | toolwindow
    excluded = bool(user32.SetWindowDisplayAffinity(hwnd, 0x11))      # 캡처에서 제외
    return root, canvas, excluded, user32


def draw(canvas, apples, shown, sx, sy, banner=("", "info")):
    """캡처 픽셀 좌표(apple.x 등)를 화면 논리 좌표로 나눠 그린다.
    박스가 1개뿐이면(올클리어 안내) 더 굵고 선명하게, 여러 개면 서로 다른 색으로 그린다. 번호는 붙이지 않는다."""
    canvas.delete("all")
    w = v2.REGION[2]
    by = {(a.r, a.c): a for a in apples}

    if banner[0]:                                                     # 위쪽 판정 배너 (항상 표시)
        canvas.create_rectangle(0, 0, w, 26, fill=BANNER_BG.get(banner[1], "#37474f"), outline="")
        canvas.create_text(8, 13, text=banner[0], anchor="w", fill="#ffffff", font=("Malgun Gothic", 10, "bold"))

    for a in apples:
        if a.val == "?":                                              # 인식 실패: 주황 점선
            canvas.create_rectangle(a.x / sx, a.y / sy, (a.x + a.w) / sx, (a.y + a.h) / sy,
                                    outline="#ff9100", width=2, dash=(3, 3))

    single = len(shown) == 1
    for i, group in enumerate(shown):                                 # group = 사과 2개 이상의 묶음
        members = [by[(r, c)] for r, c, _ in group]
        x0 = min(m.x for m in members) / sx - 4
        y0 = min(m.y for m in members) / sy - 4
        x1 = max(m.x + m.w for m in members) / sx + 4
        y1 = max(m.y + m.h for m in members) / sy + 4
        width = 6 if single else 4
        canvas.create_rectangle(x0, y0, x1, y1, outline="black", width=width + 3)
        canvas.create_rectangle(x0, y0, x1, y1, outline=COLORS[0] if single else COLORS[i % len(COLORS)], width=width)


def describe(shown, status="", banner=("", "info")):
    if banner[0]:
        print(f"\n[{banner[0]}]")
    if len(shown) == 1:                                               # 1수 안내: 위치까지 알려 줌
        print("  다음: " + " + ".join(str(v) for _, _, v in shown[0]) + "   @ " +
              ", ".join(f"({r + 1}행 {c + 1}열)" for r, c, _ in shown[0]))
    elif shown:
        print(f"  지울 수 있는 박스 {len(shown)}개 (아무 순서로 지워도 됩니다)")


def main():
    if os.name != "nt":
        sys.exit("이 오버레이는 Windows 전용입니다. 다른 OS에서는 apple_detect.py --check --solve 를 사용하세요.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--think", type=float, default=asv.PLAN_SECONDS,
                    help="분석 시간(초). 무작위 판에서 점수를 더 높이고 싶을 때 늘림 (기본 %(default)s)")
    ap.add_argument("--record", nargs="?", const="demo.mp4", default=None, metavar="파일",
                    help="박스와 배너를 게임 화면에 직접 그려 넣은 영상으로 저장 (기본 demo.mp4). 오버레이는 화면 녹화에 안 찍히므로 자랑용")
    a = ap.parse_args()

    lock = ad.GridLock()                                              # 처음 격자 눈금 기억
    planner = asv.Planner(a.think)
    print("[인식] 템플릿 매칭 사용" if ad.load_templates()
          else "[알림] apple_templates.npz 없음 → OCR 임시 추정(오인식 가능). python apple_detect.py --learn 권장")

    root, canvas, excluded, user32 = make_overlay()
    if not excluded:
        print("[알림] 캡처 제외 설정 실패 → 캡처 순간마다 오버레이를 잠깐 숨깁니다(깜빡임).")
    print("실행 중... 종료: ESC / 새 판 시작·다시 분석: F9")
    print("※ 사과가 가득 찬 처음 상태에서 실행하면 가장 정확합니다.")
    state = {"sig": None, "key": None}
    latest = {"v": ([], [], ("", "info"))}                            # 녹화 스레드가 읽는 최신 화면 상태
    stop = threading.Event()
    rec = None
    if a.record:
        rec = threading.Thread(target=record_loop, args=(a.record, latest, stop), daemon=True)
        rec.start()
        print(f"[녹화] {a.record} 로 저장합니다. 종료(ESC) 시 파일이 완성됩니다.")

    def tick():
        if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            root.destroy()
            return
        if user32.GetAsyncKeyState(VK_F9) & 0x8000:
            lock.reset()
            planner.reset()
            state["sig"] = state["key"] = None
            print("[새 판: 칸 간격과 계획을 다시 기억합니다]")

        if not excluded:
            root.withdraw()
            root.update()
            time.sleep(0.03)
        img = v2.capture()
        if not excluded:
            root.deiconify()
            root.attributes("-topmost", True)

        sx = img.shape[1] / v2.REGION[2]                              # 캡처 픽셀 / 논리 좌표 배율 (HiDPI 대응)
        sy = img.shape[0] / v2.REGION[3]
        apples, grid, groups = ad.analyze(img, lock)

        shown, status, kind = planner.update(grid)                    # 계산 없이 즉시: 계획의 다음 단계
        if kind == "fallback":                                        # 계획 단계가 더 없으면 보이는 쌍을 임시 표시
            shown = ad.pick_disjoint(groups)
        banner = planner.banner
        latest["v"] = (apples, list(shown), banner)

        sig = (tuple((p.r, p.c, p.val) for p in apples), tuple(shown), status, banner)
        if sig != state["sig"]:
            state["sig"] = sig
            draw(canvas, apples, shown, sx, sy, banner=banner)
            # 터미널은 판정이 바뀌거나 안내가 바뀔 때만 출력 (분석 중 초 단위 카운트다운은 제외)
            key = (kind, tuple(shown), banner[1], "" if kind == "computing" else banner[0])
            if key != state["key"]:
                state["key"] = key
                describe(shown, status, banner)
        root.after(int(v2.INTERVAL * 1000), tick)

    root.after(100, tick)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if rec is not None:
            rec.join(timeout=10)                                      # 영상 파일 마무리


if __name__ == "__main__":
    main()
