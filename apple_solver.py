"""
apple_solver.py  -  제거 순서 계획기 (심층 분석)

문제: 합이 10인 쌍을 보이는 대로 제거하면, 다른 쌍에 필요한 사과를 먼저 없애 버려
      사과가 많이 남은 채 더 이상 제거할 쌍이 없어진다.
해법: 현재 판에서 가능한 모든 제거를 '미리 시뮬레이션'해 가며, 끝까지 가장 많이 지울 수 있는
      제거 순서를 빔 서치(beam search)로 찾는다.

  - 상태 평가 = (지금까지 제거한 사과 수) + w × (아직 남아 있는 제거 가능 묶음 수)
    → 많이 지우되, 앞으로 쓸 수 있는 선택지를 많이 남기는 순서를 우선한다.
  - 후보 상태를 넘파이로 한꺼번에 계산(배치)해서 빠르게 평가한다.
  - 제한 시간이 지나면 폭 1(탐욕)로 전환해 반드시 끝까지 계획을 완성한다.

판 표현: 2차원 배열. 0 = 빈자리, 1~9 = 사과, WALL(100) = 인식 실패('?') 칸(절대 포함할 수 없는 벽)

규칙: 직사각형 영역 안 사과 숫자의 합이 10이면 그 영역의 사과가 모두 제거된다 (2개 이상, 대각선·빈자리 포함, 모든 경우의 수).

판정: 계획이 모든 사과를 제거하면 '정답 판(올클리어 가능)'. 숫자 합계가 10의 배수가 아니면 올클리어가 수학적으로 불가능.
      올클리어 경로를 못 찾으면 '무작위 판 의심'. (역순 배치로 만든 판이라면 해답이 존재하므로 경로가 나와야 한다)
"""

import threading
import time

import numpy as np

WALL = 100
NEWBOARD_CONFIRM = 1.5       # 새 판으로 의심되는 상태가 이 시간(초) 이상 계속돼야 새 판으로 인정 (제거 연출 중 일시적 변화 무시)
PLAN_SECONDS = 6.0          # 판을 처음 볼 때 계획을 계산하는 시간(초). 길수록 더 좋은 순서


class Solver:
    def __init__(self, rows, cols, target=10):
        self.R, self.C, self.T = rows, cols, target
        r1, r2, c1, c2 = [], [], [], []
        for a in range(rows):
            for b in range(a, rows):
                for c in range(cols):
                    for d in range(c, cols):
                        h, w = b - a + 1, d - c + 1
                        if h * w < 2:
                            continue
                        r1.append(a); r2.append(b); c1.append(c); c2.append(d)
        # dtype 명시: 사각형이 하나도 없을 때(격자가 너무 작거나 비었을 때) 빈 배열이 float로 만들어져 인덱싱 오류가 나는 것을 방지
        self.r1, self.r2 = np.array(r1, dtype=np.intp), np.array(r2, dtype=np.intp)
        self.c1, self.c2 = np.array(c1, dtype=np.intp), np.array(c2, dtype=np.intp)
        self._rows = np.arange(rows)[None, :, None]
        self._cols = np.arange(cols)[None, None, :]
        self._rnd = np.random.default_rng(12345).integers(1, 2 ** 62, size=rows * cols, dtype=np.uint64)

    # ───────────── 가능한 제거(=합이 10인 최소 사각형) 찾기: 여러 판을 한꺼번에 ─────────────
    def moves(self, G):
        """G:(N,R,C) → (si, mi): si=판 번호, mi=사각형 번호. 테두리 네 변에 모두 사과가 있는 '최소' 사각형만."""
        R, C = self.R, self.C
        M = len(self.r1)
        chunk = max(1, 3_000_000 // max(M, 1))
        out_s, out_m = [], []
        for a in range(0, len(G), chunk):
            g = G[a:a + chunk]
            n = len(g)
            P = np.zeros((n, R + 1, C + 1), np.int16)
            P[:, 1:, 1:] = g.astype(np.int16).cumsum(1, dtype=np.int16).cumsum(2, dtype=np.int16)
            S = (P[:, self.r2 + 1, self.c2 + 1] - P[:, self.r1, self.c2 + 1]
                 - P[:, self.r2 + 1, self.c1] + P[:, self.r1, self.c1])
            si, mi = np.nonzero(S == self.T)
            if len(si) == 0:
                continue
            Q = np.zeros((n, R + 1, C + 1), np.int16)
            Q[:, 1:, 1:] = (g > 0).astype(np.int16).cumsum(1, dtype=np.int16).cumsum(2, dtype=np.int16)
            r1, r2, c1, c2 = self.r1[mi], self.r2[mi], self.c1[mi], self.c2[mi]

            def cnt(ra, ca, rb, cb):
                return Q[si, rb + 1, cb + 1] - Q[si, ra, cb + 1] - Q[si, rb + 1, ca] + Q[si, ra, ca]

            ok = ((cnt(r1, c1, r1, c2) > 0) & (cnt(r2, c1, r2, c2) > 0) &
                  (cnt(r1, c1, r2, c1) > 0) & (cnt(r1, c2, r2, c2) > 0))
            out_s.append(si[ok] + a)
            out_m.append(mi[ok])
        if not out_s:
            return np.zeros(0, np.intp), np.zeros(0, np.intp)
        return np.concatenate(out_s), np.concatenate(out_m)

    def apply(self, G, si, mi):
        """판 si[k]에서 사각형 mi[k]를 제거한 새 판들 (N,R,C)."""
        Gc = G[si].copy()
        mask = ((self._rows >= self.r1[mi][:, None, None]) & (self._rows <= self.r2[mi][:, None, None]) &
                (self._cols >= self.c1[mi][:, None, None]) & (self._cols <= self.c2[mi][:, None, None]))
        Gc[mask] = 0
        return Gc

    def _hash(self, G):
        return (G.reshape(len(G), -1).astype(np.uint64) * self._rnd).sum(axis=1)

    # ───────────── 계획 만들기 ─────────────
    def live_count(self, child, csi, cmi):
        """각 판에서 '어떤 제거에든 포함될 수 있는' 사과 수 (남은 가능성)."""
        R, C = self.R, self.C
        cov = np.zeros((len(child), R + 1, C + 1), np.int32)
        r1, r2, c1, c2 = self.r1[cmi], self.r2[cmi], self.c1[cmi], self.c2[cmi]
        np.add.at(cov, (csi, r1, c1), 1)
        np.add.at(cov, (csi, r1, c2 + 1), -1)
        np.add.at(cov, (csi, r2 + 1, c1), -1)
        np.add.at(cov, (csi, r2 + 1, c2 + 1), 1)
        cov = cov.cumsum(1).cumsum(2)[:, :R, :C]
        return ((cov > 0) & (child > 0) & (child < WALL)).sum(axis=(1, 2))

    def solve(self, grid, time_budget=3.0, width=16, w_mob=4.0, w_live=4.0, seed=0, noise=1e-3):
        """grid → (제거 순서 [(r1,c1,r2,c2), ...], 최종 제거 사과 수).
        각 단계는 '그 시점의 판'에서 합이 10인 최소 사각형이다."""
        rng = np.random.default_rng(seed)
        G0 = np.array([[WALL if v == "?" else (v if isinstance(v, (int, np.integer)) else 0)
                        for v in row] for row in grid], np.int16)
        live0 = int(((G0 > 0) & (G0 < WALL)).sum())
        t0 = time.time()

        layers = []                       # 각 층: (부모 번호 배열, 사각형 번호 배열)
        cur, cur_rem = G0[None], np.array([0])
        best = (-1, 0, 0)                 # (제거 수, 층, 번호): 더 이상 제거할 게 없는 최종 상태 중 최고
        depth = 0
        while True:
            si, mi = self.moves(cur)
            cnts = np.bincount(si, minlength=len(cur))
            for k in np.nonzero(cnts == 0)[0]:
                if cur_rem[k] > best[0]:
                    best = (int(cur_rem[k]), depth, int(k))
            if len(si) == 0:
                break

            child = self.apply(cur, si, mi)
            _, uniq = np.unique(self._hash(child), return_index=True)        # 같은 판 중복 제거
            si, mi, child = si[uniq], mi[uniq], child[uniq]
            rem = live0 - ((child > 0) & (child < WALL)).sum(axis=(1, 2))

            W = width if (time.time() - t0) < time_budget else 1             # 시간 초과 시 탐욕으로 마무리
            if len(child) > W:
                csi, cmi = self.moves(child)
                mob = np.bincount(csi, minlength=len(child))
                score = rem + w_mob * mob + rng.random(len(child)) * noise
                if w_live:
                    score = score + w_live * self.live_count(child, csi, cmi)
                keep = np.argsort(-score)[:W]
            else:
                keep = np.arange(len(child))
            layers.append((si[keep], mi[keep]))
            cur, cur_rem = child[keep], rem[keep]
            depth += 1

        removed, d, k = best
        path = []
        for layer in range(d - 1, -1, -1):
            psi, pmi = layers[layer]
            m = pmi[k]
            path.append((int(self.r1[m]), int(self.c1[m]), int(self.r2[m]), int(self.c2[m])))
            k = psi[k]
        return path[::-1], removed


def _solve_anytime(self, grid, budget=8.0, progress=None, **kw):
    """가장 좁은 탐색(폭 1)부터 시작해 점점 넓히며(1→2→4→8…) 제한 시간 안에서 가장 좋은 계획을 반환한다.
    모든 사과를 제거하는 순서(올클리어)를 찾으면 즉시 멈춘다.
    (정답이 있는 판은 폭 1만으로도 대개 1초 안에 찾는다. 무작위 판은 시간이 남는 만큼 더 깊게 찾아 점수를 높인다.)"""
    t0 = time.time()
    best_path, best_rem = [], -1
    live0 = sum(1 for row in grid for v in row if isinstance(v, (int, np.integer)) and 0 < v < WALL)
    width = 1
    while True:
        t1 = time.time()
        left = budget - (t1 - t0)
        path, rem = self.solve(grid, time_budget=max(0.3, left * 0.7), width=width, **kw)
        if rem > best_rem:
            best_path, best_rem = path, rem
        if progress:
            progress(best_rem, width)                  # 분석 도중 '현재 최고 점수'를 화면에 보여 주기 위함
        spent = time.time() - t1
        if best_rem >= live0:                         # 전부 제거하는 순서를 찾았으면 더 볼 필요 없음
            break
        if (time.time() - t0) + spent * 2.2 > budget or width >= 256:
            break
        width *= 2
    return best_path, best_rem


Solver.solve_anytime = _solve_anytime


def moves_to_groups(grid, path):
    """계획(사각형 목록)을 '(행,열,값)' 묶음 목록으로 변환 (그 시점에 실제로 지워지는 사과만)."""
    g = [[v if isinstance(v, int) else (0 if v is None else WALL) for v in row] for row in grid]
    groups = []
    for r1, c1, r2, c2 in path:
        grp = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if 0 < g[r][c] < WALL:
                    grp.append((r, c, g[r][c]))
                    g[r][c] = 0
        groups.append(tuple(grp))
    return groups


MAX_BOXES = 20    # 올클리어가 아닌 판에서 한 번에 보여 줄 박스 수 상한


def analyze_board(grid, think=PLAN_SECONDS, solver=None, progress=None):
    """판을 분석해 최적 제거 계획과 '정답 판 / 무작위 판' 판정 재료를 돌려준다."""
    norm = Planner.norm(grid)
    sv = solver or Solver(len(norm), len(norm[0]))
    path, removed = sv.solve_anytime(norm, think, progress=progress)
    vals = [v for row in norm for v in row if 0 < v < WALL]
    unknown = sum(1 for row in norm for v in row if v == WALL)
    return {"path": path, "removed": removed, "total": len(vals), "sum": sum(vals), "unknown": unknown,
            "full": len(vals) > 0 and removed >= len(vals),
            "sum_ok": (sum(vals) % 10 == 0) if unknown == 0 else None}


def verdict(info):
    """판정 문구와 등급('good' / 'warn' / 'bad')."""
    note = f" (인식 실패 {info['unknown']}칸 제외)" if info["unknown"] else ""
    if info["full"]:
        return f"정답 판 · 올클리어 가능 · 예상 {info['total']}점{note}", "good"
    if info["sum_ok"] is False:
        return (f"무작위 판 · 올클리어 불가 (숫자 합계 {info['sum']}이 10의 배수 아님) · 예상 {info['removed']}점", "bad")
    return f"무작위 판 의심 · 올클리어 경로 못 찾음 · 예상 {info['removed']}점{note}", "warn"


class Planner:
    """판을 처음 본 순간 '한 번만' 분석(올클리어 가능 여부 + 전체 계획)하고, 그 계획으로 사용자를 안내한다.

    - 계산은 판당 1번(백그라운드 스레드). 이후에는 계산 없이 계획 중 '지금 지울 수 있는 단계'만 골라 보여 준다.
    - 올클리어 판이면 다음 1수만, 아니면 지금 지울 수 있는 계획 단계를 모두(번호 없이) 보여 준다.
    - 사용자가 계획과 다르게 지워도 자동으로 다시 계산하지 않는다. 계획 단계 중 아직 유효한 것만 계속 안내한다.
    - 새 판(사과 수가 다시 늘어남)이거나 reset()(F9)일 때만 새로 계산한다.
    - self.banner = (판정 문구, 등급) : 화면 위쪽에 계속 표시할 내용"""

    def __init__(self, think=PLAN_SECONDS):
        self.think = think
        self.solvers = {}
        self.plan = None
        self.last_live = None
        self.min_live = None                                         # 지금까지 본 가장 적은 사과 수
        self.peak_live = 0                                           # 분석 시작 시점의 사과 수(판의 크기)
        self._susp = {}                                              # 새 판 의심이 시작된 시각
        self._last_ret = None                                        # 직전 안내 (의심 구간에 유지)
        self.banner = ("판 분석 중...", "info")
        self._progress = None          # (현재 최고 점수, 탐색 폭)
        self._quick = None             # 계산 전에 바로 알 수 있는 판정 (합계가 10의 배수가 아님)
        self._gen = 0                  # 세대 번호: reset 이후 도착한 옛 계산 결과를 버리기 위함
        self._started_gen = -1
        self._t_start = 0.0
        self._result = None

    # ── 보조 함수 ──
    @staticmethod
    def norm(grid):
        return tuple(tuple(WALL if v == "?" else (0 if v is None else int(v)) for v in row) for row in grid)

    @staticmethod
    def _fit(cur, R, C):
        """현재 판이 계획 판보다 작으면(가장자리 줄이 다 비어 격자가 줄어든 경우) 0으로 채워 크기를 맞춘다."""
        r, c = len(cur), len(cur[0])
        if r > R or c > C:
            return None
        return tuple([tuple(row) + (0,) * (C - c) for row in cur] + [(0,) * C] * (R - r))

    def _solver(self, R, C):
        if (R, C) not in self.solvers:
            self.solvers[(R, C)] = Solver(R, C)
        return self.solvers[(R, C)]

    def reset(self):
        """새 판으로 보고 계획을 버린다 (다음 update 때 다시 계산)."""
        self._gen += 1
        self.plan = None
        self.last_live = None
        self.min_live = None                                         # 지금까지 본 가장 적은 사과 수
        self.peak_live = 0                                           # 분석 시작 시점의 사과 수(판의 크기)
        self._susp = {}
        self._last_ret = None
        self._result = None

    def _start(self, cur):
        gen = self._gen
        sv = self._solver(len(cur), len(cur[0]))

        vals = [v for row in cur for v in row if 0 < v < WALL]
        has_unknown = any(v == WALL for row in cur for v in row)
        self._quick = (sum(vals) if (vals and not has_unknown and sum(vals) % 10) else None)
        self._progress = None

        def on_progress(best, width):
            if gen == self._gen:
                self._progress = (best, width)

        def work():
            info = analyze_board(cur, self.think, sv, on_progress)
            if gen == self._gen:                         # 그 사이 reset 됐으면 결과를 버린다
                self._result = (gen, cur, info)

        self._started_gen = gen
        self._t_start = time.time()
        threading.Thread(target=work, daemon=True).start()

    def _adopt(self):
        gen, base, info = self._result
        self._result = None
        groups = moves_to_groups(base, info["path"])
        self.plan = {"steps": list(zip(info["path"], groups)), "info": info, "shape": (len(base), len(base[0]))}
        self.banner = verdict(info)

    @staticmethod
    def _step_state(step, g):
        """'done'(이미 지움) / 'valid'(지금 지울 수 있음) / 'waiting'(앞 단계가 먼저 필요) / 'broken'(계획과 달라짐)"""
        (r1, c1, r2, c2), group = step
        cells = {(r, c) for r, c, _ in group}
        cur = [g[r][c] for r, c, _ in group]
        if all(v == 0 for v in cur):
            return "done"
        if any(v == 0 or (v != pv and v != WALL) for v, (_, _, pv) in zip(cur, group)):
            return "broken"                                # 일부만 지워졌거나 숫자가 달라짐
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if (r, c) not in cells and g[r][c] != 0:
                    return "waiting"                       # 사각형 안에 아직 다른 사과가 남아 있음
        return "valid"

    # ── 매 화면마다 호출 (계산 없음, 순간 처리) ──
    def update(self, grid):
        """_update 의 결과를 기억해 두었다가, 새 판 의심 구간(NEWBOARD_CONFIRM 초)에는 직전 안내를 유지한다."""
        ret = self._update(grid)
        if ret[2] in ("plan", "done", "fallback"):
            self._last_ret = ret
        return ret

    def _hold(self):
        """의심 구간: 직전 박스는 화면에 그대로 두되 종류를 'hold'로 바꿔 자동 드래그가 옛 박스를 쓰지 않게 한다."""
        shown, status, _ = self._last_ret
        return shown, status + " · 새 판 확인 중", "hold"

    def _confirm(self, kind, reason):
        """같은 '새 판 의심'이 NEWBOARD_CONFIRM 초 이상 계속될 때만 True (지우는 연출의 일시적 변화는 무시)."""
        now = time.time()
        t0 = self._susp.setdefault(kind, now)
        if now - t0 >= NEWBOARD_CONFIRM:
            print(f"[새 판 감지] {reason} : 다시 분석합니다 (F9로도 수동 가능)")
            return True
        return False

    def _update(self, grid):
        """→ (묶음 목록, 상태 문장, 종류)
        종류: 'plan' 계획 안내 / 'computing' 계산 중 / 'fallback' 계획 이탈 / 'done' 모두 완료 / 'empty' 사과 없음"""
        cur = self.norm(grid)
        if not cur or not cur[0]:
            return [], "사과 없음", "empty"
        live = sum(1 for row in cur for v in row if v > 0)         # '?' 칸도 사과로 센다 (읽기 실패와 무관하게)
        # 새 판 감지: 지우는 연출 중 사과가 잠깐 안 읽혔다가 다시 읽히는 흔들림(수 개)은 무시하고,
        # '지금까지 본 최저치보다 크게(판의 15% 또는 12개 이상) 늘었을 때'만 새 판으로 본다.
        if self.min_live is not None and live > self.min_live + max(12, int(0.15 * self.peak_live)):
            if self._confirm("count", f"사과 수 {self.min_live} → {live}"):
                self.reset()
            elif self._last_ret is not None:
                return self._hold()                                  # 연출 중일 수 있으니 직전 안내 유지(드래그는 안 함)
        else:
            self._susp.pop("count", None)
        if self.min_live is None or live < self.min_live:
            self.min_live = live
        self.peak_live = max(self.peak_live, live)
        self.last_live = live

        if self._result is not None and self._result[0] == self._gen:
            self._adopt()

        if self.plan is not None:
            R, C = self.plan["shape"]
            fit = self._fit(cur, R, C)
            if fit is None:                                          # 격자가 계획보다 커짐 = 새 판(이거나 연출 중 잘못 읽힌 것)
                if self._confirm("fit", f"격자 크기 {len(cur)}x{len(cur[0])} > 계획 {R}x{C}"):
                    self.reset()
                elif self._last_ret is not None:
                    return self._hold()
            else:
                self._susp.pop("fit", None)
                info = self.plan["info"]
                states = [self._step_state(st, fit) for st in self.plan["steps"]]
                n, done, broken = len(states), states.count("done"), states.count("broken")
                valid = [st[1] for st, s in zip(self.plan["steps"], states) if s == "valid"]
                if info["full"] and broken:
                    self.banner = ("올클리어 계획에서 이탈했습니다 · F9로 현재 판을 다시 분석", "warn")
                else:
                    self.banner = verdict(info)
                warn = f" · 계획 이탈 {broken}단계" if broken else ""
                if valid:
                    shown = valid[:1] if info["full"] else valid[:MAX_BOXES]     # 올클리어 판: 1수씩 리드
                    return shown, f"{done}/{n}단계{warn}", "plan"
                if done == n:
                    return [], f"계획 완료 · {done}/{n}단계", "done"
                return [], "계획 이탈 · 보이는 쌍 표시 중 (F9: 다시 계산)", "fallback"

        if self._started_gen != self._gen:
            self._start(cur)
        left = max(0.0, self.think - (time.time() - self._t_start))
        best = f" · 현재 최고 {self._progress[0]}점" if self._progress else ""
        if self._quick is not None:                                  # 합계가 10의 배수가 아니면 계산 전에 이미 확정
            self.banner = (f"무작위 판 · 올클리어 불가 (숫자 합계 {self._quick}이 10의 배수 아님){best} · 점수 계산 중 {left:.0f}초", "bad")
        else:
            self.banner = (f"판 분석 중 (올클리어 탐색){best} · 최대 {left:.0f}초", "info")
        return [], "분석 중", "computing"
