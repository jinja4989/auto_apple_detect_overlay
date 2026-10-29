"""
apple_backtrack.py - 휴리스틱 백트래킹 솔버 (apple_solver.Solver 의 NumPy 후보 필터링을 재사용)

입력  : 10x17 격자 (숫자 1~9, 빈칸 0/None, 못 읽은 칸 '?')
출력  : (지울 직사각형 순서 [(r1,c1,r2,c2), ...], 제거한 사과 수)

구조
  1) 후보 생성  : Solver.moves() 가 NumPy 2차원 누적합으로 '합이 10인 최소 직사각형'(네 변에 모두 사과) 을 한 번에 걸러 낸다.
  2) 휴리스틱   : 각 후보를 적용한 자식 판을 배치로 평가해 점수를 매기고 높은 순으로 정렬한다.
       점수 = 제거 수 + w_mob * (남은 후보 수) + w_live * (아직 어떤 제거에든 쓰일 수 있는 사과 수)
       → 고립(어디에도 못 쓰이는) 사과가 적고 선택지가 많은 쪽을 먼저 탐색
  3) 백트래킹   : 깊이 우선. 막다른 길(더 지울 게 없는데 사과가 남음)을 만나면 직전 선택지로 되돌아가
                 다음 순위 후보를 시도한다. 이미 실패한 판(해시)은 다시 방문하지 않는다.
  4) 폭 제한 반복: 각 단계에서 상위 k개 후보만 시도(k=1,2,3,...)하는 반복 심화 + 약한 무작위 잡음.
  5) 시간 제한  : time_budget 초를 넘으면 즉시 중단하고 그때까지의 최고 기록 경로를 반환한다.
"""
import time

import numpy as np

import apple_solver as asv

WALL = asv.WALL


class _Timeout(Exception):
    pass


def solve_backtrack(grid, time_budget=3.0, w_mob=4.0, w_live=4.0, max_eval=48, seed=0, solver=None, k_max=8):
    R, C = len(grid), len(grid[0])
    sv = solver or asv.Solver(R, C)
    G0 = np.array([[WALL if v == "?" else (v if isinstance(v, (int, np.integer)) else 0) for v in row]
                   for row in grid], np.int16)
    live0 = int(((G0 > 0) & (G0 < WALL)).sum())
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = {"rem": -1, "path": []}
    stats = {"nodes": 0}

    def live_of(G):
        return int(((G > 0) & (G < WALL)).sum())

    def dfs(G, path, k, dead):
        if time.time() - t0 > time_budget:
            raise _Timeout
        stats["nodes"] += 1
        rem = live0 - live_of(G)
        si, mi = sv.moves(G[None])
        if len(si) == 0:                                     # 막다른 길(또는 올클리어)
            if rem > best["rem"]:
                best["rem"], best["path"] = rem, list(path)
            return rem >= live0
        child = sv.apply(G[None], si, mi)
        h = sv._hash(child)
        _, uniq = np.unique(h, return_index=True)
        uniq = [u for u in uniq if int(h[u]) not in dead]    # 이미 실패한 판 제외
        if not uniq:
            return False
        uniq = np.array(uniq)
        mi, child, h = mi[uniq], child[uniq], h[uniq]
        # 후보가 너무 많으면 '지우는 사과가 적은' 것부터 일부만 정밀 평가 (연산 폭발 방지)
        if len(child) > max_eval:
            removed_n = live_of_batch(child)
            order = np.argsort(-removed_n + rng.random(len(child)) * 0.5)[:max_eval]
            mi, child, h = mi[order], child[order], h[order]
        csi, cmi = sv.moves(child)
        mob = np.bincount(csi, minlength=len(child))
        score = w_mob * mob + w_live * sv.live_count(child, csi, cmi) + rng.random(len(child)) * 1e-3
        score = score - live_of_batch(child) * 1.0           # 지운 만큼 가산 (남은 사과 수가 작을수록 좋음)
        for j in np.argsort(-score)[:k]:                     # 상위 k개만 시도
            m = mi[j]
            path.append((int(sv.r1[m]), int(sv.c1[m]), int(sv.r2[m]), int(sv.c2[m])))
            if dfs(child[j], path, k, dead):
                return True
            path.pop()
            dead.add(int(h[j]))                              # 이 판에서는 해답을 못 찾음
        return False

    def live_of_batch(B):
        return ((B > 0) & (B < WALL)).sum(axis=(1, 2))

    try:
        for k in range(1, k_max + 1):                        # 폭 제한 반복(1, 2, 3, ...)
            if dfs(G0, [], k, set()):
                break
    except _Timeout:
        pass
    return best["path"], max(best["rem"], 0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    print("사용: from apple_backtrack import solve_backtrack; path, removed = solve_backtrack(grid, 3.0)")
