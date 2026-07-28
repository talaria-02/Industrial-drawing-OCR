# -*- coding: utf-8 -*-
"""외곽선을 노드/엣지 그래프로 쪼개고, 빠진 치수를 방정식으로 채운다.

[왜 선분 단위로는 안 되는가]
"하나의 외곽선에 치수가 여러 개 겹친다"는 건 선을 단위로 봤기 때문에 생기는
착시다. 10cm 자에 3과 10이 적혀 있으면 물리적으로는 한 개의 뼈대지만,
논리적으로는 점 세 개(0, 3, 10)와 구간 두 개다. 치수를 '선분'이 아니라
'점 쌍'에 걸면 겹침 자체가 사라진다.

    선분 기반:  [외곽선 A] = 10   <- 3을 적을 자리가 없다
    노드 기반:  N1-N2 = 3,  N1-N3 = 10,  N2-N3 = 7(계산)

[노드를 어디서 얻는가 — 역추적과 맞물린다]
따로 찾을 필요가 없다. traceback_points가 이미 치수마다 두 점을 준다:
치수선에서 90도로 치수보조선을 타고 올라가 외곽선에 닿는 지점. 그 점이 곧
외곽선을 잘라야 할 자리다. 치수가 세 개면 칼자국이 세 쌍 생기고, 같은 외곽선
위에 모이면 자연히 직렬 체인이 된다.

[빠진 치수는 방정식으로]
도면은 중복 없이 최소한만 적는다. 3과 10만 적혀 있으면 7은 사람이 빼서 안다.
같은 직선 위 노드들을 좌표순으로 세우면 '구간의 합 = 전체'라는 1차 방정식이
서고, 미지수가 하나면 바로 풀린다. 계산으로 얻은 값은 is_explicit=False로
표시해 OCR로 읽은 값과 구분한다 — 나중에 학습에 쓸 때 섞이면 안 된다.

[모순을 삼키지 않는다]
치수 기입 실수, OCR 오독, 축척 다른 뷰가 섞이면 방정식이 맞지 않는다.
그때 조용히 하나를 버리면 틀린 값이 그대로 흘러간다. 어긋난 정도를 기록해
사람에게 보고한다.
"""
import numpy as np

# 두 점이 이만큼 가까우면 같은 노드로 본다. 글자 높이 대비 비율 —
# 절대 픽셀로 두면 도면 축척이 다를 때 깨진다(글자 높이는 13~35px로 2.7배 차이).
NODE_MERGE_RATIO = 0.35
# 노드가 한 직선 위에 있다고 볼 수직 허용치(글자 높이 대비).
COLLINEAR_RATIO = 0.30
# 방정식이 이보다 어긋나면 모순으로 보고한다(구간 길이 대비).
CONFLICT_RATIO = 0.05


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def build_nodes(measure_points, text_h, extra_points=()):
    """측정점들을 모아 가까운 것끼리 하나의 노드로 합친다.

    measure_points : {text_id: [(x,y), (x,y)]}
    반환 (nodes, ref) — nodes: [(x,y)...],  ref: {text_id: (n_i, n_j)}
    """
    tol = max(2.0, text_h * NODE_MERGE_RATIO)
    nodes, ref = [], {}

    def node_of(p):
        p = np.asarray(p, float)
        for k, q in enumerate(nodes):
            if np.linalg.norm(p - q) <= tol:
                # 같은 노드로 판정되면 평균으로 위치를 다듬는다
                nodes[k] = (q + p) / 2
                return k
        nodes.append(p)
        return len(nodes) - 1

    for tid, pts in measure_points.items():
        if not pts or len(pts) < 2:
            continue
        ref[tid] = (node_of(pts[0]), node_of(pts[1]))
    for p in extra_points:
        node_of(p)
    return [tuple(map(float, q)) for q in nodes], ref


def collinear_chains(nodes, ref, text_h):
    """치수가 걸린 노드들을 '같은 직선 위' 묶음으로 나눈다.

    방정식(구간 합 = 전체)은 일직선 위에서만 성립한다. 축이 다른 치수를 한
    방정식에 넣으면 엉뚱한 값이 나오므로, 먼저 축별로 갈라야 한다.
    반환: [{axis, origin, members:[node_idx...], order:[node_idx 정렬]}]
    """
    tol = max(1.5, text_h * COLLINEAR_RATIO)
    N = [np.asarray(p, float) for p in nodes]
    chains = []
    for tid, (a, b) in ref.items():
        if a == b:
            continue
        axis = _unit(N[b] - N[a])
        placed = False
        for ch in chains:
            # 방향이 같고(부호 무시), 두 노드가 그 직선에서 벗어나지 않으면 합류
            if abs(float(axis @ ch["axis"])) < 0.985:
                continue
            n = np.array([-ch["axis"][1], ch["axis"][0]])
            if max(abs(float((N[a] - ch["origin"]) @ n)),
                   abs(float((N[b] - ch["origin"]) @ n))) > tol:
                continue
            ch["members"].update((a, b))
            ch["dims"].append(tid)
            placed = True
            break
        if not placed:
            chains.append({"axis": axis, "origin": N[a].copy(),
                           "members": {a, b}, "dims": [tid]})
    for ch in chains:
        ch["order"] = sorted(ch["members"], key=lambda k: float(N[k] @ ch["axis"]))
        ch["axis"] = tuple(map(float, ch["axis"]))
        ch["origin"] = tuple(map(float, ch["origin"]))
        ch["members"] = sorted(ch["members"])
    return chains


def build_edges(nodes, chains):
    """체인 위 이웃한 노드를 이어 논리 구간(엣지)을 만든다.

    긴 외곽선 하나가 치수 개수만큼 잘려 직렬로 연결된다 — 이게 '한 선에 여러
    치수' 문제가 사라지는 지점이다.
    """
    edges = []
    for ci, ch in enumerate(chains):
        o = ch["order"]
        for k in range(len(o) - 1):
            edges.append({"id": f"E{len(edges)+1}", "start": o[k], "end": o[k + 1],
                          "chain": ci, "type": "object_line"})
    return edges


def solve_chain(nodes, chain, known, tol_ratio=CONFLICT_RATIO):
    """한 체인에서 빠진 구간 길이를 방정식으로 채운다.

    known : {(node_i, node_j): value}  도면에 적힌 치수 (mm 또는 도면 단위)
    반환 (solved, conflicts)
      solved    : {(i,j): (value, derived_from)}  새로 알아낸 것만
      conflicts : [{pair, stated, implied, diff}]  적힌 값끼리 안 맞는 경우

    구간을 미지수로 두고 누적 위치를 세운다. 노드 순서가 정해져 있으므로
    '앞에서부터의 누적 거리'만 알면 모든 쌍의 거리가 결정된다.
    """
    order = chain["order"]
    pos = {n: None for n in order}          # 체인 시작점 기준 누적 좌표
    pos[order[0]] = 0.0
    idx = {n: k for k, n in enumerate(order)}

    pairs = {}
    for (a, b), v in known.items():
        if a in idx and b in idx:
            lo, hi = (a, b) if idx[a] < idx[b] else (b, a)
            pairs[(lo, hi)] = float(v)

    conflicts = []
    # 알려진 값으로 누적 좌표를 최대한 전파한다(반복하며 퍼뜨림)
    for _ in range(len(order) + 1):
        changed = False
        for (a, b), v in pairs.items():
            if pos[a] is not None and pos[b] is None:
                pos[b] = pos[a] + v
                changed = True
            elif pos[b] is not None and pos[a] is None:
                pos[a] = pos[b] - v
                changed = True
            elif pos[a] is not None and pos[b] is not None:
                implied = pos[b] - pos[a]
                if abs(implied - v) > max(1e-6, tol_ratio * max(abs(v), 1.0)):
                    conflicts.append({"pair": (a, b), "stated": v,
                                      "implied": float(implied),
                                      "diff": float(implied - v)})
        if not changed:
            break

    solved = {}
    for k in range(len(order) - 1):
        a, b = order[k], order[k + 1]
        if (a, b) in pairs or pos[a] is None or pos[b] is None:
            continue
        src = [f"{x}-{y}" for (x, y) in pairs]
        solved[(a, b)] = (float(pos[b] - pos[a]), src)
    # 이웃뿐 아니라 건너뛴 구간도 채운다(전체 길이 등)
    for i in range(len(order)):
        for j in range(i + 2, len(order)):
            a, b = order[i], order[j]
            if (a, b) in pairs or pos[a] is None or pos[b] is None:
                continue
            solved[(a, b)] = (float(pos[b] - pos[a]),
                              [f"{x}-{y}" for (x, y) in pairs])
    # 중복 보고 제거
    seen, uniq = set(), []
    for c in conflicts:
        k = (c["pair"], round(c["stated"], 4))
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return solved, uniq


def build_graph(measure_points, dim_values, text_h):
    """전체 조립. 반환 dict(nodes, edges, chains, dimensions, conflicts).

    measure_points : {text_id: [(x,y),(x,y)]}   traceback_points 결과
    dim_values     : {text_id: 공칭값}          compare.parse_dimension 결과
    """
    nodes, ref = build_nodes(measure_points, text_h)
    chains = collinear_chains(nodes, ref, text_h)
    edges = build_edges(nodes, chains)

    dims, conflicts = [], []
    for tid, (a, b) in ref.items():
        v = dim_values.get(tid)
        dims.append({"id": tid, "value": v, "ref_nodes": [a, b],
                     "is_explicit": v is not None})

    for ch in chains:
        known = {}
        for tid in ch["dims"]:
            v = dim_values.get(tid)
            if v is not None:
                known[ref[tid]] = v
        if len(known) < 1:
            continue
        solved, conf = solve_chain(nodes, ch, known)
        conflicts.extend(conf)
        for (a, b), (val, src) in solved.items():
            dims.append({"id": f"derived_{a}_{b}", "value": round(val, 4),
                         "ref_nodes": [a, b], "is_explicit": False,
                         "derived_from": src})
    return {"nodes": nodes, "edges": edges, "chains": chains,
            "dimensions": dims, "conflicts": conflicts}
