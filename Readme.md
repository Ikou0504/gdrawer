### このリポジトリについて / About this repository

---

**Gdrawer**は、任意の2次元平面における頂点に対して$(k,l)$-MTGを描画するツールです。<br>(MTG:Minimum Weight Tight Graphs)

#### $(k,l)$-MTGの定義

$(k,l)$-MTGとは，幾何的グラフ $G=(V,E)$ に対して

- $|E| = k|V| - l$
- 任意の部分集合 $V' \subseteq V$ について $|E'| \leq k|V'| - l$
（ここで $E'$ は $V'$ 上の部分グラフの辺集合）

を満たす辺長の総和が最小なグラフのことを指します。

---

/ **Gdrawer** is a tool for drawing proximity graphs by placing vertices on a 2D plane.
<br>(MTG:Minimum Weight Tight Graphs)

#### Definition of $(k,l)$-MTG

A $(k,l)$-MTG is a geometric graph $G=(V, E)$ such that

- $|E| = k|V| - l$
- For any subset $V' \subseteq V$, $|E'| \leq k|V'| - l$
(where $E'$ is the set of edges in the subgraph induced by $V'$)
