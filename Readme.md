### このリポジトリについて / About this repository

---

**Gdrawer**は、任意の2次元平面における頂点に対して $(k,l)$ -MTGを描画するツールです。<br>(MTG:Minimum Weight Tight Graphs)

#### $(k,l)$-MTGの定義

$(k,l)$-MTGとは，幾何的グラフ $G=(V,E)$ に対して

- $|E| = k|V| - l$
- 任意の部分集合 $V' \subseteq V$ について $|E'| \leq k|V'| - l$
（ここで $E'$ は $V'$ 上のグラフの辺集合）

を満たす辺長の総和が最小なグラフのことを指します。

**実行方法**
ファイル内の Gdrawer.py を実行することで、$(k,l)$-MTGを描画できます。

**使用方法**

- 基本操作
  - Gdrawer.py を実行
  - 画面上のキャンパスで左クリック→頂点を追加
  - 画面上のキャンパスで右クリック→頂点を削除
  - 画面上のキャンパスでドラッグ→頂点を移動
  - k,lの値を変更する場合は、window左下の入力欄を変更した後、頂点を追加または移動してください.

---

/ **Gdrawer** is a tool for drawing proximity graphs by placing vertices on a 2D plane.
<br>(MTG:Minimum Weight Tight Graphs)

#### Definition of $(k,l)$-MTG

A $(k,l)$-MTG is a geometric graph $G=(V, E)$ such that

- $|E| = k|V| - l$
- For any subset $V' \subseteq V$, $|E'| \leq k|V'| - l$
(where $E'$ is the set of edges induced by $V'$)

**How to run**
You can draw $(k,l)$-MTG by running Gdrawer.py in the file.

**How to use**

- Basic Operations
  - Run Gdrawer.py
  - Left click on the canvas to add a vertex
  - Right click on the canvas to remove a vertex
  - Drag on the canvas to move a vertex
  - To change the values of k and l, modify the input fields at the bottom left of the window, then add or move vertices.
