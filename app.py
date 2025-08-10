import json
import pandas as pd
from pathlib import Path
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, ctx
import dash
import plotly.graph_objects as go
from dash import MATCH
import numpy as np
from dash import State 
import scipy
from scipy.stats import ks_2samp

# ------------------- ユーティリティ -------------------
def ecdf_np(x):
    x = np.asarray(pd.to_numeric(x, errors="coerce").dropna(), dtype=float)
    if x.size == 0:
        return np.array([]), np.array([])
    x_sorted = np.sort(x)
    y = np.arange(1, x_sorted.size + 1) / x_sorted.size
    return x_sorted, y

def load_cdfks_bundle(name: str):
    """保存済みの cdf_ks_data/cdf_{name}.json を読み込む。無ければ None"""
    p = Path(f"cdf_ks_data/cdf_{name}.json")
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)

# JSONファイルの読み込み関数
def load_data(file_path, task_name):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        rows = []
        for row in data["results"]:
            giveup = row[1] is None
            if task_name == "task1" or task_name == "task2":
                try:
                    score = float(row[2].split("score: ")[1])
                except:
                    score = None
                if giveup:
                    score = 0.0
                rows.append({
                    "q": row[0],
                    "score": score,
                    "time": row[3],
                    "colorChange": str(row[4]),
                    "depth": row[5],
                    "task": task_name,
                    "giveup": giveup
                })
            else:
                if giveup:
                    score = 0.0
                else:
                    score = 1.0 if row[2] == "○" else 0.0
                rows.append({
                    "q": row[0],
                    "score": score,
                    "time": row[3],
                    "colorChange": str(row[4]),
                    "depth": row[5],
                    "task": task_name,
                    "giveup": giveup
                })
        return pd.DataFrame(rows)
    
def get_highlight_q(q_num):
    q = int(q_num)
    if q + 20 <= 39:
        return f"q{q + 20}"
    else:
        return f"q{q - 20}"


# 参加者×日付の候補（必要に応じて追加/並べ替えOK）
PAIRS_RAW = [
    ("2025-07-29", "P6"),
    ("2025-07-29", "P7"),
    ("2025-07-30", "P8"),
    ("2025-07-30", "P9"),   
    ("2025-07-30", "P10"),
    ("2025-07-30", "P11"),
    ("2025-08-01", "P12"),
    ("2025-08-01", "P13"),
    ("2025-08-01", "P14"),
    ("2025-08-02", "P15"),
    ("2025-08-02", "P16"),
    ("2025-08-02", "P17"),
    ("2025-08-03", "P18"),
    ("2025-08-06", "P19"),
]

# ドロップダウンに使う [{label, value}] 形式（valueは "date|name"）
PAIR_OPTIONS = [{"label": f"{d} {p}", "value": f"{d}|{p}"} for d, p in PAIRS_RAW]
DEFAULT_PAIR = PAIR_OPTIONS[0]["value"]  # 先頭を初期値

def build_figures_for(name, date):
    # 動的にファイルパスを構築
    task_names = [f"task{i}" for i in range(1, 6)]
    file_map = {
        task: f"test/v0/{name}/{date}-{name}-{task}.json"
        for task in task_names
    }

    # すべてロード
    df_list = []
    for task, file in file_map.items():
        df_list.append(load_data(file, task))
    df_all = pd.concat(df_list)

    figures = {}
    time_fig_ids = []
    bundle = load_cdfks_bundle(name)

    # ======== ここから先は、あなたの for ループ本体をほぼコピペでOK ========
    for task in file_map.keys():
        df = df_all[df_all["task"] == task].copy()
        total_score = df["score"].sum()
        if task in ["task1", "task2"]:
            title_suffix = f"（{total_score:.2f}/40.00）"
        else:
            title_suffix = f"（{int(total_score)}/40）"

        df["z_time"] = (df["time"] - df["time"].mean()) / df["time"].std()
        df["status"] = df.apply(
            lambda r: "Give up" if r["giveup"] 
                      else ("Correct" if r["score"] == 1.0 else "Incorrect"),
            axis=1
        )

        def apply_conditional_jitter(df, jitter_distance=300, jitter_step=0.1):
            df = df.copy()
            df["jittered_depth"] = df["depth"].astype(float)
            for d in df["depth"].unique():
                group = df[df["depth"] == d]
                times = group["time"].values
                indices = group.index.tolist()
                assigned = [False] * len(times)
                offsets = np.zeros(len(times))
                for i in range(len(times)):
                    if assigned[i]:
                        continue
                    cluster = [i]
                    for j in range(i + 1, len(times)):
                        if abs(times[i] - times[j]) < jitter_distance:
                            cluster.append(j)
                    n = len(cluster)
                    for k, idx in enumerate(sorted(cluster)):
                        offsets[idx] = (k - (n - 1)/2) * jitter_step
                        assigned[idx] = True
                df.loc[indices, "jittered_depth"] += offsets
            return df

        # ---- score 図 ----
        if task in ["task1", "task2"]:
            df = apply_conditional_jitter(df)
            fig_score = px.scatter(
                df, x="jittered_depth", y="score", color="status",
                hover_name="q", title=f"{task}: Score vs Depth {title_suffix}",
                category_orders={"status": ["Correct", "Incorrect", "Give up"]},
                color_discrete_map={"Correct":"blue","Incorrect":"red","Give up":"black"}
            )
            # 凡例名を変更
            fig_score.for_each_trace(
                lambda t: t.update(name={
                    "Correct": ": ○ (Correct)",
                    "Incorrect": ": × (Incorrect)",
                    "Give up": ": Give up"
                }.get(t.name, t.name))
            )
            low_score_df = df[df["score"] <= 0.3].copy()
            low_score_df["label"] = low_score_df["q"].str.replace("^q", "", regex=True)
            low_score_df = low_score_df.sort_values("jittered_depth").reset_index(drop=True)
            y_offset = 0.04
            y_positions = [s + (y_offset if i % 2 == 0 else -y_offset) 
                           for i, s in enumerate(low_score_df["score"])]
            fig_score.add_trace(go.Scatter(
                x=low_score_df["jittered_depth"], y=y_positions,
                mode="text", text=low_score_df["label"],
                textposition="middle center", textfont=dict(size=9),
                showlegend=False
            ))
            mean_0 = df[df["colorChange"] == "0"]["score"].mean()
            mean_1 = df[df["colorChange"] == "1"]["score"].mean()
            fig_score.add_hline(y=mean_0, line=dict(color="red", dash="dash"),
                                annotation_text="mean score", annotation_position="bottom left")
            fig_score.add_hline(y=mean_1, line=dict(color="blue", dash="dash"),
                                annotation_text="mean score", annotation_position="top left")
            fig_score.update_layout(yaxis=dict(range=[-0.1, 1.1], fixedrange=True))
            fig_score.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font=dict(size=10))
            )
            score_for_symbol = (df["score"] >= 0.5).astype(float)
        else:
            df["correct"] = df["score"].map({1.0: "Correct", 0.0: "Incorrect"})
            fig_score = px.histogram(
                df, x="depth", color="correct", barmode="stack", facet_col="colorChange",
                category_orders={"correct": ["Correct", "Incorrect", "Give up"]},
                color_discrete_map={"Correct":"blue","Incorrect":"red","Give up":"black"},
                title=f"{task}: Correctness by depth × coloring {title_suffix}"
            )
            # 凡例名を変更
            fig_score.for_each_trace(
                lambda t: t.update(name={
                    "Correct": "○",
                    "Incorrect": "×",
                    "Give up": "Give up"
                }.get(t.name, t.name))
            )
            fig_score.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font=dict(size=10))
            )
            score_for_symbol = df["score"]
            fig_score.for_each_annotation(lambda a: a.update(
                text=a.text.replace("colorChange=0", "Static coloring")
                          .replace("colorChange=1", "Dynamic coloring")
            ))
            ymax = df.groupby(["depth", "colorChange"])["score"].count().max()
            for cc in [0, 1]:
                incorrect_df = df[(df["score"] == 0.0) & (df["colorChange"] == str(cc))].copy()
                incorrect_df["label_y"] = np.nan
                for d in incorrect_df["depth"].unique():
                    group_idx = incorrect_df[incorrect_df["depth"] == d].index
                    n = len(group_idx)
                    offsets = [ymax + 0.5 - k*0.3 for k in range(n)]
                    incorrect_df.loc[group_idx, "label_y"] = offsets
                fig_score.add_trace(
                    go.Scatter(
                        x=incorrect_df["depth"].astype(float),
                        y=incorrect_df["label_y"], mode="text",
                        text=incorrect_df["q"], textposition="top center",
                        showlegend=False
                    ), row=1, col=cc+1
                )

        # ---- time 図 ----
        df["score_bin"] = score_for_symbol.astype(int)
        df["color"] = df.apply(
            lambda row: f"{'Dynamic' if row['colorChange'] == '1' else 'Static'}・{'Correct' if row['score_bin'] == 1 else 'Incorrect'}",
            axis=1
        )
        custom_color_map = {
            "Static・Incorrect": "gray",
            "Static・Correct": "blue",
            "Dynamic・Incorrect": "orange",
            "Dynamic・Correct": "red"
        }
        df = apply_conditional_jitter(df)
        fig_time = px.scatter(
            df, x="jittered_depth", y="time", color="color",
            hover_name="q", title=f"{task}: Time vs Depth {title_suffix}",
            category_orders={
                "color": [
                    "Static・Correct", "Static・Incorrect", "Static・Give up",
                    "Dynamic・Correct", "Dynamic・Incorrect", "Dynamic・Give up"
                ]
            },
            color_discrete_map=custom_color_map
        )
        
        fig_time.for_each_trace(
            lambda t: t.update(name=t.name
                .replace("Correct", "○")
                .replace("Incorrect", "×")
            )
        )
        outliers = df[df["z_time"] >= 1.5]
        fig_time.add_trace(go.Scatter(
            x=outliers["jittered_depth"], y=outliers["time"],
            text=outliers["q"], mode="text", textposition="top center",
            showlegend=False
        ))
        fig_time.update_layout(
            xaxis=dict(fixedrange=True, autorange=False, range=[df["depth"].min()-0.5, df["depth"].max()+0.5]),
            yaxis=dict(fixedrange=True, autorange=False, range=[df["time"].min()*0.9, df["time"].max()*1.1])
        )
        fig_time.update_layout(
            xaxis=dict(tickmode="array",
                       tickvals=sorted(df["depth"].unique()),
                       ticktext=[str(int(d)) for d in sorted(df["depth"].unique())],
                       fixedrange=True),
            yaxis=dict(fixedrange=True, autorange=False,
                       range=[df["time"].min()*0.9, df["time"].max()*1.1])
        )
        fig_time.update_layout(
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font=dict(size=10))
        )

        # ---- cdf 図（保存済みJSONがあれば利用、なければその場で計算）----
        xs = ys = xd = yd = np.array([])
        ks_stat = p_val = None
        n_static = n_dynamic = 0

        if bundle and "tasks" in bundle and task in bundle["tasks"]:
            t = bundle["tasks"][task]
            cdf_s = t.get("cdf", {}).get("static", {})
            cdf_d = t.get("cdf", {}).get("dynamic", {})
            xs = np.array(cdf_s.get("x", []), dtype=float)
            ys = np.array(cdf_s.get("y", []), dtype=float)
            xd = np.array(cdf_d.get("x", []), dtype=float)
            yd = np.array(cdf_d.get("y", []), dtype=float)
            ks_info = t.get("ks", {})
            ks_stat = ks_info.get("statistic", None)
            p_val = ks_info.get("p_value", None)
            n_static = ks_info.get("n_static", 0)
            n_dynamic = ks_info.get("n_dynamic", 0)
        else:
            # フォールバック: 今のdfから作る
            time_static = df[df["colorChange"] == "0"]["time"]
            time_dynamic = df[df["colorChange"] == "1"]["time"]
            xs, ys = ecdf_np(time_static)
            xd, yd = ecdf_np(time_dynamic)
            n_static, n_dynamic = len(time_static), len(time_dynamic)
            if n_static >= 2 and n_dynamic >= 2:
                ks_stat, p_val = ks_2samp(time_static, time_dynamic)
            else:
                ks_stat, p_val = None, None

        # Plotlyで階段（step-post）を再現するには line_shape="hv"
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"Static (n={n_static})",
            line=dict(shape="hv")
        ))
        fig_cdf.add_trace(go.Scatter(
            x=xd, y=yd, mode="lines", name=f"Dynamic (n={n_dynamic})",
            line=dict(shape="hv")
        ))
        fig_cdf.update_layout(
            title=dict(text=f"{task} - CDF", x=0.5, xanchor="center"),
            xaxis_title="Time (ms)",
            yaxis_title="Cumulative Probability",
            margin=dict(t=80, r=10, b=40, l=60),  # ← 上に余白を確保（重なり回避）
            legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom")
        )
        # 上段に統計値（タイトルより上・左右分離）
        if ks_stat is not None and p_val is not None:
            fig_cdf.add_annotation(
                x=0.2, y=1.12, xref="paper", yref="paper",
                text=f"KS statistic = {ks_stat:.3f}",
                showarrow=False, font=dict(size=12, color="black"), xanchor="center"
            )
            fig_cdf.add_annotation(
                x=0.66, y=1.12, xref="paper", yref="paper",
                text=f"p-value = {p_val:.3f}",
                showarrow=False, font=dict(size=12, color=("red" if p_val <= 0.05 else "black")),
                xanchor="center"
            )
        else:
            fig_cdf.add_annotation(
                x=0.5, y=1.12, xref="paper", yref="paper",
                text="KS/p: n不足で未実施", showarrow=False, font=dict(size=12), xanchor="center"
            )

        figures[f"{task}-score"] = fig_score
        figures[f"{task}-time"] = fig_time
        figures[f"{task}-cdf"]   = fig_cdf
        time_fig_ids.append(f"{task}-time")

    return figures  # dict: { "task1-score": fig, "task1-time": fig, ... }

'''
# タスク名のリスト
task_names = [f"task{i}" for i in range(1, 6)]

# 動的にファイルパスを構築
file_map = {
    task: f"test/v0/{name}/{date}-{name}-{task}.json"
    for task in task_names
}

df_list = []
for task, file in file_map.items():
    df_list.append(load_data(file, task))
df_all = pd.concat(df_list)

# アプリ定義
app = Dash(__name__)

# マーカータイプの指定（score=1.0 → ○, score=0.0 → ×）
#symbol_map = {1.0: "circle", 0.0: "x"}
# グラフ格納用
figures = {}
time_fig_ids = []

for task in file_map.keys():
    df = df_all[df_all["task"] == task].copy()
    total_score = df["score"].sum()
    # task1, task2 の場合のみ
    if task in ["task1", "task2"]:
        total_score = df["score"].sum()
        title_suffix = f"（{total_score:.2f}/40.00）"
    else:
        total_score = df["score"].sum()
        title_suffix = f"（{int(total_score)}/40）"
    
    df["z_time"] = (df["time"] - df["time"].mean()) / df["time"].std()
    df["status"] = df.apply(
        lambda r: "Give up" if r["giveup"] 
                  else ("Correct" if r["score"] == 1.0 else "Incorrect"),
        axis=1
    )
    def apply_conditional_jitter(df, jitter_distance=300, jitter_step=0.1):
        df = df.copy()
        df["jittered_depth"] = df["depth"].astype(float)  # float化しておく
        
        for d in df["depth"].unique():
            group = df[df["depth"] == d]
            times = group["time"].values
            indices = group.index.tolist()
            
            # timeの近さでクラスタ分け（簡易）
            assigned = [False] * len(times)
            offsets = np.zeros(len(times))
            
            for i in range(len(times)):
                if assigned[i]:
                    continue
                cluster = [i]
                for j in range(i + 1, len(times)):
                    if abs(times[i] - times[j]) < jitter_distance:
                        cluster.append(j)
                # 重なりの中で左右にずらす：-k, ..., 0, ..., +k
                n = len(cluster)
                for k, idx in enumerate(sorted(cluster)):
                    offsets[idx] = (k - (n - 1)/2) * jitter_step
                    assigned[idx] = True
            
            # オフセット適用
            df.loc[indices, "jittered_depth"] += offsets
        
        return df
    
    # ---- scoreグラフ（task1だけ散布図、それ以外はbar） ----
    if (task == "task1" or task == "task2"):
        df = apply_conditional_jitter(df)
        fig_score = px.scatter(
            df, x="jittered_depth", y="score", color="status",
            hover_name="q", title=f"{task}: Score vs Depth {title_suffix}",
            color_discrete_map={
                "Correct": "blue",
                "Incorrect": "red",
                "Give up": "black"
            }
        )
        # 条件付きラベル付与（例：score <= 0.3 の点だけラベル表示）
        low_score_df = df[df["score"] <= 0.3].copy()
        low_score_df["label"] = low_score_df["q"].str.replace("^q", "", regex=True)
        
        # jittered_depth順に並び替え
        low_score_df = low_score_df.sort_values("jittered_depth").reset_index(drop=True)
        
        # 上下に交互でオフセット
        y_offset = 0.04  # ← 距離を調整（大きくすればさらに離れる）
        y_positions = [
            s + (y_offset if i % 2 == 0 else -y_offset) 
            for i, s in enumerate(low_score_df["score"])
        ]
        
        fig_score.add_trace(
            go.Scatter(
                x=low_score_df["jittered_depth"],
                y=y_positions,                # ← 点から少しずらした位置に表示
                mode="text",
                text=low_score_df["label"],
                textposition="middle center",  # ← 中央揃えでOK
                textfont=dict(size=9),
                showlegend=False
            )
        )

        # 平均スコアの計算
        mean_0 = df[df["colorChange"] == "0"]["score"].mean()
        mean_1 = df[df["colorChange"] == "1"]["score"].mean()
    
        # 水平線の追加
        fig_score.add_hline(
            y=mean_0, line=dict(color="red", dash="dash"),
            annotation_text="mean score", annotation_position="bottom left"
        )
        fig_score.add_hline(
            y=mean_1, line=dict(color="blue", dash="dash"),
            annotation_text="mean score", annotation_position="top left"
        )
        fig_score.update_layout(
            yaxis=dict(range=[-0.1, 1.1], fixedrange=True)
        )
        score_for_symbol = (df["score"] >= 0.5).astype(float)
    else:
        df["correct"] = df["score"].map({1.0: "Correct", 0.0: "Incorrect"})
        fig_score = px.histogram(
            df, x="depth", color="correct",
            barmode="stack", facet_col="colorChange",
            #category_orders={"correct": ["Incorrect", "Correct"]},
            category_orders={"status": ["Incorrect","Correct","Give up"]},
            color_discrete_map={
                "Correct": "blue",
                "Incorrect": "red",
                "Give up": "black"
            },
            title=f"{task}: Correctness by depth × coloring {title_suffix}"
        )
        score_for_symbol = df["score"]
        # facet のタイトル書き換え
        fig_score.for_each_annotation(lambda a: a.update(
            text=a.text.replace("colorChange=0", "Static coloring")
                      .replace("colorChange=1", "Dynamic coloring")
        ))
        # y 軸の最大値を取得（ヒストグラムの bar の高さから推定）
        ymax = df.groupby(["depth", "colorChange"])["score"].count().max()

        for cc in [0, 1]:
            incorrect_df = df[(df["score"] == 0.0) & (df["colorChange"] == str(cc))].copy()
            incorrect_df["label_y"] = np.nan  # 初期化
        
            # depthごとに処理
            for d in incorrect_df["depth"].unique():
                group_idx = incorrect_df[incorrect_df["depth"] == d].index
                n = len(group_idx)
                # 上端から下方向に 0.3 刻みで配置
                offsets = [ymax + 0.5 - k*0.3 for k in range(n)]
                incorrect_df.loc[group_idx, "label_y"] = offsets
        
            fig_score.add_trace(
                go.Scatter(
                    x=incorrect_df["depth"].astype(float),
                    y=incorrect_df["label_y"],
                    mode="text",
                    text=incorrect_df["q"],
                    textposition="top center",
                    showlegend=False
                ),
                row=1, col=cc+1
            )

    # --- Time vs Depth（scoreによってマーカー記号を変える） ---
    # colorChangeとscoreで4値分類用のラベル列を追加
    df["score_bin"] = score_for_symbol.astype(int)  # 0 or 1
    df["color_score_group"] = df.apply(
        lambda row: f"{'Dynamic' if row['colorChange'] == '1' else 'Static'}・{'Correct' if row['score_bin'] == 1 else 'Incorrect'}",
        axis=1
    )
    
    # カラーマップ定義（任意の色を4種類指定）
    custom_color_map = {
        "Static・Incorrect": "gray",
        "Static・Correct": "blue",
        "Dynamic・Incorrect": "orange",
        "Dynamic・Correct": "red"
    }

    df = apply_conditional_jitter(df)

    fig_time = px.scatter(
        df, x="jittered_depth", y="time", color="color_score_group",
        hover_name="q",
        title=f"{task}: Time vs Depth {title_suffix}",
        color_discrete_map=custom_color_map
    )
    # 全てマーカーサイズ大きめかつcircleに統一
    #fig_time.update_traces(marker=dict(size=10, symbol="circle"))
    # z>=2 の点に q id を表示
    outliers = df[df["z_time"] >= 1.5]
    fig_time.add_trace(go.Scatter(
        x=outliers["jittered_depth"], y=outliers["time"],
        text=outliers["q"],
        mode="text", textposition="top center",
        showlegend=False
    ))

    fig_time.update_layout(
        xaxis=dict(fixedrange=True, autorange=False, range=[df["depth"].min()-0.5, df["depth"].max()+0.5]),
        yaxis=dict(fixedrange=True, autorange=False, range=[df["time"].min() * 0.9, df["time"].max()*1.1])
    )
    fig_time.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=sorted(df["depth"].unique()),
            ticktext=[str(int(d)) for d in sorted(df["depth"].unique())],
            fixedrange=True
        ),
        yaxis=dict(
            fixedrange=True,
            autorange=False,
            range=[df["time"].min() * 0.9, df["time"].max()*1.1]
        )
    )

    # 保存
    figures[f"{task}-score"] = fig_score
    figures[f"{task}-time"] = fig_time
    time_fig_ids.append(f"{task}-time")'''

# レイアウト定義（左右に並べる）
'''app.layout = html.Div([
    dcc.Location(id="url", refresh=True),
    html.H1("ToLテスト結果ダッシュボード", style={"textAlign": "center"}),
    *[
        html.Div([
            html.Div(dcc.Graph(id=f"{task}-score", figure=figures[f"{task}-score"]),
                     style={"width": "50%"}),
            html.Div(dcc.Graph(id=f"{task}-time", figure=figures[f"{task}-time"]),
                     style={"width": "50%"})
        ], style={"display": "flex", "marginBottom": "40px"})
        for task in file_map.keys()
    ]
])'''
app = Dash(__name__)
# 初期表示用（先頭のペア）
_init_date, _init_name = DEFAULT_PAIR.split("|")
_init_figs = build_figures_for(_init_name, _init_date)
GRAPH_STYLE = {"height": "420px"}

app.layout = html.Div([
    html.H1("User Study Result Dashboard", style={"textAlign": "center"}),

    # 参加者×日付の選択
    html.Div([
        html.Label("Date × Participant"),
        dcc.Dropdown(
            id="pair-select",
            options=PAIR_OPTIONS,
            value=DEFAULT_PAIR,
            clearable=False,
            style={"width": "360px"}
        )
    ], style={"display": "flex", "justifyContent": "center", "marginBottom": "8px"}),

    # 被験者名と日時の表示
    html.Div([
        html.P(id="disp-participant", style={"margin": "5px"}),
        html.P(id="disp-date", style={"margin": "5px"})
    ], style={"textAlign": "center", "fontSize": "16px", "marginBottom": "10px"}),

    *[
        html.Div([
            html.Div(
                dcc.Graph(id=f"{task}-score",
                          figure=_init_figs[f"{task}-score"],
                          style=GRAPH_STYLE, config={"responsive": True}),
                style={"width": "40%"}
            ),
            html.Div(children=[
              dcc.Graph(id=f"{task}-time",
                        figure=_init_figs[f"{task}-time"],
                        style=GRAPH_STYLE, config={"responsive": True}),
              html.Div(id=f"{task}-url",
                       style={"textAlign": "center", "marginTop": "10px"})
            ], style={"width": "40%"}),
            html.Div(
                dcc.Graph(id=f"{task}-cdf",
                          figure=_init_figs[f"{task}-cdf"],
                          style=GRAPH_STYLE, config={"responsive": True}),
                style={"width": "25%"}
            )
        ], style={"display": "flex", "marginBottom": "40px"})
        for task in [f"task{i}" for i in range(1, 6)]
    ]
])

# 図更新（score/time × 5タスク = 10出力）＋ 見出し2出力
score_outputs = [Output(f"task{i}-score", "figure") for i in range(1, 6)]
time_outputs  = [Output(f"task{i}-time", "figure")  for i in range(1, 6)]
cdf_outputs   = [Output(f"task{i}-cdf",   "figure") for i in range(1, 6)]
head_outputs  = [Output("disp-participant", "children"), Output("disp-date", "children")]

@app.callback(
    score_outputs + time_outputs + cdf_outputs + head_outputs,
    Input("pair-select", "value"),
)
def update_all_figs(pair_value):
    date, name = pair_value.split("|")
    figs = build_figures_for(name, date)
    score_figs = [figs[f"task{i}-score"] for i in range(1, 6)]
    time_figs  = [figs[f"task{i}-time"]  for i in range(1, 6)]
    cdf_figs   = [figs[f"task{i}-cdf"]   for i in range(1, 6)]
    head = [f"Participant: {name}", f"Date: {date}"]
    return score_figs + time_figs + cdf_figs + head

@app.callback(
    [Output(f"task{i}-url", "children") for i in range(1, 6)],
    [Input(f"task{i}-time", "clickData") for i in range(1, 6)],
    Input("pair-select", "value"),
    prevent_initial_call=True
)
def display_clicked_urls(*args):
    # args = [clickData_task1, ..., clickData_task5, pair_value]
    pair_value = args[-1]
    date, name = pair_value.split("|")
    clickDatas = args[:-1]

    outputs = [None]*5
    triggered_id = ctx.triggered_id
    if not triggered_id:
        return outputs

    # どのタスクのグラフが発火したか
    if triggered_id.endswith("-time"):
        task = triggered_id.replace("-time", "")
        idx = int(task.replace("task", "")) - 1
        clickData = clickDatas[idx]
        if clickData:
            q_str = clickData["points"][0]["hovertext"]
            q_num = q_str.replace("q", "")
            # ★ ここで選択中の name/date を使ってURL生成（必要ならクエリに追加）
            url = f"http://127.0.0.1:5000/ToL?task={task}&tasknum={q_num}"
            link = html.A(url, href=url, target="_blank", style={"color": "blue"})
            outputs[idx] = html.Div(["選択されたリンク: ", link])
    return outputs

# ファイル末尾あたりに追加
server = app.server  # ← Render用

if __name__ == "__main__":
    app.run_server(debug=False)
    #app.run(debug=True, port=8050)