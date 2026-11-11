"""
架构图渲染脚本 - 生成 HR-Compliance-RAG 系统架构图 PNG
运行: python scripts/make_architecture.py
产物: docs/architecture.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "docs" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(17, 9.5))
ax.set_xlim(0, 17)
ax.set_ylim(0, 9.5)
ax.axis("off")

C_DATA = "#FFE0B2"   # 数据源
C_OFF = "#BBDEFB"    # 离线
C_ON = "#C8E6C9"     # 在线
C_OUT = "#F8BBD0"    # 输出
EDGE = "#37474F"


def box(x, y, w, h, text, color, fs=9.5, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                       linewidth=1.4, edgecolor=EDGE, facecolor=color)
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color="#102027", linespacing=1.5)


def arrow(x1, y1, x2, y2, style="-|>", color=EDGE, ls="-", lw=1.6):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=17,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)


# 标题
ax.text(8.5, 9.1, "企业 HR 合规智能问答系统 (HR-Compliance-RAG) 架构",
        ha="center", va="center", fontsize=17, fontweight="bold", color="#0D47A1")
ax.text(8.5, 8.62, "复用 C8 食谱RAG 骨架 · 数据源: 国家法律法规数据库(真实) · LangChain + FAISS + bge + kimi",
        ha="center", va="center", fontsize=10, color="#455A64")

# ---------------- 离线建库泳道 ----------------
ax.text(0.35, 7.75, "离线建库", fontsize=12, fontweight="bold", color="#1565C0")
box(0.35, 6.0, 3.0, 1.55,
    "真实数据源\ntwang2218/law-datasets\n国家法律法规数据库\n(22552 部法规)", C_DATA, 9, True)
box(3.7, 6.0, 3.1, 1.55,
    "① download_data.py\n下载 laws.json\n(~98MB, git-lfs)", C_OFF, 9.5)
box(7.15, 6.0, 3.35, 1.55,
    "② build_corpus.py\n筛选劳动用工法规(现行有效)\n解析 编/章/条\n→ data/laws/*.md", C_OFF, 9.5)
box(10.85, 6.0, 2.9, 1.55,
    "③ ingestion.py\n按“条”分块 + 父子映射\n元数据: 效力级别/\n领域/条号/时效", C_OFF, 9.5)
box(14.1, 6.0, 2.55, 1.55,
    "④ indexing.py\nbge-small-zh\n向量化\nFAISS 持久化", C_OFF, 9.5)

arrow(3.35, 6.78, 3.7, 6.78)
arrow(6.8, 6.78, 7.15, 6.78)
arrow(10.5, 6.78, 10.85, 6.78)
arrow(13.75, 6.78, 14.1, 6.78)

# ---------------- 在线问答泳道 ----------------
ax.text(0.35, 4.55, "在线问答", fontsize=12, fontweight="bold", color="#2E7D32")
box(0.35, 2.55, 2.5, 1.7,
    "用户问题\n+ 角色视角\n(员工/HR/法务)", C_OUT, 9.5, True)
box(3.2, 2.55, 3.5, 1.7,
    "⑤ dialogue.py\n意图路由\nlist/detail/general/out_of_scope\n查询重写 + 越界兜底(guardrail)", C_ON, 9.5)
box(7.05, 2.55, 4.3, 1.7,
    "⑥ retrieval.py\n向量 + jieba-BM25 混合检索\nRRF 重排 + 效力级别优先加权\n+ 元数据前置过滤", C_ON, 9.5)
box(11.7, 2.55, 2.5, 1.7,
    "命中法条子块\n↓\n父法规回溯\n(按命中条数)", C_ON, 9.5)
box(14.55, 2.55, 2.1, 1.7,
    "⑦ generation.py\n合规回答\n+确定性引用溯源\n(条号+链接+免责)", C_ON, 9)

arrow(2.85, 3.4, 3.2, 3.4)
arrow(6.7, 3.4, 7.05, 3.4)
arrow(11.35, 3.4, 11.7, 3.4)
arrow(14.2, 3.4, 14.55, 3.4)

# 索引 从离线 feeding 到在线检索
arrow(15.37, 6.0, 9.2, 4.25, color="#1565C0", ls="--", lw=1.8)
ax.text(12.6, 5.25, "向量索引 / 法条库", fontsize=9, color="#1565C0", style="italic")

# 输出
arrow(15.6, 2.55, 15.6, 1.75, color="#AD1457")
box(13.4, 0.95, 3.35, 0.8, "带法条溯源的合规回答", C_OUT, 10, True)

# ---------------- 技术亮点 ----------------
hl = ("四大技术亮点： ① 父子文档检索(条→法)   ② 混合检索 jieba-BM25 + RRF + 效力级别优先   "
      "③ 确定性引用溯源(法条号+官方链接+免责声明)   ④ 意图路由 + 越界兜底防幻觉")
ax.text(0.35, 1.35, hl, fontsize=10.5, color="#37474F", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", edgecolor="#FBC02D", linewidth=1.3))
ax.text(0.35, 0.55, "P2 规划：多轮对话改写 · 角色视角差异化 · RAGAS+法条引用准确率评测    |    P3：bge-reranker 精排 · Streamlit UI · 版本时效提醒",
        fontsize=9, color="#607D8B", va="center")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("[OK] architecture saved:", str(OUT))
