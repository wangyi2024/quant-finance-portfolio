"""
多股对比：腾讯 vs 阿里 vs 美团（P1 地基：相关性 + 风险收益）
=========================================================
你会学到 3 个量化最常用的"组合视角"：
  1) 归一化净值  —— 不同股票放在同一起点(100)比谁涨得多
  2) 风险-收益散点 —— 横轴波动(风险)，纵轴收益，一眼看性价比
  3) 相关性矩阵 —— 两两有多"同涨同跌"，决定能不能分散风险

运行：python3 02_multi_stock_compare.py
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ---- 选股：三只港股科技 ----
TICKERS = ["0700.HK", "9988.HK", "3690.HK"]
NAMES   = {"0700.HK": "腾讯", "9988.HK": "阿里", "3690.HK": "美团"}

# ============================================================
# 1) 取数据：一次拉多只
# ============================================================
raw = yf.download(TICKERS, period="1y", auto_adjust=True)
close = raw["Close"].dropna()            # 收盘价，每列一只股票
print("=== 收盘价前几行 ===")
print(close.head())

# ============================================================
# 2) 算收益 + 年化统计
# ============================================================
returns = close.pct_change().dropna()
ann_ret = returns.mean() * 252            # 年化收益
ann_vol = returns.std() * np.sqrt(252)    # 年化波动

print("\n=== 年化收益 / 波动 / 夏普 ===")
summary = pd.DataFrame({"年化收益": ann_ret, "年化波动": ann_vol})
summary["夏普(无风险=0)"] = summary["年化收益"] / summary["年化波动"]
summary.index = [NAMES[t] for t in summary.index]
print(summary.round(3))

# ============================================================
# 3) 相关性矩阵（关键：越低越能分散风险）
# ============================================================
corr = returns.corr()
corr.index = [NAMES[t] for t in corr.index]
corr.columns = [NAMES[t] for t in corr.columns]
print("\n=== 收益率相关性矩阵（1=完全同步）===")
print(corr.round(2))

# ============================================================
# 4) 画图：3 连图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# (a) 归一化净值：起点都设为 100，看相对涨跌
norm = close / close.iloc[0] * 100
ax = axes[0]
for t in TICKERS:
    ax.plot(norm.index, norm[t], label=NAMES[t], lw=1.6)
ax.axhline(100, color='gray', ls='--', lw=0.8)
ax.set_title("(a) 归一化净值（起点=100）", fontsize=12)
ax.set_ylabel("净值"); ax.legend(); ax.grid(alpha=0.3)

# (b) 风险-收益散点：横=波动，纵=收益
ax = axes[1]
for t in TICKERS:
    ax.scatter(ann_vol[t], ann_ret[t], s=120)
    ax.annotate(NAMES[t], (ann_vol[t], ann_ret[t]),
                textcoords="offset points", xytext=(8, 6), fontsize=11)
ax.axhline(0, color='gray', lw=0.8)
ax.set_title("(b) 风险-收益（左上=又稳又赚）", fontsize=12)
ax.set_xlabel("年化波动率"); ax.set_ylabel("年化收益率"); ax.grid(alpha=0.3)

# (c) 相关性热力图
ax = axes[2]
im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=0.4, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                color="black", fontsize=11)
ax.set_title("(c) 相关性（越低越能分散）", fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out = "02_multi_stock_compare.png"
plt.savefig(out, dpi=130)
print(f"\n图已保存: {out}")
