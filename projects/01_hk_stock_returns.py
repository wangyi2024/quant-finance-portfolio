"""
第一个量化项目：港股数据 → 收益率（P1 地基入门）
=================================================
目标：用免费库 yfinance 拉取腾讯(0700.HK)近一年日线，
      计算每日收益率，做基本统计，并画图。

你会学到（Python + 量化最基础的 4 个动作）：
  1) 取数据   2) 算收益   3) 看统计   4) 画图

运行：python3 01_hk_stock_returns.py
"""
import yfinance as yf          # 免费拉雅虎财经数据
import pandas as pd            # 数据分析主力库
import numpy as np             # 数值计算
import matplotlib.pyplot as plt

# ---- 中文字体（macOS）----
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1) 取数据：腾讯港交所代码 0700.HK（港股代码后加 .HK）
# ============================================================
ticker = "0700.HK"
data = yf.download(ticker, period="1y", auto_adjust=True)

# yfinance 有时返回双层列名，压平成单层，方便后面用 data["Close"]
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print("=== 数据前 5 行 ===")
print(data.head())

# ============================================================
# 2) 算收益：今日相对昨日的涨跌幅
#    简单收益率 r_t = (P_t - P_{t-1}) / P_{t-1} = P_t/P_{t-1} - 1
#    pct_change() 一行搞定
# ============================================================
close = data["Close"]
data["Return"] = close.pct_change()
print("\n=== 最近 5 天：收盘价 + 每日收益率 ===")
print(data[["Close", "Return"]].tail())

# ============================================================
# 3) 看统计：年化收益与年化波动率（量化最常报的两个数）
#    一年约 252 个交易日
# ============================================================
rets = data["Return"].dropna()
ann_return = rets.mean() * 252            # 日均×252 ≈ 年化
ann_vol = rets.std() * np.sqrt(252)       # 日波动×√252 ≈ 年化波动

print("\n=== 关键统计（年化）===")
print(f"年化平均收益 ≈ {ann_return*100:.2f}%")
print(f"年化波动率   ≈ {ann_vol*100:.2f}%   （越大越‘颠簸’）")
print(f"夏普比率(粗略, 无风险利率设0) ≈ {ann_return/ann_vol:.2f}")

# ============================================================
# 4) 画图：上图价格、下图每日收益率
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

axes[0].plot(data.index, close, color='#1f77b4')
axes[0].set_title(f"{ticker} 腾讯 — 近一年收盘价", fontsize=13)
axes[0].set_ylabel("价格 (HKD)")
axes[0].grid(alpha=0.3)

axes[1].plot(data.index, data["Return"], color='#d62728', lw=0.8)
axes[1].axhline(0, color='gray', lw=0.8)
axes[1].set_title("每日收益率", fontsize=13)
axes[1].set_ylabel("日收益率")
axes[1].grid(alpha=0.3)

plt.tight_layout()
out = "01_hk_stock_returns.png"
plt.savefig(out, dpi=130)
print(f"\n图已保存: {out}")
