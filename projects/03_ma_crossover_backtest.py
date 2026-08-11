"""
项目3：移动平均线交叉策略 + 最简回测（P1 地基 → 量化策略入门）
=========================================================
核心概念（重点学）：
  1) 移动平均 MA：rolling(window).mean() —— 把价格"平滑"，看趋势
  2) 金叉/死叉：短期MA 上穿/下穿 长期MA → 买/卖信号
  3) 回测 Backtest：把信号套到历史数据上，模拟"如果当时照做，赚没赚"
  4) ⚠️ 偷看未来(look-ahead bias)：必须用"昨天收盘的信号"决定"今天持仓"
  5) 交易成本 & 最大回撤：决定策略到底能不能赚钱

策略：腾讯(0700.HK)，MA50 上穿 MA200 持仓(做多)，否则空仓。
运行：python3 03_ma_crossover_backtest.py
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1) 取数据：3 年日线（MA200 需要足够长度）
# ============================================================
ticker = "0700.HK"
df = yf.download(ticker, period="3y", auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
close = df["Close"].copy()

# ============================================================
# 2) 移动平均线：rolling(window).mean()
#    MA_t = 最近 window 天的平均收盘价 → 平滑掉每日噪音，看趋势
# ============================================================
df["MA50"]  = close.rolling(50).mean()
df["MA200"] = close.rolling(200).mean()

# 信号：MA50 > MA200 时持仓(1)，否则空仓(0)
df["Position"] = (df["MA50"] > df["MA200"]).astype(int)

# 金叉(0→1)与死叉(1→0)的位置，用于画图标记
golden = df["Position"].diff() == 1
death  = df["Position"].diff() == -1

# ============================================================
# 3) 回测 —— 最关键的一行：shift(1) 防止"偷看未来"
#    逻辑：用【第t天收盘】算出的信号，决定【第t+1天】的持仓收益
#    所以 Position 要往后挪一天(shift)再去乘当天的收益
# ============================================================
df["Return"] = close.pct_change()
df["Strategy"] = df["Position"].shift(1) * df["Return"]   # ⭐ 关键
df["BuyHold"]  = df["Return"]                              # 对照：一直持有

# 交易成本：每次换仓扣 0.1%（一来一回约 0.2%）
trade_cost = 0.001
trades = df["Position"].diff().abs().fillna(0)             # 1=发生换仓
df["Strategy_net"] = df["Strategy"] - trades * trade_cost

bt = df.dropna(subset=["Strategy"]).copy()                 # 去掉 MA 还没算出来的前期

# ============================================================
# 4) 算指标：年化收益/波动/夏普/最大回撤
# ============================================================
def metrics(r):
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()                # 最大回撤：从峰值最多跌多少
    return pd.Series({"年化收益": ann_ret, "年化波动": ann_vol,
                      "夏普": sharpe, "最大回撤": max_dd,
                      "换仓次数": int((trades.reindex(r.index) > 0).sum())})

table = pd.DataFrame([
    metrics(bt["BuyHold"]),          metrics(bt["Strategy"]),
    metrics(bt["Strategy_net"]),
], index=["买入持有", "MA交叉(扣成本前)", "MA交叉(扣成本后)"])
print("=== 回测对比（腾讯 0700.HK, 近3年）===")
print(table.round(3))

# ============================================================
# 5) 画图：价格+均线+交叉点 / 净值曲线 / 回撤
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# (a) 价格 + MA + 金叉死叉标记
ax = axes[0]
ax.plot(close.index, close, color='#999', label='收盘价', lw=1)
ax.plot(df["MA50"].index, df["MA50"], label='MA50', lw=1.3)
ax.plot(df["MA200"].index, df["MA200"], label='MA200', lw=1.3)
ax.scatter(close.index[golden], close[golden], marker='^', color='red', s=60, label='金叉(买)', zorder=5)
ax.scatter(close.index[death], close[death], marker='v', color='green', s=60, label='死叉(卖)', zorder=5)
ax.set_title("(a) 腾讯 收盘价 + MA50/MA200 + 交叉信号", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (b) 净值曲线：三条都起点=100
nav = pd.DataFrame({"买入持有": bt["BuyHold"], "MA交叉(扣成本前)": bt["Strategy"],
                    "MA交叉(扣成本后)": bt["Strategy_net"]})
nav = (1 + nav).cumprod() * 100
ax = axes[1]
for c in nav.columns:
    ax.plot(nav.index, nav[c], label=c, lw=1.5)
ax.axhline(100, color='gray', ls='--', lw=0.8)
ax.set_title("(b) 净值曲线（起点=100，越高越好）", fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (c) 策略回撤
cum = (1 + bt["Strategy_net"]).cumprod()
dd = cum / cum.cummax() - 1
ax = axes[2]
ax.fill_between(dd.index, dd.values, 0, color='#d62728', alpha=0.4)
ax.set_title("(c) 策略回撤（谷底=最大回撤，越浅越好）", fontsize=12)
ax.set_ylabel("回撤"); ax.grid(alpha=0.3)

plt.tight_layout()
out = "03_ma_crossover_backtest.png"
plt.savefig(out, dpi=130)
print(f"\n图已保存: {out}")
