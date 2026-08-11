"""
项目4：过拟合演示 —— 为什么大部分回测都是假的（P1 核心课）
=========================================================
今天项目3的 MA 策略跑输买入持有。一个常见诱惑：
  "我多试几组 MA 参数，总有一组能跑赢吧？"

本脚本演示这个陷阱：
  1) 全样本扫几十组 (短期,长期) 窗口，挑出"最优"——看，像不像找到 alpha 了？
  2) 但这是"偷看全部数据后"挑的 = 数据窥探 (data snooping) = 过拟合
  3) 正确做法：训练集挑参数 → 在没见过的测试集检验 → alpha 通常消失

核心概念：过拟合 / 样本内(in-sample) / 样本外(out-of-sample) / 多重检验
运行：python3 04_overfitting_demo.py
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1) 取数据：腾讯 5 年日线（足够切训练/测试）
# ============================================================
ticker = "0700.HK"
df = yf.download(ticker, period="5y", auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
close = df["Close"].copy()

# ============================================================
# 工具函数：给定窗口，返回"防偷看未来"的策略日收益
# ============================================================
def strat_returns(price, short, long):
    pos = (price.rolling(short).mean() > price.rolling(long).mean()).astype(int)
    return (pos.shift(1) * price.pct_change()).dropna()

def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

# ============================================================
# 2) 陷阱演示：全样本扫参数，挑"最优"
# ============================================================
shorts = [5, 10, 15, 20, 25, 30, 40, 50]
longs  = [50, 80, 100, 120, 150, 200]
combos = [(s, l) for s in shorts for l in longs if s < l]

full_rows = []
for s, l in combos:
    r = strat_returns(close, s, l)
    full_rows.append({"short": s, "long": l, "sharpe_full": sharpe(r)})
full = pd.DataFrame(full_rows)

best_full = full.loc[full["sharpe_full"].idxmax()]
print("=" * 60)
print(f"【陷阱】在全样本上扫 {len(combos)} 组参数，挑夏普最高的：")
print(f"  最优参数 = MA{int(best_full.short)}/MA{int(best_full.long)}，"
      f"全样本夏普 = {best_full.sharpe_full:.2f}")
print("  👀 看起来不错？但这是‘偷看完整数据后’才选的 → 过拟合")
print("=" * 60)

# ============================================================
# 3) 正确做法：训练集选参 → 测试集验证
# ============================================================
n = len(close)
cut = int(n * 0.7)                       # 前 70% 训练，后 30% 测试
close_train, close_test = close.iloc[:cut], close.iloc[cut:]

tt_rows = []
for s, l in combos:
    rt = strat_returns(close_train, s, l)   # 训练集
    re = strat_returns(close_test,  s, l)   # 同参数套到测试集
    tt_rows.append({"short": s, "long": l, "train": sharpe(rt), "test": sharpe(re)})
tt = pd.DataFrame(tt_rows)

best_train = tt.loc[tt["train"].idxmax()]
bh_test_sharpe = sharpe(close_test.pct_change().dropna())   # 测试期买入持有夏普
corr_train_test = tt["train"].corr(tt["test"])

print("\n【正确做法】训练集挑最优，再看测试集表现：")
print(f"  训练集最优参数 = MA{int(best_train.short)}/MA{int(best_train.long)}")
print(f"  训练集夏普 = {best_train.train:.2f}   →  测试集夏普 = {best_train.test:.2f}")
print(f"  测试期买入持有夏普 = {bh_test_sharpe:.2f}")
print(f"  所有参数：训练夏普 vs 测试夏普 的相关系数 = {corr_train_test:.2f}"
      f"  （越接近0越说明=过拟合，训练好≠未来好）")
print("\n  💡 结论：训练集的‘最优’到了没见过的测试集通常缩水甚至反超买入持有。")
print("     这就是‘过拟合’——你拟合了过去的噪音，而不是规律。")

# ============================================================
# 4) 画图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# (a) 全样本夏普分布 —— 右尾被人捡走当"alpha"
ax = axes[0]
ax.hist(full["sharpe_full"], bins=15, color='#4C78A8', edgecolor='white')
ax.axvline(best_full["sharpe_full"], color='red', ls='--', lw=1.5,
           label=f'被挑走的"最优"={best_full.sharpe_full:.2f}')
ax.set_title("(a) 全样本夏普分布\n(有人专挑右尾说是alpha)", fontsize=11)
ax.set_xlabel("夏普"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (b) 训练夏普 vs 测试夏普 —— 过拟合时点散乱，不落在对角线
ax = axes[1]
ax.scatter(tt["train"], tt["test"], color='#9C9', alpha=0.7)
ax.scatter(best_train["train"], best_train["test"], color='red', s=120,
           zorder=5, label=f"训练最优→测试={best_train.test:.2f}")
lims = [min(tt["train"].min(), tt["test"].min())-0.1,
        max(tt["train"].max(), tt["test"].max())+0.1]
ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)  # 45度线：若不过拟合应落在线上
ax.set_title("(b) 训练夏普 vs 测试夏普\n(散乱=过拟合)", fontsize=11)
ax.set_xlabel("训练集夏普"); ax.set_ylabel("测试集夏普")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (c) 测试期净值：买入持有 vs 训练集选出的策略
ax = axes[2]
r_best_test = strat_returns(close_test, int(best_train.short), int(best_train.long))
bh = close_test.pct_change().dropna()
ax.plot((1 + r_best_test).cumprod().index, (1 + r_best_test).cumprod().values,
        color='#E45756', label=f"策略 MA{int(best_train.short)}/{int(best_train.long)}")
ax.plot((1 + bh).cumprod().index, (1 + bh).cumprod().values,
        color='#4C78A8', label="买入持有")
ax.axhline(1, color='gray', ls='--', lw=0.8)
ax.set_title(f"(c) 测试期(未见数据)净值\n训练最优策略 vs 买入持有", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
out = "04_overfitting_demo.png"
plt.savefig(out, dpi=130)
print(f"\n图已保存: {out}")
