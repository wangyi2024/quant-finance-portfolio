"""
项目5：金融新闻 → 情感信号 → 收益（P2 护城河：最小可用管线 v1）
=========================================================
这是"另类数据/NLP 量化"的核心骨架，4 步：
  1) 取文本  —— 腾讯(0700.HK)近期财经新闻标题（yfinance 免费tier，仅~10条）
  2) 抽情感  —— v1 用 VADER 词典法（轻量、免key）；真正该用 FinBERT/LLM
  3) 对齐    —— 把"某天的新闻情感"对齐到"次日的股票收益"
  4) 检验    —— 情感能否预测收益？（相关系数 / 信息系数 IC）

⚠️ 诚信声明：免费新闻只有~10条、覆盖2周，样本太小，结论无统计意义。
   本脚本演示【方法】，不是【证据】。真信号需要数千条新闻×数年。
运行：python3 05_news_sentiment_signal.py
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

TICKER = "0700.HK"

# ============================================================
# 1) 取文本：新闻标题 + 日期
# ============================================================
news = yf.Ticker(TICKER).news
rows = []
for it in news:
    c = it.get("content", it)
    title = c.get("title", "")
    date = pd.to_datetime(c.get("pubDate")).date()      # 只取日期
    rows.append({"date": date, "title": title})
news_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
print(f"=== 抓到 {len(news_df)} 条新闻 ===")

# ============================================================
# 2) 抽情感：VADER compound (-1 最负, +1 最正)
#    词典法：按词查表加减分。快、免key，但不懂金融语境。
# ============================================================
analyzer = SentimentIntensityAnalyzer()
news_df["sentiment"] = news_df["title"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
print("\n=== 每条标题 + VADER 情感 ===")
for _, r in news_df.iterrows():
    print(f"  {r['sentiment']:+.2f}  {r['date']}  {r['title'][:60]}")

# 按天聚合：同一天多条新闻取平均
daily = news_df.groupby("date")["sentiment"].mean().rename("sentiment")

# ============================================================
# 3) 取收益，并把"当天情感"对齐到"次日收益"
# ============================================================
df = yf.download(TICKER, start="2026-07-20", end="2026-08-12", auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df["Return"] = df["Close"].pct_change()
rets = df["Return"].dropna()

def next_day_return(d):
    """新闻日期 d 之后第一个交易日的收益（防偷看未来：只用当时已知信息）"""
    fut = rets[rets.index > pd.Timestamp(d)]
    return fut.iloc[0] if len(fut) else np.nan

aligned = pd.DataFrame({"sentiment": daily})
aligned["next_ret"] = [next_day_return(d) for d in aligned.index]
aligned = aligned.dropna()
print("\n=== 对齐后的样本（情感 vs 次日收益）===")
print(aligned.round(4))

# ============================================================
# 4) 检验：相关系数（金融里叫信息系数 IC）
# ============================================================
corr = aligned["sentiment"].corr(aligned["next_ret"])
print("\n" + "=" * 60)
print(f"情感 vs 次日收益 的相关系数(IC) = {corr:+.3f}")
print(f"样本量 n = {len(aligned)}（个交易日）")
print("=" * 60)
print("⚠️ n 太小，这个相关系数【没有统计意义】，只演示方法。")
print("   真要验证信号，需要 n=数百~数千，且要做样本外检验（见项目4）。")

# ============================================================
# 5) 画图：情感条 + 散点
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axes[0]
colors = ['#2ca02c' if v >= 0 else '#d62728' for v in aligned["sentiment"]]
ax.bar(range(len(aligned)), aligned["sentiment"], color=colors)
ax.axhline(0, color='gray', lw=0.8)
ax.set_xticks(range(len(aligned)))
ax.set_xticklabels([d.strftime('%m-%d') for d in aligned.index], rotation=45, fontsize=8)
ax.set_title("(a) 每日新闻情感均值", fontsize=12)
ax.set_ylabel("VADER 情感"); ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(aligned["sentiment"], aligned["next_ret"], s=90, color='#4C78A8')
# 拟合线（仅示意，n太小不可信）
if len(aligned) >= 3:
    m, b = np.polyfit(aligned["sentiment"], aligned["next_ret"], 1)
    xs = np.linspace(aligned["sentiment"].min(), aligned["sentiment"].max(), 50)
    ax.plot(xs, m*xs + b, '--', color='red', alpha=0.6, label=f"IC={corr:+.2f}")
ax.axhline(0, color='gray', lw=0.8)
ax.set_title("(b) 情感 vs 次日收益\n[n=5 太小·不可信]", fontsize=12)
ax.set_xlabel("当日新闻情感"); ax.set_ylabel("次日收益率")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
out = "05_news_sentiment_signal.png"
plt.savefig(out, dpi=130)
print(f"\n图已保存: {out}")
