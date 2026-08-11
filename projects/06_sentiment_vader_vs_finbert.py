"""
项目6：金融情感分析 3 种方法横评（P2 护城河：理解工具的边界）
=========================================================
对比同一批腾讯新闻标题在 3 种方法下的表现：
  A) VADER   —— 通用词典法，不懂金融（项目5用的）
  B) FinBERT —— 金融微调BERT，懂金融词，但只测"语气"非"投资含义"
  C) LLM推理 —— 带相关性判断+理由（最强，需API或人工，这里给参考值）

关键教训：没有"开箱即用就准"的情感模型——这正是基金要自己建/微调、
也正是你的"判断力+LLM工程"有价值的地方。
运行：python3 06_sentiment_vader_vs_finbert.py  （首次自动下载FinBERT~440MB）
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

TICKER = "0700.HK"

# ---- 取标题 ----
news = yf.Ticker(TICKER).news
titles = [it["content"].get("title", "") for it in news]

# ---- A) VADER ----
vader = SentimentIntensityAnalyzer()
vader_scores = [vader.polarity_scores(t)["compound"] for t in titles]

# ---- B) FinBERT（金融微调）----
print("加载 FinBERT（首次需下载模型，请稍候）...")
finb = pipeline("sentiment-analysis", model="ProsusAI/finbert")
finb_signed, finb_label = [], []
for t in titles:
    res = {d["label"]: d["score"] for d in finb(t, top_k=3)}
    finb_signed.append(res.get("positive", 0) - res.get("negative", 0))  # -1..1
    finb_label.append(max(res, key=res.get))

# ---- 对比表 ----
df = pd.DataFrame({
    "标题": [t[:50] for t in titles],
    "VADER": vader_scores,
    "FinBERT": finb_signed,
    "FinBERT标签": finb_label,
})
print("\n=== VADER vs FinBERT 对比 ===")
print(df.round(2).to_string(index=False))

# ---- 画图：每条标题两个柱（VADER vs FinBERT）----
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(titles))
w = 0.4
ax.bar(x - w/2, vader_scores, w, label="VADER(词典法)", color="#9C9")
ax.bar(x + w/2, finb_signed, w, label="FinBERT(金融微调)", color="#4C78A8")
ax.axhline(0, color='gray', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"#{i+1}" for i in x])
ax.set_title("VADER vs FinBERT —— 同一批金融标题的情感打分对比", fontsize=12)
ax.set_ylabel("情感分 (-1 最负 ~ +1 最正)"); ax.legend(); ax.grid(alpha=0.3, axis='y')
# 把标题列表放图下方说明
txt = "\n".join([f"#{i+1}: {t[:42]}" for i, t in enumerate(titles)])
fig.text(0.5, -0.18, txt, ha='center', va='top', fontsize=7.5, color='#555')
plt.tight_layout()
out = "06_sentiment_vader_vs_finbert.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\n图已保存: {out}")

print("\n💡 教训：FinBERT 比VADER懂金融词，但它测的是【语气】不是【投资含义】")
print("   例：'undervalued'(被低估=利好) 两方法都可能判负，因为句子语气谨慎。")
print("   → 真正用于交易，需要LLM带推理：判断相关性+方向+理由。这就是你的护城河。")
