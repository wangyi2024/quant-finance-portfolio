# 量化金融学习作品集 · Quant Finance Portfolio

WANG Yi · Lingnan University · yiwang25@ln.hk

从教育/翻译背景转型**量化金融 / 金融 NLP** 的学习与项目记录。
代码以 Python（pandas / numpy / matplotlib / yfinance）为主，后续加入 NLP/LLM（FinBERT、transformers）。

## 进度路线（6 个月）
- **P1 地基**：Python 金融数据处理、收益率/波动/夏普、相关性、时间序列基础
- **P2 护城河**：金融 NLP —— 财经文本情绪/事件抽取 → 信号
- **P3 落地**：ML for finance、回测、防过拟合/防数据泄露

## 项目
| # | 文件 | 学到的概念 |
|---|------|-----------|
| 01 | `projects/01_hk_stock_returns.py` | 取数据 · 收益率 · 年化收益/波动 · 夏普比率 |
| 02 | `projects/02_multi_stock_compare.py` | 归一化净值 · 风险-收益散点 · 相关性矩阵 · 分散风险 |

## 运行
```bash
pip install yfinance pandas numpy matplotlib
python3 projects/01_hk_stock_returns.py
python3 projects/02_multi_stock_compare.py
```

## 数据来源
雅虎财经（yfinance），仅供学习，不构成投资建议。
