# Trader Performance vs Market Sentiment

Exploring how Bitcoin market sentiment (Fear & Greed Index) influences trader behavior and profitability on Hyperliquid. Includes data analysis, predictive modeling (86.8% accuracy), and an interactive Streamlit dashboard.

---

## What This Project Does

- **Part A:** Cleans and aligns two datasets (sentiment + trades), builds daily trading metrics per account
- **Part B:** Analyzes performance differences across Fear/Greed regimes, identifies behavioral changes, segments traders into archetypes
- **Part C:** Proposes actionable trading strategies based on findings
- **Bonus:** Gradient Boosting model predicting daily profitability + KMeans clustering + Streamlit dashboard

---

## Datasets

| Dataset | File | Rows | Description |
|---------|------|------|-------------|
| Bitcoin Fear & Greed Index | `fear_greed_index.csv` | ~2,600 | Daily sentiment score (0-100) and label (Extreme Fear → Extreme Greed) |
| Hyperliquid Trader Data | `historical_data.csv` | ~211,000 | Trade executions with account, coin, side, size, PnL, timestamps |

---

## Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter streamlit
```

### 2. Run the notebook

```bash
jupyter notebook analysis.ipynb
```

Run all cells top-to-bottom. Charts save automatically as PNGs.

### 3. Launch the dashboard

```bash
streamlit run dashboard.py
```

Opens a browser with 4 interactive tabs: Overview, Analysis, Predict, and Model Info.

---

## Project Structure

```
/
├── analysis.ipynb              ← Full analysis notebook (Parts A, B, C + Bonus)
├── dashboard.py                ← Streamlit interactive dashboard
├── fear_greed_index.csv        ← Sentiment dataset
├── historical_data.csv         ← Trader dataset
├── README.md                   ← This file
├── chart_pnl_sentiment.png     ← PnL by sentiment regime
├── chart_behavior_sentiment.png← Behavioral shifts chart
├── chart_segments.png          ← Trader segment comparison
├── chart_scatter_sentiment_pnl.png ← Sentiment vs PnL scatter
├── chart_model_results.png     ← Confusion matrix + feature importance
└── chart_clusters.png          ← Trader archetypes (KMeans)
```

---

## Key Findings

| Metric | Fear Days | Greed Days |
|--------|-----------|------------|
| Avg PnL | Varies by segment | Varies by segment |
| Win Rate | Lower on average | Higher on average |
| Trade Frequency | Adjusted | Higher activity |
| Long Ratio | Shifts toward sells | More long-biased |

**Top Insights:**
1. Sentiment regime meaningfully correlates with trading outcomes
2. Traders adjust position sizing and direction bias based on Fear/Greed
3. Consistent winners maintain stable performance regardless of sentiment

---

## Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | 85.3% |
| **Gradient Boosting** | **86.8%** |
| Cross-validation (5-fold) | 84.5% ± 2.5% |

Features used: sentiment value, trade count, position size, volume, long ratio, coins traded, day of week, and lagged behavioral metrics (previous day PnL, win rate, trade count).

---

## Strategy Recommendations

1. **Sentiment-Adjusted Sizing** — Reduce positions 20-30% during Fear days to limit drawdowns
2. **Contrarian Longs in Fear** — Selectively increase long exposure during Fear for better entries (only for disciplined, consistent traders)

---

## Dashboard Guide

The Streamlit app (`dashboard.py`) has 4 tabs:

- **Overview** — Key stats, daily PnL timeline colored by sentiment, sentiment distribution
- **Analysis** — Performance tables and charts broken down by Fear/Neutral/Greed
- **Predict** — Enter trading parameters (sentiment, trade count, size, etc.) and get a real-time profitability prediction from the trained model
- **Model Info** — Confusion matrix, feature importance ranking, and strategy recommendations
