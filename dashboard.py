import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title='Trader Sentiment Dashboard', layout='wide')

@st.cache_data
def load_and_process():
    sent = pd.read_csv('fear_greed_index.csv')
    trades = pd.read_csv('historical_data.csv')

    sent['date'] = pd.to_datetime(sent['date'])
    trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M')
    trades['date'] = trades['Timestamp IST'].dt.normalize()

    start = trades['date'].min()
    end = trades['date'].max()
    sent = sent[(sent['date'] >= start) & (sent['date'] <= end)].copy()

    trades['pnl'] = pd.to_numeric(trades['Closed PnL'], errors='coerce').fillna(0)
    trades['size_usd'] = pd.to_numeric(trades['Size USD'], errors='coerce').fillna(0)
    trades['fee'] = pd.to_numeric(trades['Fee'], errors='coerce').fillna(0)

    daily = trades.groupby(['Account', 'date']).agg(
        total_pnl=('pnl', 'sum'),
        trade_count=('pnl', 'count'),
        positive_trades=('pnl', lambda x: (x > 0).sum()),
        avg_trade_size=('size_usd', 'mean'),
        total_volume=('size_usd', 'sum'),
        buy_count=('Side', lambda x: (x == 'BUY').sum()),
        coins=('Coin', 'nunique'),
        total_fee=('fee', 'sum')
    ).reset_index()

    daily['win_rate'] = daily['positive_trades'] / daily['trade_count']
    daily['long_ratio'] = daily['buy_count'] / daily['trade_count']
    daily['net_pnl'] = daily['total_pnl'] - daily['total_fee']
    daily['is_profitable'] = (daily['net_pnl'] > 0).astype(int)

    df = daily.merge(sent[['date', 'value', 'classification']], on='date', how='inner')
    df.rename(columns={'value': 'fg_value', 'classification': 'fg_class'}, inplace=True)
    df['sentiment'] = df['fg_class'].apply(
        lambda x: 'Fear' if 'Fear' in x else ('Greed' if 'Greed' in x else 'Neutral')
    )

    return df, sent, trades

@st.cache_resource
def train_model(df):
    sent_map = {'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4}
    df = df.copy()
    df['fg_encoded'] = df['fg_class'].map(sent_map).fillna(2).astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek
    df = df.sort_values(['Account', 'date'])
    df['prev_pnl'] = df.groupby('Account')['net_pnl'].shift(1)
    df['prev_winrate'] = df.groupby('Account')['win_rate'].shift(1)
    df['prev_trades'] = df.groupby('Account')['trade_count'].shift(1)
    df['prev_volume'] = df.groupby('Account')['total_volume'].shift(1)
    df['rolling_pnl_3d'] = df.groupby('Account')['net_pnl'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['rolling_wr_3d'] = df.groupby('Account')['win_rate'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    model_df = df.dropna(subset=['prev_pnl', 'prev_winrate', 'prev_trades']).copy()

    features = ['fg_value', 'fg_encoded', 'trade_count', 'avg_trade_size', 'total_volume',
                'long_ratio', 'coins', 'day_of_week', 'prev_pnl', 'prev_winrate',
                'prev_trades', 'prev_volume', 'rolling_pnl_3d', 'rolling_wr_3d']

    X = model_df[features].fillna(0)
    y = model_df['is_profitable']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                    min_samples_leaf=10, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    imp = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=False)

    return gb, features, acc, cm, imp

df, sent, trades = load_and_process()
model, feature_names, accuracy, conf_matrix, importances = train_model(df)

st.title('Trader Performance vs Market Sentiment')

tab1, tab2, tab3, tab4 = st.tabs(['Overview', 'Analysis', 'Predict', 'Model Info'])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Trades', f'{len(trades):,}')
    col2.metric('Unique Traders', f'{df["Account"].nunique()}')
    col3.metric('Date Range', f'{df["date"].min().date()} → {df["date"].max().date()}')
    col4.metric('Model Accuracy', f'{accuracy:.1%}')

    st.subheader('Daily PnL Over Time')
    daily_agg = df.groupby('date').agg(total_pnl=('net_pnl', 'sum'), sentiment=('sentiment', 'first')).reset_index()
    colors_map = {'Fear': '#e74c3c', 'Neutral': '#95a5a6', 'Greed': '#27ae60'}
    fig, ax = plt.subplots(figsize=(14, 4))
    for s in ['Fear', 'Neutral', 'Greed']:
        subset = daily_agg[daily_agg['sentiment'] == s]
        ax.bar(subset['date'], subset['total_pnl'], color=colors_map[s], label=s, alpha=0.7, width=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_ylabel('Aggregate PnL (USD)')
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.subheader('Sentiment Distribution')
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    sent_counts = df['fg_class'].value_counts()
    ax2.barh(sent_counts.index, sent_counts.values, color='steelblue')
    ax2.set_xlabel('Count')
    st.pyplot(fig2)
    plt.close()

with tab2:
    st.subheader('Performance by Sentiment Regime')

    perf = df.groupby('sentiment').agg(
        mean_pnl=('net_pnl', 'mean'),
        median_pnl=('net_pnl', 'median'),
        mean_winrate=('win_rate', 'mean'),
        mean_trades=('trade_count', 'mean'),
        mean_size=('avg_trade_size', 'mean'),
        mean_long_ratio=('long_ratio', 'mean'),
        count=('net_pnl', 'count')
    ).round(4)
    st.dataframe(perf, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = df.groupby('sentiment')['net_pnl'].mean()
        ax.bar(vals.index, vals.values, color=[colors_map.get(x, '#333') for x in vals.index])
        ax.set_title('Mean PnL by Sentiment')
        ax.set_ylabel('USD')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        wr = df.groupby('sentiment')['win_rate'].mean()
        ax.bar(wr.index, wr.values, color=[colors_map.get(x, '#333') for x in wr.index])
        ax.set_title('Win Rate by Sentiment')
        ax.set_ylabel('Win Rate')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close()

    st.subheader('Behavioral Changes')
    col1, col2, col3 = st.columns(3)
    for c, col, title in zip([col1, col2, col3],
                              ['trade_count', 'avg_trade_size', 'long_ratio'],
                              ['Trade Frequency', 'Avg Trade Size', 'Long Ratio']):
        with c:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            vals = df.groupby('sentiment')[col].mean()
            ax.bar(vals.index, vals.values, color=[colors_map.get(x, '#333') for x in vals.index])
            ax.set_title(title)
            st.pyplot(fig)
            plt.close()

with tab3:
    st.subheader('Predict Trading Profitability')
    st.write('Enter the parameters below to predict if a trading session will be profitable.')

    col1, col2 = st.columns(2)
    with col1:
        fg_value = st.slider('Fear & Greed Index (0-100)', 0, 100, 50)
        fg_class = st.selectbox('Sentiment Class', ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])
        trade_count = st.number_input('Number of Trades', 1, 500, 50)
        avg_size = st.number_input('Avg Trade Size (USD)', 10.0, 100000.0, 1000.0)
        total_vol = st.number_input('Total Volume (USD)', 100.0, 5000000.0, 50000.0)

    with col2:
        long_ratio = st.slider('Long Ratio', 0.0, 1.0, 0.5)
        coins = st.number_input('Coins Traded', 1, 20, 2)
        dow = st.selectbox('Day of Week', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        prev_pnl = st.number_input('Previous Day PnL (USD)', -50000.0, 50000.0, 0.0)
        prev_wr = st.slider('Previous Day Win Rate', 0.0, 1.0, 0.4)

    prev_trades = st.number_input('Previous Day Trade Count', 1, 500, 30)
    prev_volume = st.number_input('Previous Day Volume (USD)', 100.0, 5000000.0, 40000.0)

    sent_map = {'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4}
    dow_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

    rolling_pnl = prev_pnl
    rolling_wr = prev_wr

    input_data = pd.DataFrame([[fg_value, sent_map[fg_class], trade_count, avg_size, total_vol,
                                 long_ratio, coins, dow_map[dow], prev_pnl, prev_wr,
                                 prev_trades, prev_volume, rolling_pnl, rolling_wr]],
                               columns=feature_names)

    if st.button('Predict', type='primary'):
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        if pred == 1:
            st.success(f'Prediction: **PROFITABLE** (confidence: {prob[1]:.1%})')
        else:
            st.error(f'Prediction: **NOT PROFITABLE** (confidence: {prob[0]:.1%})')

        st.write('Probability breakdown:')
        prob_df = pd.DataFrame({'Outcome': ['Not Profitable', 'Profitable'], 'Probability': prob})
        st.bar_chart(prob_df.set_index('Outcome'))

with tab4:
    st.subheader('Model Performance')
    st.metric('Test Accuracy', f'{accuracy:.1%}')

    col1, col2 = st.columns(2)
    with col1:
        st.write('**Confusion Matrix**')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Profitable', 'Profitable'],
                    yticklabels=['Not Profitable', 'Profitable'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.write('**Feature Importance (Top 10)**')
        fig, ax = plt.subplots(figsize=(5, 4))
        importances.head(10).sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Importance')
        st.pyplot(fig)
        plt.close()

    st.subheader('Strategy Recommendations')
    st.markdown("""
**1. Sentiment-Adjusted Position Sizing**
> During Fear days, reduce position sizes by 20-30% and focus on shorter-duration trades.
> Fear regimes tend to have higher volatility and wider drawdowns.

**2. Contrarian Long Bias in Fear**
> Increase long ratio selectively during Fear days for better average entry points.
> Only consistent winners should apply this aggressively.
""")
