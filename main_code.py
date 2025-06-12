# -*- coding: utf-8 -*-
"""
Gold–Silver–Cash 3-State Backtest + Optuna
 • 매일 1-Month T-Bill(^IRX) 실시간 복리 적용
 • 거래비용 0.2 % : 매도 (1-fee), 매수 (1+fee)
 • 리밸런싱 주기 = 거래일(period_days) 단위
 • look-ahead bias 제거, 훈련/검증(7:3) Split, Sortino 최대화
 • 오늘 추천 포지션(GOLD / SILVER / CASH) 출력
"""

# 필요한 패키지가 설치되어 있지 않다면 pip 를 통해 설치한다.
try:
    import FinanceDataReader as fdr
    import optuna
    import koreanize_matplotlib  # noqa: F401  # 폰트 설정만을 위함
except Exception:  # pragma: no cover - 런타임 설치
    import subprocess
    import sys
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "finance-datareader",
            "optuna",
            "koreanize-matplotlib",
        ]
    )
    import FinanceDataReader as fdr
    import optuna
    try:
        import koreanize_matplotlib  # noqa: F401
    except Exception:
        import warnings
        warnings.warn("koreanize_matplotlib import failed; proceeding without it")
        koreanize_matplotlib = None

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# ------------------------------- Config -------------------------------
CFG = dict(
    START="1990-01-01",
    END=None,                 # 오늘까지
    initial_cash=100.0,
    fee=0.002,                # 0.2 %
    n_trials=100,
    seed=42,
)

# ------------------------------- 0. 데이터 -------------------------------
def load_price(start, end):
    g = fdr.DataReader("GC=F", start, end)["Close"]
    s = fdr.DataReader("SI=F", start, end)["Close"]
    df = pd.concat([g, s], axis=1, join="inner").reset_index()
    df.columns = ["날짜", "금_가격", "은_가격"]
    df["금은비"] = df["금_가격"] / df["은_가격"]
    return df.sort_values("날짜").reset_index(drop=True)

def load_rf(start, end, dates):
    rf = fdr.DataReader("^IRX", start, end)["Close"] / 100  # 백분율 → 소수
    rf = rf.reindex(dates).fillna(method="ffill").fillna(method="bfill")
    return (rf / 252).to_numpy()  # 일 단리 ≈ 연리/252

price = load_price(CFG["START"], CFG["END"])
rf_daily = load_rf(CFG["START"], CFG["END"], price["날짜"])

# ------------------------------- 1. 보조 함수 -------------------------------
def trade_once(cash, g, s, pg, ps, tgt, fee):
    """보유 자산 전량 청산 후 목표자산으로 이동"""
    cash += g * pg * (1 - fee) + s * ps * (1 - fee)  # 매도 후 현금
    g = s = 0.0
    if tgt == "GOLD":
        g = cash / (pg * (1 + fee))
        cash -= g * pg * (1 + fee)
    elif tgt == "SILVER":
        s = cash / (ps * (1 + fee))
        cash -= s * ps * (1 + fee)
    return cash, g, s

def run_dyn(df, rf_arr, lookback, period, ql, qu, cash0, fee, start_idx=0):
    cash, g, s = cash0, 0.0, 0.0
    last_rb = start_idx          # 마지막 리밸런싱 index
    state = "CASH"
    vals = np.empty(len(df) - start_idx)
    for i in range(start_idx, len(df)):
        if state == "CASH":
            cash *= np.exp(rf_arr[i])       # 일 복리
        # 리밸런싱 조건
        if i - last_rb >= period:
            hist = df["금은비"].iloc[max(0, i-lookback):i]
            q1, q3 = hist.quantile([ql, qu])
            cur = df["금은비"].iat[i]
            tgt = "GOLD" if cur <= q1 else "SILVER" if cur >= q3 else "CASH"
            if tgt != state:
                cash, g, s = trade_once(cash, g, s,
                                        df["금_가격"].iat[i],
                                        df["은_가격"].iat[i],
                                        tgt, fee)
                state, last_rb = tgt, i
        vals[i - start_idx] = cash + g*df["금_가격"].iat[i] + s*df["은_가격"].iat[i]
    return vals

# ------------------------------- 2. 성과 지표 -------------------------------
def vec_stats(vals, rf_arr_slice):
    r = np.diff(vals) / vals[:-1]
    excess = r - rf_arr_slice[1:]          # 첫날 제외
    mean, std = excess.mean(), excess.std(ddof=1)
    neg_std = excess[excess < 0].std(ddof=1)
    sharpe = 0 if std == 0 else mean * np.sqrt(252) / std
    sortino = 0 if neg_std == 0 else mean * np.sqrt(252) / neg_std
    return sharpe, sortino

def cagr(vals, days):
    return (vals[-1]/vals[0])**(365.25/days)-1

# ------------------------------- 3. Train / Test Split -------------------
if __name__ == "__main__":
    split = int(len(price) * 0.7)
    train_df, test_df = price.iloc[:split], price.iloc[split:]
    rf_train, rf_test = rf_daily[:split], rf_daily[split:]

    # ------------------------------- 4. Optuna -------------------------------
    def objective(trial):
        p = dict(
            lookback=trial.suggest_int("lookback", 60, 512),
            period=trial.suggest_int("period", 20, 120),
            ql=trial.suggest_float("ql", 0.1, 0.4),
            qu=trial.suggest_float("qu", 0.6, 0.9),
        )
        vals = run_dyn(
            train_df, rf_train, **p, cash0=CFG["initial_cash"], fee=CFG["fee"]
        )
        _, sortino = vec_stats(vals, rf_train)
        return sortino

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CFG["seed"]),
    )
    study.optimize(objective, n_trials=CFG["n_trials"], show_progress_bar=False)
    best = study.best_params
    print("★ Best params (Train):", best, "/ Sortino =", round(study.best_value, 4))

    # ------------------------------- 5. Test 평가 ----------------------------
    vals_test = run_dyn(
        test_df,
        rf_test,
        **best,
        cash0=CFG["initial_cash"],
        fee=CFG["fee"],
    )
    days_test = (test_df["날짜"].iat[-1] - test_df["날짜"].iat[0]).days
    sh, so = vec_stats(vals_test, rf_test)
    print(
        f" Test → CAGR {cagr(vals_test, days_test):.4%} | Sharpe {sh:.3f} | Sortino {so:.3f}"
    )

    # ------------------------------- 6. 전체 구간 실행 ------------------------
    vals_all = run_dyn(
        price,
        rf_daily,
        **best,
        cash0=CFG["initial_cash"],
        fee=CFG["fee"],
    )
    gold_only = CFG["initial_cash"] * price["금_가격"] / price["금_가격"].iat[0]

    # ------------------------------- 7. 오늘 포지션 ---------------------------
    hist = price["금은비"].iloc[-best["lookback"] :]
    q1, q3 = hist.quantile([best["ql"], best["qu"]])
    cur = price["금은비"].iat[-1]
    signal = "GOLD" if cur <= q1 else "SILVER" if cur >= q3 else "CASH"
    print(f"\n▶ 오늘({price['날짜'].iat[-1].date()}) 추천 포지션 = {signal}")

    # ------------------------------- 8. 시각화 -------------------------------
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(price["날짜"], gold_only, label="Gold 100%", lw=1)
    plt.plot(price["날짜"], vals_all, label="Optimized", lw=1)
    plt.title(f"Equity Curve  • Params {best}")
    plt.legend(); plt.grid(alpha=.3)

    plt.subplot(2, 1, 2)
    dd_g = gold_only / gold_only.cummax() - 1
    dd_o = vals_all / vals_all.cummax() - 1
    plt.plot(price["날짜"], dd_g, label="Gold DD")
    plt.plot(price["날짜"], dd_o, label="Opt DD")
    plt.title("Drawdown"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.show()
