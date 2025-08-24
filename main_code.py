# 0) 환경 준비 -----------------------------------------------------------------
!pip install -q yfinance koreanize-matplotlib

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib  # ✅ 한글 폰트 자동 설정

from itertools import product
from typing import List, Dict, Any, Optional
from IPython.display import display

# 보기 좋은 숫자 포맷
pd.options.display.float_format = lambda x: f"{x:,.6f}"

# 재현성을 위한 시드 고정
np.random.seed(42)

# -----------------------------------------------------------------------------
# 1) 유틸리티 함수
# -----------------------------------------------------------------------------
def _tz_naive(index_like) -> pd.DatetimeIndex:
    """DatetimeIndex의 tz 정보를 제거(naive). tz-naive면 그대로 반환."""
    try:
        return index_like.tz_localize(None)
    except (TypeError, AttributeError):
        # already tz-naive or not a DatetimeIndex; try to coerce
        return pd.to_datetime(index_like)


def clean_prices(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    가격 데이터의 결측치 처리 (날짜 무결성 강화):
    - 상장 전/후 '끝단'은 NaN으로 남김 (bfill 금지)
    - 시계열 '내부'만 시간보간 허용(limit_area='inside'), 이후 ffill
    - 결과: get_common_start_index()가 실제 공통 상장일을 잡도록 보장
    """
    df_copy = df.copy()
    if "날짜" not in df_copy.columns:
        raise ValueError("'날짜' 컬럼이 필요합니다.")

    # 날짜 정렬 + 중복 제거 + 인덱스 설정
    df_copy = df_copy.sort_values("날짜").drop_duplicates(subset=["날짜"]).reset_index(drop=True)
    df_copy = df_copy.set_index("날짜")

    # 필수 컬럼 확인
    miss_cols = [c for c in columns if c not in df_copy.columns]
    if miss_cols:
        raise ValueError(f"가격 컬럼 누락: {miss_cols}")

    # 0 → NaN
    df_copy[columns] = df_copy[columns].replace(0, np.nan)

    # 🔒 내삽만 허용: 양 끝단(리드/테일)은 NaN 유지 → 상장 전/후 구간을 가짜로 메우지 않음
    df_copy[columns] = df_copy[columns].interpolate(method="time", limit_area="inside")

    # 🔒 과거→미래 방향으로만 채움(리드 NaN은 그대로 유지, bfill 사용 금지)
    df_copy[columns] = df_copy[columns].ffill()

    return df_copy.reset_index()


def max_drawdown(cum_series: pd.Series) -> float:
    """누적 수익률(또는 포트 가치) 시리즈로부터 최대 낙폭(MDD, 음수)을 계산."""
    if cum_series is None or len(cum_series) == 0:
        return np.nan
    s = cum_series.astype(float)
    peak = s.cummax()
    dd = (s - peak) / peak
    return float(dd.min()) if len(dd) else np.nan


def get_common_start_index(df: pd.DataFrame, price_cols: List[str]) -> int:
    """모든 자산 가격 데이터가 유효하게 시작되는 첫 인덱스를 찾음."""
    if any(c not in df.columns for c in price_cols):
        raise ValueError(f"가격 컬럼 누락: {price_cols}")
    mask = np.logical_and.reduce([df[c].notna() for c in price_cols])
    valid_idx = df.index[mask]
    if len(valid_idx) == 0:
        raise ValueError("모든 자산 가격이 동시에 존재하는 시점이 없습니다.")
    return int(valid_idx[0])


def calculate_performance_metrics(
    df: pd.DataFrame,
    strategy_cols: List[str],
    risk_free_rate_annual: float = 0.02
) -> pd.DataFrame:
    """
    전략별 성과 지표(CAGR, 샤프, 소르티노, MDD)를 계산.
    - 입력: 포트 가치(누적) 시리즈
    - 위험무위험 수익률은 연율(annual) 입력 → 일간 환산 후 사용
    """
    out = {}
    rf_daily = risk_free_rate_annual / 252.0

    for col in strategy_cols:
        series = df[col].dropna().astype(float)
        if len(series) < 2:
            out[col] = {"CAGR": np.nan, "샤프지수": np.nan, "소르티노지수": np.nan, "MaxDrawdown": np.nan}
            continue

        start_val, end_val = float(series.iloc[0]), float(series.iloc[-1])
        start_date, end_date = df.loc[series.index[0], "날짜"], df.loc[series.index[-1], "날짜"]
        years = max((end_date - start_date).days / 365.25, 1e-9)
        cagr = (end_val / max(start_val, 1e-12)) ** (1 / years) - 1 if start_val > 0 else np.nan

        ret = series.pct_change().dropna()
        mdd = max_drawdown(series)

        if len(ret) < 2 or ret.std() == 0:
            out[col] = {"CAGR": cagr, "샤프지수": np.nan, "소르티노지수": np.nan, "MaxDrawdown": mdd}
            continue

        excess = ret - rf_daily
        sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else np.nan
        downside = excess[excess < 0]
        sortino = (excess.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else np.nan

        out[col] = {"CAGR": cagr, "샤프지수": sharpe, "소르티노지수": sortino, "MaxDrawdown": mdd}

    return pd.DataFrame(out).T


def estimate_rf_from_shy(df: pd.DataFrame, col: str = "SHY_가격") -> Optional[float]:
    """
    SHY 일일 수익률 평균 기반 연 환산 무위험 수익률 추정 (선택).
    - 데이터 문제 시 None 반환 → 기본 RF 사용.
    """
    if col not in df.columns:
        return None
    daily = df[col].pct_change().dropna()
    if len(daily) == 0:
        return None
    rf_annual = float(daily.mean() * 252)
    # 비정상적으로 큰 값 방지(클리핑)
    return min(max(rf_annual, -0.02), 0.06)


# -----------------------------------------------------------------------------
# 2) 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
def load_and_preprocess_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """YFinance에서 데이터를 로드하고 필요한 지표를 계산하여 전처리."""
    # yfinance (2025-07 기준) 기본 auto_adjust=True지만 명시하여 일관성 유지
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # Close 레벨만 사용 (멀티인덱스 → 단일 DataFrame)
    close = raw["Close"] if "Close" in raw else raw
    # 인덱스 tz-naive로
    dates = _tz_naive(close.index)

    # 컬럼/순서 안전 접근
    def _safe_col(cframe, tk):
        if tk not in cframe.columns:
            raise KeyError(f"티커 '{tk}'의 종가가 다운로드되지 않았습니다.")
        return cframe[tk]

    df = pd.DataFrame({
        "날짜": dates,
        "금_가격": _safe_col(close, tickers[0]),
        "은_가격": _safe_col(close, tickers[1]),
        "SPY_가격": _safe_col(close, tickers[2]),
        "SHY_가격": _safe_col(close, tickers[3]),
    }).reset_index(drop=True)

    price_cols = ["금_가격", "은_가격", "SPY_가격", "SHY_가격"]
    df = clean_prices(df, price_cols)

    # 지표 사전 계산(효율성 ↑, 분할 후 재사용)
    df["금은비"] = df["금_가격"] / df["은_가격"]
    df["금_200_EMA"] = df["금_가격"].ewm(span=200, adjust=False).mean()
    df["SPY_200_MA"] = df["SPY_가격"].rolling(window=200, min_periods=200).mean()

    # 동적 전략용 보조지표
    df["금은비_EWMA_20"] = df["금은비"].ewm(span=20, adjust=False).mean()
    df["ratio_vol_30"] = df["금은비_EWMA_20"].pct_change().rolling(window=30, min_periods=5).std()

    return df


# -----------------------------------------------------------------------------
# 3) 기본 전략
# -----------------------------------------------------------------------------
def run_strategy_from(df: pd.DataFrame, start_idx: int, strategy_func, *args, **kwargs) -> pd.Series:
    """전략 함수를 데이터의 특정 시작점부터 실행. 앞부분 NaN 패딩."""
    df_slice = df.iloc[start_idx:].reset_index(drop=True)
    series = strategy_func(df_slice, *args, **kwargs)
    # 전체 길이에 맞게 앞부분 NaN 패딩
    return pd.Series([np.nan] * start_idx + series.tolist(), name=series.name)


# 전략 ① 금만 보유 (Buy & Hold)
def strat_buy_hold_gold(df: pd.DataFrame, initial_cash: float, **kwargs) -> pd.Series:
    units = initial_cash / df.loc[0, "금_가격"]
    portfolio = units * df["금_가격"]
    return pd.Series(portfolio, name="① 금만 보유 (Buy&Hold)")


# 전략 ② 6개월마다 50:50 리밸런싱 (초기 체결 수수료 옵션)
def strat_periodic_rebalance(
    df: pd.DataFrame,
    initial_cash: float,
    transaction_cost: float,
    period_days: int = 182,
    apply_initial_fee: bool = False,
    **kwargs
) -> pd.Series:
    total = initial_cash * (1 - transaction_cost) if apply_initial_fee else initial_cash
    gold_units = (total / 2.0) / df.loc[0, "금_가격"]
    silver_units = (total / 2.0) / df.loc[0, "은_가격"]
    last_reb_date = df.loc[0, "날짜"]
    values = [total]

    for i in range(1, len(df)):
        row = df.loc[i]
        total = gold_units * row["금_가격"] + silver_units * row["은_가격"]
        if (row["날짜"] - last_reb_date).days >= period_days:
            cost = total * transaction_cost
            total_after = total - cost
            gold_units = (total_after / 2.0) / row["금_가격"]
            silver_units = (total_after / 2.0) / row["은_가격"]
            last_reb_date = row["날짜"]
            total = total_after
        values.append(total)

    return pd.Series(values, name="② 반기 50:50 리밸런싱")


# 전략 ③ 금 200일 EMA 추세추종 (※ 수수료 영구 반영 패치는 다음 버전에서 별도 제공 예정)
def strat_gold_ema_trend(df: pd.DataFrame, initial_cash: float, transaction_cost: float, **kwargs) -> pd.Series:
    gold_u, cash = 0.0, initial_cash
    values = [initial_cash]

    for i in range(1, len(df)):
        price = df.loc[i, "금_가격"]
        ema = df.loc[i, "금_200_EMA"]
        current = gold_u * price if gold_u > 0 else cash

        # 매수
        if (price > ema) and (gold_u == 0):
            buy_u = cash / price
            fee = buy_u * price * transaction_cost
            gold_u, cash = buy_u, 0.0
            current = gold_u * price - fee
        # 매도
        elif (price <= ema) and (gold_u > 0):
            sell_val = gold_u * price
            fee = sell_val * transaction_cost
            cash, gold_u = sell_val - fee, 0.0
            current = cash

        values.append(current)

    return pd.Series(values, name="③ 금 200일 EMA 추세추종")


# -----------------------------------------------------------------------------
# 4) 동적 리밸런싱 V2/V3 전략
# -----------------------------------------------------------------------------
def _dynamic_thresholds(
    df: pd.DataFrame, i: int, lookback_periods: List[int], qtiles: List[float]
) -> Optional[Dict[str, float]]:
    """동적 임계치(q1, q3) 계산: 최근 구간의 금은비_EWMA 분포 기반, 가중치는 기간 역수."""
    q_lows, q_highs, weights = [], [], []
    for lp in lookback_periods:
        start = max(0, i - lp)
        past = df.loc[start: i - 1, "금은비_EWMA_20"]
        if len(past) >= 20:
            q_lows.append(past.quantile(qtiles[0]))
            q_highs.append(past.quantile(qtiles[1]))
            weights.append(1.0 / lp)

    if not weights:
        return None

    w = np.array(weights) / np.sum(weights)
    q1 = float(np.sum(np.array(q_lows) * w))
    q3 = float(np.sum(np.array(q_highs) * w))
    return {"q1": q1, "q3": q3}


# 전략 ④ 금은비 동적 리밸런싱 v2
def strat_dynamic_rebalance_v2(
    df: pd.DataFrame, initial_cash: float, transaction_cost: float,
    lookback_periods: List[int], quantile_thresholds: List[float],
    base_hysteresis: float, min_holding_months: int, check_months: int, **kwargs
) -> pd.Series:
    gold_u = (initial_cash / 2.0) / df.loc[0, "금_가격"]
    silver_u = (initial_cash / 2.0) / df.loc[0, "은_가격"]
    last_sw = df.loc[0, "날짜"]
    next_check = last_sw + pd.DateOffset(months=check_months)
    pos = "balanced"
    values = [initial_cash]

    for i in range(1, len(df)):
        row = df.loc[i]
        total = gold_u * row["금_가격"] + silver_u * row["은_가격"]

        th = _dynamic_thresholds(df, i, lookback_periods, quantile_thresholds)
        if th is None:
            values.append(total)
            continue

        vol = row.get("ratio_vol_30", 0.01)
        hyst = base_hysteresis * (1 + float(vol) * 10.0)
        ratio = row["금은비_EWMA_20"]

        signal_gold = (ratio <= th["q1"] * (1 - hyst))
        signal_silver = (ratio >= th["q3"] * (1 + hyst))

        now = row["날짜"]
        months_since = (now - last_sw).days / 30.44

        if (now >= next_check) and (months_since >= min_holding_months):
            new_pos = pos
            if signal_silver and pos != "silver":
                new_pos = "silver"
            elif signal_gold and pos != "gold":
                new_pos = "gold"

            if new_pos != pos:
                fee = total * transaction_cost
                total_after = total - fee
                if new_pos == "silver":
                    gold_u, silver_u = 0.0, total_after / row["은_가격"]
                else:
                    silver_u, gold_u = 0.0, total_after / row["금_가격"]
                total = total_after
                pos = new_pos
                last_sw = now
                next_check = last_sw + pd.DateOffset(months=check_months)

        values.append(total)

    return pd.Series(values, name="④ 금은비 동적 리밸런싱 v2")


# 전략 ⑤ 금은비 동적 리스크관리 v3 (SPY 200일선 기반)
def strat_dynamic_rebalance_v3_risk(
    df: pd.DataFrame, initial_cash: float, transaction_cost: float,
    lookback_periods: List[int], quantile_thresholds: List[float],
    base_hysteresis: float, min_holding_months: int, check_months: int, **kwargs
) -> pd.Series:
    gold_u = silver_u = 0.0
    shy_u = initial_cash / df.loc[0, "SHY_가격"]
    pos = "shy"
    last_sw = df.loc[0, "날짜"]
    next_check = last_sw + pd.DateOffset(months=check_months)
    values = [initial_cash]

    for i in range(1, len(df)):
        row = df.loc[i]
        total = gold_u * row["금_가격"] + silver_u * row["은_가격"] + shy_u * row["SHY_가격"]

        risk_on = (row["SPY_가격"] > row["SPY_200_MA"]) if pd.notna(row["SPY_200_MA"]) else False
        target = pos

        if risk_on:
            th = _dynamic_thresholds(df, i, lookback_periods, quantile_thresholds)
            if th:
                vol = row.get("ratio_vol_30", 0.01)
                hyst = base_hysteresis * (1 + float(vol) * 10.0)
                ratio = row["금은비_EWMA_20"]

                if ratio >= th["q3"] * (1 + hyst):
                    target = "silver"
                elif ratio <= th["q1"] * (1 - hyst):
                    target = "gold"
                else:
                    target = pos if pos in ["gold", "silver"] else "shy"
        else:
            target = "shy"

        now = row["날짜"]
        months_since = (now - last_sw).days / 30.44

        if (now >= next_check) and (months_since >= min_holding_months) and (target != pos):
            fee = total * transaction_cost
            total_after = total - fee
            gold_u = silver_u = shy_u = 0.0

            if target == "silver":
                silver_u = total_after / row["은_가격"]
            elif target == "gold":
                gold_u = total_after / row["금_가격"]
            else:
                shy_u = total_after / row["SHY_가격"]

            pos = target
            last_sw = now
            next_check = last_sw + pd.DateOffset(months=check_months)
            total = total_after

        values.append(total)

    return pd.Series(values, name="⑤ 금은비 동적 리스크관리 v3")


def run_dynamic_wrapper(
    df: pd.DataFrame, initial_cash: float, transaction_cost: float,
    best_params: Dict[str, Any], strategy_func
) -> pd.Series:
    """동적 리밸런싱 전략 실행 래퍼(사전 계산 지표 활용)."""
    return strategy_func(
        df, initial_cash, transaction_cost,
        **{k: best_params[k] for k in ["lookback_periods", "quantile_thresholds", "base_hysteresis", "min_holding_months", "check_months"]}
    )

# -----------------------------------------------------------------------------
# 5) 파라미터 탐색 (검증 세트에서) — RF 일관성 적용
# -----------------------------------------------------------------------------
def find_best_params_v2(
    df_val: pd.DataFrame,
    initial_cash: float,
    transaction_cost: float,
    risk_free_rate_annual: float = 0.02
) -> Dict[str, Any]:
    """검증 데이터셋으로 v2 전략의 최적 파라미터 탐색(목표: 소르티노 최대)."""
    lookback_sets = [[100, 200, 300], [150, 300, 450], [200, 400]]
    quantile_candidates = [[0.20, 0.80], [0.25, 0.75]]
    hysteresis_candidates = [0.01, 0.02, 0.03]
    min_holdings = [1, 2]  # 최소 보유 개월
    check_months = [1]     # 점검 주기 고정 (실무 안정성)

    best = {"sortino": -np.inf}
    param_grid = list(product(lookback_sets, quantile_candidates, hysteresis_candidates, min_holdings, check_months))

    for lp, qt, h, mh, cm in param_grid:
        params = {"lookback_periods": lp, "quantile_thresholds": qt, "base_hysteresis": h, "min_holding_months": mh, "check_months": cm}
        try:
            series = strat_dynamic_rebalance_v2(df_val, initial_cash, transaction_cost, **params)
            tmp = pd.DataFrame({"날짜": df_val["날짜"], "strategy": series})
            metrics = calculate_performance_metrics(tmp, ["strategy"], risk_free_rate_annual=risk_free_rate_annual)
            sortino = float(metrics.loc["strategy", "소르티노지수"])
            if pd.notna(sortino) and sortino > best["sortino"]:
                best = {**params, "sortino": sortino}
        except Exception:
            continue

    if "lookback_periods" not in best:
        print("[경고] 최적 파라미터 탐색 실패 → 기본값 사용")
        return {"lookback_periods": [200, 400], "quantile_thresholds": [0.25, 0.75], "base_hysteresis": 0.02, "min_holding_months": 1, "check_months": 1, "sortino": np.nan}

    return best

# -----------------------------------------------------------------------------
# 6) 시각화 (NaN/짧은구간 방어)
# -----------------------------------------------------------------------------
def _smart_annotate(ax: plt.Axes, x_series: pd.Series, y_series_dict: Dict[str, pd.Series], top_k: int = None):
    """그래프 마지막 지점에 겹치지 않게 스마트 주석 추가."""
    items = [(name, s.dropna()) for name, s in y_series_dict.items()]
    items = [(name, s, s.iloc[-1]) for name, s in items if len(s)]
    if not items:
        return
    items.sort(key=lambda t: t[2], reverse=True)
    if top_k is not None:
        items = items[:top_k]

    last_x = x_series.iloc[-1]
    for i, (name, series, last_val) in enumerate(items):
        ax.annotate(
            f"{name}: {last_val:,.2f}",
            xy=(last_x, last_val),
            xytext=(12, 0 if i == 0 else (8 if i % 2 == 0 else -8)),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color='gray', connectionstyle="arc3,rad=0.08"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75)
        )

def plot_results(results_dict: Dict[str, pd.DataFrame]) -> None:
    """전략별 백테스팅 결과 시각화."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'hspace': 0.32})
    ax1, ax2 = axes

    # --- 1) 전체 기간 ---
    df_all = results_dict["전체"].copy()
    # 유효 데이터만
    all_cols = [c for c in df_all.columns if c != "날짜" and df_all[c].notna().sum() >= 2]
    x_all = df_all["날짜"]
    y_all = {}
    for col in all_cols:
        label = col.replace("_전체", "")
        ax1.plot(x_all, df_all[col], label=label, linewidth=1.6, alpha=0.9)
        y_all[label] = df_all[col]

    if len(all_cols):
        ax1.set_yscale("log")
        ax1.set_title("전체 기간 전략별 포트폴리오 가치 (로그 스케일)", fontsize=14, weight='bold')
        ax1.set_ylabel("포트폴리오 가치 ($)", fontsize=11)
        ax1.legend(loc="upper left", fontsize=9, ncol=2, frameon=True)
        ax1.grid(True, which="both", linestyle='--', linewidth=0.5)
        _smart_annotate(ax1, x_all, y_all, top_k=5)
    else:
        ax1.text(0.5, 0.5, "전체 기간에 유효한 시리즈가 없습니다.", ha='center', va='center', transform=ax1.transAxes)

    # --- 2) 테스트 기간 ---
    df_test = results_dict["테스트"].copy()
    test_cols = [c for c in df_test.columns if c != "날짜" and df_test[c].notna().sum() >= 2]

    if len(test_cols) == 0:
        ax2.text(0.5, 0.5, "테스트 기간에 유효한 시리즈가 없습니다.", ha='center', va='center', transform=ax2.transAxes)
    else:
        x_test = df_test["날짜"]
        y_test = {}
        for col in test_cols:
            label = col.replace("_테스트", "")
            ax2.plot(x_test, df_test[col], label=label, linewidth=1.8)
            y_test[label] = df_test[col]

        ax2.set_title("테스트 기간 전략별 포트폴리오 가치", fontsize=14, weight='bold')
        ax2.set_xlabel("날짜", fontsize=11)
        ax2.set_ylabel("포트폴리오 가치 ($)", fontsize=11)
        ax2.legend(loc="upper left", fontsize=9, ncol=2, frameon=True)
        ax2.grid(True, which="both", linestyle='--', linewidth=0.5)
        _smart_annotate(ax2, x_test, y_test, top_k=5)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 7) 파이프라인 실행
# -----------------------------------------------------------------------------
def run_full_pipeline(
    tickers: List[str] = ["GLD", "SLV", "SPY", "SHY"],
    start_date: str = "2006-01-01",
    end_date: Optional[str] = None,
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,  # 0.1%
    use_shy_as_rf: bool = False       # True: SHY 기반 RF 추정 사용
) -> None:
    """전체 백테스팅 파이프라인 실행."""
    END_DATE = end_date or pd.to_datetime("today").strftime("%Y-%m-%d")

    # 1) 데이터 로드
    print("1) 데이터 로드 및 전처리...")
    df_all = load_and_preprocess_data(tickers, start_date, END_DATE)
    print(f"- 전체 기간: {df_all['날짜'].min().date()} ~ {df_all['날짜'].max().date()}")

    # 2) 위험무위험 수익률 결정(일관성)
    rf_annual = estimate_rf_from_shy(df_all) if use_shy_as_rf else None
    if rf_annual is None:
        rf_annual = 0.02
        print(f"- 위험무위험 수익률: 고정 {rf_annual:.2%}")
    else:
        print(f"- 위험무위험 수익률(SHY 추정): {rf_annual:.2%}")

    # 3) 데이터 분할 (지표 사전계산 재사용)
    n = len(df_all)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    df_train = df_all.iloc[:train_end].reset_index(drop=True)
    df_val   = df_all.iloc[train_end:val_end].reset_index(drop=True)
    df_test  = df_all.iloc[val_end:].reset_index(drop=True)

    # 4) v2 최적 파라미터 탐색 (검증 세트; 동일 RF로 평가)
    print("\n2) v2 최적 파라미터 탐색 중(목표: 소르티노 최대)...")
    best_params_v2 = find_best_params_v2(df_val, initial_cash, transaction_cost, risk_free_rate_annual=rf_annual)
    print("=" * 70)
    print("동적 리밸런싱 v2 - 검증 세트 최적 파라미터")
    for k, v in best_params_v2.items():
        print(f"- {k}: {v}")
    print("=" * 70)

    # 5) 전략 맵 정의 (가독성 명칭)
    strategy_map = {
        "① 금만 보유 (Buy&Hold)": strat_buy_hold_gold,
        "② 반기 50:50 리밸런싱": lambda _df, **kw: strat_periodic_rebalance(_df, apply_initial_fee=False, **kw),
        "③ 금 200일 EMA 추세추종": strat_gold_ema_trend,
        "④ 금은비 동적 리밸런싱 v2": (lambda _df, **kw: run_dynamic_wrapper(_df, strategy_func=strat_dynamic_rebalance_v2, **kw)),
        "⑤ 금은비 동적 리스크관리 v3": (lambda _df, **kw: run_dynamic_wrapper(_df, strategy_func=strat_dynamic_rebalance_v3_risk, **kw)),
    }

    # 6) 세그먼트별 실행 + 성과표
    results: Dict[str, pd.DataFrame] = {}
    price_cols = ["금_가격", "은_가격", "SPY_가격", "SHY_가격"]

    for name, seg in [("훈련", df_train), ("검증", df_val), ("테스트", df_test), ("전체", df_all)]:
        start_idx = get_common_start_index(seg, price_cols)
        res = pd.DataFrame({"날짜": seg["날짜"]})

        for sname, func in strategy_map.items():
            series = run_strategy_from(
                seg, start_idx, func,
                initial_cash=initial_cash,
                transaction_cost=transaction_cost,
                best_params=best_params_v2,
            )
            res[f"{sname}_{name}"] = series

        results[name] = res

        # 성과표 출력 (NaN-safe + MDD 절댓값 표시)
        metric_cols = [c for c in res.columns if c != "날짜"]
        metrics_df = calculate_performance_metrics(res, metric_cols, risk_free_rate_annual=rf_annual)\
            .sort_values("소르티노지수", ascending=False)

        metrics_df_disp = metrics_df.copy()
        if "MaxDrawdown" in metrics_df_disp.columns:
            metrics_df_disp["MaxDrawdown"] = metrics_df_disp["MaxDrawdown"].abs()

        print(f"\n=== {name} 세트 성과 지표 ===")
        display(
            metrics_df_disp.style
            .format({"CAGR": "{:.4%}", "샤프지수": "{:.4f}", "소르티노지수": "{:.4f}", "MaxDrawdown": "{:.2%}"})
            .background_gradient(cmap='viridis', subset=['CAGR', '샤프지수', '소르티노지수'])
            .highlight_max(subset=['MaxDrawdown'], color='salmon')  # 큰 값이 더 나쁜 낙폭
        )

    # 7) 시각화
    print("\n3) 백테스팅 결과 시각화...")
    plot_results(results)


# -----------------------------------------------------------------------------
# 8) 실행
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_full_pipeline(
        tickers=["GLD", "SLV", "SPY", "SHY"],
        start_date="2006-01-01",
        end_date=None,             # 오늘까지
        initial_cash=10_000.0,
        transaction_cost=0.001,    # 0.1%
        use_shy_as_rf=True         # SHY 기반 RF로 샤프/소르티노 평가(일관성)
    )
