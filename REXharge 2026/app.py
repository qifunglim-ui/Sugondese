import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_EXT = ".xlsx"
OLD_MD_RATE = 30.30
NEW_MD_RATE = 97.06


@dataclass
class SimulationResult:
    baseline: pd.DataFrame
    optimized: pd.DataFrame
    baseline_cost_old: float
    baseline_cost_new: float
    optimized_cost_new: float
    baseline_md_kw: float
    optimized_md_kw: float


def _normalize_cols(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        c = str(c).strip().lower().replace("/", " ")
        c = "".join(ch if ch.isalnum() else "_" for ch in c)
        while "__" in c:
            c = c.replace("__", "_")
        out.append(c.strip("_"))
    return out


def _find_header_row(raw: pd.DataFrame) -> int | None:
    for i in range(min(30, len(raw))):
        row = " | ".join(str(x).lower() for x in raw.iloc[i].tolist())
        if "date / end time" in row or "start_time" in row:
            return i
    return None


def _load_one_excel(path: str) -> pd.DataFrame:
    try:
        clean = pd.read_excel(path)
        clean.columns = _normalize_cols(clean.columns.tolist())
        if {"start_time", "end_time", "kw_import"}.issubset(set(clean.columns)):
            clean["timestamp"] = pd.to_datetime(clean["end_time"], errors="coerce")
            clean["kw_import"] = pd.to_numeric(clean["kw_import"], errors="coerce").fillna(0.0)
            clean["kw_export"] = pd.to_numeric(clean.get("kw_export", 0), errors="coerce").fillna(0.0)
            clean["source"] = os.path.basename(path)
            return clean[["timestamp", "kw_import", "kw_export", "source"]].dropna(subset=["timestamp"])
    except Exception:
        pass

    raw = pd.read_excel(path, header=None)
    header_row = _find_header_row(raw)
    if header_row is None:
        return pd.DataFrame(columns=["timestamp", "kw_import", "kw_export", "source"])

    headers = _normalize_cols(raw.iloc[header_row].astype(str).tolist())
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = headers

    rename_map = {}
    for col in data.columns:
        if "date" in col and "time" in col:
            rename_map[col] = "timestamp_raw"
        elif col.endswith("kw_import") or col == "main_meter_1" or col == "unnamed_2":
            rename_map[col] = "kw_import"
        elif col.endswith("kw_export") or col == "main_meter" or col == "unnamed":
            rename_map[col] = "kw_export"

    data = data.rename(columns=rename_map)
    if "timestamp_raw" not in data.columns:
        candidates = [c for c in data.columns if "time" in c]
        if candidates:
            data = data.rename(columns={candidates[0]: "timestamp_raw"})

    data["timestamp"] = pd.to_datetime(data.get("timestamp_raw"), errors="coerce")
    data["kw_import"] = pd.to_numeric(data.get("kw_import", 0), errors="coerce").fillna(0.0)
    data["kw_export"] = pd.to_numeric(data.get("kw_export", 0), errors="coerce").fillna(0.0)
    data["source"] = os.path.basename(path)
    return data[["timestamp", "kw_import", "kw_export", "source"]].dropna(subset=["timestamp"])


def load_all_data(base_dir: str) -> pd.DataFrame:
    files = [f for f in os.listdir(base_dir) if f.lower().endswith(DATA_EXT)]
    parts = []
    for f in files:
        df = _load_one_excel(os.path.join(base_dir, f))
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "kw_import", "kw_export", "source", "net_kw"])
    data = pd.concat(parts, ignore_index=True)
    data = data.sort_values("timestamp").drop_duplicates(subset=["timestamp", "source"])
    data["net_kw"] = (data["kw_import"] - data["kw_export"]).clip(lower=0)
    return data


def estimate_step_hours(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.5
    diffs = df["timestamp"].sort_values().diff().dropna().dt.total_seconds() / 3600.0
    mode = diffs.mode()
    return float(mode.iloc[0]) if not mode.empty else 0.5


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("timestamp")
    x["hour"] = x["timestamp"].dt.hour
    x["dayofweek"] = x["timestamp"].dt.dayofweek
    x["month"] = x["timestamp"].dt.month
    x["is_weekend"] = (x["dayofweek"] >= 5).astype(int)
    x["lag_1"] = x["net_kw"].shift(1)
    x["lag_2"] = x["net_kw"].shift(2)
    x["lag_48"] = x["net_kw"].shift(48)
    x["roll_mean_48"] = x["net_kw"].rolling(48, min_periods=1).mean().shift(1)
    return x.dropna().copy()


def train_forecast_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if len(df) < 300:
        return pd.DataFrame(), {"warning": "Not enough rows for ML forecast."}

    feat_df = add_features(df)
    cols = ["hour", "dayofweek", "month", "is_weekend", "lag_1", "lag_2", "lag_48", "roll_mean_48"]
    split = int(len(feat_df) * 0.8)
    train, test = feat_df.iloc[:split], feat_df.iloc[split:]
    if test.empty:
        return pd.DataFrame(), {"warning": "Not enough holdout data for evaluation."}

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(train[cols], train["net_kw"])
    pred = model.predict(test[cols])

    metrics = {
        "mae_kw": float(mean_absolute_error(test["net_kw"], pred)),
        "rmse_kw": float(np.sqrt(mean_squared_error(test["net_kw"], pred))),
    }

    # Recursive next 48-step forecast.
    horizon = 48
    history = df.sort_values("timestamp").copy()
    future_rows = []
    step_h = estimate_step_hours(history)
    for _ in range(horizon):
        next_ts = history["timestamp"].iloc[-1] + pd.Timedelta(hours=step_h)
        tmp = history.tail(48).copy()
        row = {
            "timestamp": next_ts,
            "net_kw": history["net_kw"].iloc[-1],
            "hour": next_ts.hour,
            "dayofweek": next_ts.dayofweek,
            "month": next_ts.month,
            "is_weekend": 1 if next_ts.dayofweek >= 5 else 0,
            "lag_1": history["net_kw"].iloc[-1],
            "lag_2": history["net_kw"].iloc[-2] if len(history) > 1 else history["net_kw"].iloc[-1],
            "lag_48": history["net_kw"].iloc[-48] if len(history) >= 48 else history["net_kw"].iloc[-1],
            "roll_mean_48": tmp["net_kw"].mean(),
        }
        yhat = float(model.predict(pd.DataFrame([row])[cols])[0])
        future_rows.append({"timestamp": next_ts, "forecast_kw": max(0.0, yhat)})
        history = pd.concat([history, pd.DataFrame([{"timestamp": next_ts, "net_kw": yhat}])], ignore_index=True)

    forecast_df = pd.DataFrame(future_rows)
    return forecast_df, metrics


def simulate_load_strategy(
    df: pd.DataFrame,
    flexible_pct: float,
    shift_pct: float,
    battery_power_kw: float,
    battery_capacity_kwh: float,
    target_peak_kw: float,
    energy_rate_rm_per_kwh: float,
) -> SimulationResult:
    work = df.sort_values("timestamp").copy()
    dt_h = estimate_step_hours(work)

    work["is_peak_window"] = (
        (work["timestamp"].dt.dayofweek < 5)
        & (work["timestamp"].dt.hour >= 14)
        & (work["timestamp"].dt.hour < 22)
    )

    baseline = work[["timestamp", "net_kw", "kw_import", "kw_export", "is_peak_window"]].copy()
    baseline = baseline.rename(columns={"net_kw": "baseline_kw"})

    # Load shifting: reduce part of flexible load in peak and spread to off-peak.
    opt = baseline.copy()
    opt["optimized_kw"] = opt["baseline_kw"]
    peak_mask = opt["is_peak_window"]
    off_mask = ~peak_mask

    reduce_kw = (opt.loc[peak_mask, "baseline_kw"] * flexible_pct * shift_pct).clip(lower=0)
    energy_to_shift_kwh = float((reduce_kw * dt_h).sum())
    opt.loc[peak_mask, "optimized_kw"] = (opt.loc[peak_mask, "optimized_kw"] - reduce_kw).clip(lower=0)

    if off_mask.sum() > 0 and energy_to_shift_kwh > 0:
        add_kw = energy_to_shift_kwh / (off_mask.sum() * dt_h)
        opt.loc[off_mask, "optimized_kw"] = opt.loc[off_mask, "optimized_kw"] + add_kw

    # Battery dispatch: charge from excess solar export, discharge during peak above threshold.
    soc = 0.0
    soc_list = []
    final_kw = []
    for _, row in opt.iterrows():
        load_kw = float(row["optimized_kw"])
        solar_surplus_kw = max(0.0, float(row["kw_export"] - row["kw_import"]))

        if battery_capacity_kwh > 0 and battery_power_kw > 0:
            charge_kw = min(battery_power_kw, solar_surplus_kw, (battery_capacity_kwh - soc) / dt_h)
            soc += charge_kw * dt_h
            if row["is_peak_window"] and load_kw > target_peak_kw:
                discharge_kw = min(battery_power_kw, soc / dt_h, load_kw - target_peak_kw)
                soc -= discharge_kw * dt_h
                load_kw = max(0.0, load_kw - discharge_kw)

        soc_list.append(soc)
        final_kw.append(load_kw)

    opt["optimized_kw"] = final_kw
    opt["battery_soc_kwh"] = soc_list

    baseline_energy_kwh = float((baseline["baseline_kw"] * dt_h).sum())
    optimized_energy_kwh = float((opt["optimized_kw"] * dt_h).sum())

    baseline_md_kw = float(baseline["baseline_kw"].max())
    optimized_md_kw = float(opt["optimized_kw"].max())

    baseline_cost_old = baseline_energy_kwh * energy_rate_rm_per_kwh + baseline_md_kw * OLD_MD_RATE
    baseline_cost_new = baseline_energy_kwh * energy_rate_rm_per_kwh + baseline_md_kw * NEW_MD_RATE
    optimized_cost_new = optimized_energy_kwh * energy_rate_rm_per_kwh + optimized_md_kw * NEW_MD_RATE

    return SimulationResult(
        baseline=baseline,
        optimized=opt,
        baseline_cost_old=baseline_cost_old,
        baseline_cost_new=baseline_cost_new,
        optimized_cost_new=optimized_cost_new,
        baseline_md_kw=baseline_md_kw,
        optimized_md_kw=optimized_md_kw,
    )


def main() -> None:
    st.set_page_config(page_title="REXharge MVP - Load Shifting & Peak Shaving", layout="wide")
    st.title("REXharge 2026 MVP: AI Load Shifting & Peak Shaving")
    st.caption("Malaysia tariff-aware prototype for forecasting, peak detection, and autonomous strategy simulation.")

    base_dir = "."
    data = load_all_data(base_dir)
    if data.empty:
        st.error("No valid Excel data found in this folder.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Points", f"{len(data):,}")
    c2.metric("Date Range", f"{data['timestamp'].min().date()} -> {data['timestamp'].max().date()}")
    c3.metric("Avg Net Load (kW)", f"{data['net_kw'].mean():.1f}")
    c4.metric("Peak Net Load (kW)", f"{data['net_kw'].max():.1f}")

    st.subheader("1) Forecasting & Peak Identification")
    forecast_df, metrics = train_forecast_model(data)
    if "warning" in metrics:
        st.warning(metrics["warning"])
    else:
        a, b = st.columns(2)
        a.metric("Forecast MAE (kW)", f"{metrics['mae_kw']:.2f}")
        b.metric("Forecast RMSE (kW)", f"{metrics['rmse_kw']:.2f}")

    peak_threshold = st.slider("Peak threshold percentile", 80, 99, 90)
    threshold_kw = float(np.percentile(data["net_kw"], peak_threshold))
    data["is_peak"] = data["net_kw"] >= threshold_kw
    st.write(f"Detected peak threshold: **{threshold_kw:.2f} kW** ({peak_threshold}th percentile)")

    show_hist = data[["timestamp", "net_kw"]].copy()
    fig_hist = px.line(show_hist, x="timestamp", y="net_kw", title="Historical Net Load")
    fig_hist.add_hline(y=threshold_kw, line_dash="dash", annotation_text="Peak threshold")
    st.plotly_chart(fig_hist, use_container_width=True)

    if not forecast_df.empty:
        fig_fc = px.line(forecast_df, x="timestamp", y="forecast_kw", title="Next 24h Forecast (48 x 30min)")
        st.plotly_chart(fig_fc, use_container_width=True)

    st.subheader("2) Automated Load Shifting + Peak Shaving Strategy")
    s1, s2, s3 = st.columns(3)
    flexible_pct = s1.slider("Flexible load share (%)", 0, 80, 30) / 100.0
    shift_pct = s2.slider("Shiftable fraction of flexible load (%)", 0, 100, 60) / 100.0
    energy_rate = s3.number_input("Energy rate (RM/kWh)", min_value=0.1, max_value=2.0, value=0.40, step=0.01)

    b1, b2, b3 = st.columns(3)
    battery_power_kw = b1.number_input("Battery power (kW)", min_value=0.0, value=300.0, step=50.0)
    battery_capacity_kwh = b2.number_input("Battery capacity (kWh)", min_value=0.0, value=600.0, step=50.0)
    target_peak_kw = b3.number_input(
        "Target peak cap (kW)",
        min_value=0.0,
        value=float(np.percentile(data["net_kw"], 95)),
        step=10.0,
    )

    res = simulate_load_strategy(
        data,
        flexible_pct=flexible_pct,
        shift_pct=shift_pct,
        battery_power_kw=float(battery_power_kw),
        battery_capacity_kwh=float(battery_capacity_kwh),
        target_peak_kw=float(target_peak_kw),
        energy_rate_rm_per_kwh=float(energy_rate),
    )

    md_reduction = res.baseline_md_kw - res.optimized_md_kw
    pct_md_reduction = (md_reduction / res.baseline_md_kw * 100.0) if res.baseline_md_kw > 0 else 0.0
    savings_vs_new = res.baseline_cost_new - res.optimized_cost_new

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline MD (kW)", f"{res.baseline_md_kw:.1f}")
    k2.metric("Optimized MD (kW)", f"{res.optimized_md_kw:.1f}", delta=f"-{md_reduction:.1f} kW")
    k3.metric("MD Reduction", f"{pct_md_reduction:.1f}%")
    k4.metric("Estimated Savings vs New Tariff", f"RM {savings_vs_new:,.0f}")

    c_old, c_new, c_opt = st.columns(3)
    c_old.metric("Baseline Bill @ Old MD", f"RM {res.baseline_cost_old:,.0f}")
    c_new.metric("Baseline Bill @ New MD", f"RM {res.baseline_cost_new:,.0f}")
    c_opt.metric("Optimized Bill @ New MD", f"RM {res.optimized_cost_new:,.0f}")

    merged = res.optimized.merge(res.baseline[["timestamp", "baseline_kw"]], on="timestamp", how="left")
    fig_compare = px.line(
        merged,
        x="timestamp",
        y=["baseline_kw", "optimized_kw"],
        title="Baseline vs Optimized Net Load",
        labels={"value": "kW", "variable": "Profile"},
    )
    fig_compare.add_hline(y=target_peak_kw, line_dash="dot", annotation_text="Target peak cap")
    st.plotly_chart(fig_compare, use_container_width=True)

    fig_soc = px.line(merged, x="timestamp", y="battery_soc_kwh", title="Battery State of Charge (kWh)")
    st.plotly_chart(fig_soc, use_container_width=True)

    # Very simple sizing hint for report narrative.
    worst_peak_excess_kw = float((res.baseline["baseline_kw"] - target_peak_kw).clip(lower=0).max())
    st.info(
        f"Suggested battery minimum for target cap: >= {worst_peak_excess_kw:.1f} kW power, "
        f"with at least {worst_peak_excess_kw * estimate_step_hours(data):.1f} kWh short-burst energy."
    )

    export_df = merged[["timestamp", "baseline_kw", "optimized_kw", "battery_soc_kwh"]].copy()
    st.download_button(
        "Download simulation results (CSV)",
        data=export_df.to_csv(index=False),
        file_name="rexharge_mvp_simulation_results.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown(
        "### MVP Scope Notes\n"
        "- Forecasting model: Random Forest with temporal + lag features.\n"
        "- Peak windows: Weekdays 14:00-22:00 (editable in code for local tariff assumptions).\n"
        "- Strategy: Demand response (load shifting) + battery dispatch from solar surplus.\n"
        "- Billing: Energy charge + MD charge (Old RM30.30/kW vs New RM97.06/kW).\n"
    )


if __name__ == "__main__":
    main()
