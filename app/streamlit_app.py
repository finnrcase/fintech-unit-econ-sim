from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Repo path setup (robust)
# -----------------------------
def add_repo_to_path() -> Path:
    """
    Ensure repo root is on sys.path so `from src...` works when running Streamlit.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # app/ -> repo root

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return repo_root


REPO_ROOT = add_repo_to_path()

from src.config import base_case
from src.model import run_deterministic_unit_econ, result_to_dict


# -----------------------------
# Helpers
# -----------------------------
def truncated_normal(
    n: int,
    mean: float,
    sd: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw samples from a truncated normal using rejection sampling.
    Avoids scipy dependency; fast enough for 10k–50k draws.
    """
    out = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        draw = rng.normal(mean, sd, size=(n - filled) * 3)
        draw = draw[(draw >= low) & (draw <= high)]
        take = min(len(draw), n - filled)
        out[filled : filled + take] = draw[:take]
        filled += take
    return out


def compute_risk_metrics(profit: np.ndarray) -> dict:
    exp = float(profit.mean())
    prob_loss = float((profit < 0).mean())
    var_5 = float(np.quantile(profit, 0.05))
    es_5 = float(profit[profit <= var_5].mean())
    req_equity = abs(es_5)
    raroc = exp / req_equity if req_equity > 0 else np.nan

    return {
        "Expected Profit": exp,
        "Probability of Loss": prob_loss,
        "VaR (5%)": var_5,
        "Expected Shortfall (5%)": es_5,
        "Required Equity (ES)": req_equity,
        "RAROC": raroc,
    }


def format_exec_table(metrics: dict, break_even_apr: float | None, apr_for_target_raroc: float | None) -> pd.DataFrame:
    rows = []

    def fmt_money(x: float) -> str:
        return f"${x:,.0f}"

    def fmt_pct(x: float) -> str:
        return f"{x:.1%}"

    rows.append(("Expected Profit", fmt_money(metrics["Expected Profit"])))
    rows.append(("Probability of Loss", fmt_pct(metrics["Probability of Loss"])))
    rows.append(("VaR (5%)", fmt_money(metrics["VaR (5%)"])))
    rows.append(("Expected Shortfall (5%)", fmt_money(metrics["Expected Shortfall (5%)"])))
    rows.append(("Required Equity (ES)", fmt_money(metrics["Required Equity (ES)"])))
    rows.append(("RAROC", fmt_pct(metrics["RAROC"])))

    if break_even_apr is not None:
        rows.append(("Break-even APR", fmt_pct(break_even_apr)))
    if apr_for_target_raroc is not None:
        rows.append(("APR for target RAROC", fmt_pct(apr_for_target_raroc)))

    return pd.DataFrame(rows, columns=["Metric", "Value"])


def solve_break_even_apr(base, default_field: str, apr_low=0.05, apr_high=0.60, steps=120) -> float | None:
    """
    Find an approximate APR where deterministic contribution profit crosses 0.
    """
    aprs = np.linspace(apr_low, apr_high, steps)
    profits = []
    for apr in aprs:
        temp = base.model_copy(update={"apr": float(apr)})
        d = result_to_dict(run_deterministic_unit_econ(temp))
        profits.append(d["contribution_profit"])
    profits = np.array(profits)

    idx = np.where(profits >= 0)[0]
    if len(idx) == 0:
        return None
    return float(aprs[idx[0]])


def solve_apr_for_target_raroc(
    base,
    default_field: str,
    target_raroc: float,
    N: int,
    rng: np.random.Generator,
    dr_draws: np.ndarray,
    cac_draws: np.ndarray,
    fc_draws: np.ndarray,
    apr_low=0.05,
    apr_high=0.60,
    steps=70,
) -> float | None:
    """
    Find the lowest APR that achieves target RAROC given fixed simulation draws.
    (Fixing draws makes it stable and fast.)
    """
    aprs = np.linspace(apr_low, apr_high, steps)

    for apr in aprs:
        temp_base = base.model_copy(update={"apr": float(apr)})

        profits = np.empty(N, dtype=float)
        for i in range(N):
            temp = temp_base.model_copy(
                update={
                    default_field: float(dr_draws[i]),
                    "cac_per_account": float(cac_draws[i]),
                    "funding_cost_rate": float(fc_draws[i]),
                }
            )
            d = result_to_dict(run_deterministic_unit_econ(temp))
            profits[i] = d["contribution_profit"]

        m = compute_risk_metrics(profits)
        if m["RAROC"] >= target_raroc:
            return float(apr)

    return None


def run_simulation(
    base,
    default_field: str,
    N: int,
    seed: int,
    use_correlated: bool,
    rho: float,
    dr_sd: float,
    cac_sd: float,
    fc_sd: float,
    beta_dr: float,
    beta_cac: float,
    beta_fc: float,
) -> pd.DataFrame:
    """
    Runs Monte Carlo and returns a DataFrame with profit and the underlying draws.
    """
    rng = np.random.default_rng(seed)

    dr_mean = float(getattr(base, default_field))
    cac_mean = float(base.cac_per_account)
    fc_mean = float(base.funding_cost_rate)

    if not use_correlated:
        dr = truncated_normal(N, dr_mean, dr_sd, 0.01, 0.30, rng)
        cac = truncated_normal(N, cac_mean, cac_sd, 50.0, 800.0, rng)
        fc = truncated_normal(N, fc_mean, fc_sd, 0.00, 0.25, rng)
    else:
        Z = rng.normal(0, 1, N)
        eps_dr = rng.normal(0, 1, N)
        eps_cac = rng.normal(0, 1, N)
        eps_fc = rng.normal(0, 1, N)

        dr = dr_mean + dr_sd * (rho * beta_dr * Z + 0.5 * eps_dr)
        cac = cac_mean + cac_sd * (rho * beta_cac * Z + 0.5 * eps_cac)
        fc = fc_mean + fc_sd * (rho * beta_fc * Z + 0.5 * eps_fc)

        dr = np.clip(dr, 0.01, 0.30)
        cac = np.clip(cac, 50.0, 800.0)
        fc = np.clip(fc, 0.00, 0.25)

    profits = np.empty(N, dtype=float)

    # (Loop is fine for 10k–20k and keeps logic identical to notebooks)
    for i in range(N):
        temp = base.model_copy(
            update={
                default_field: float(dr[i]),
                "cac_per_account": float(cac[i]),
                "funding_cost_rate": float(fc[i]),
            }
        )
        d = result_to_dict(run_deterministic_unit_econ(temp))
        profits[i] = d["contribution_profit"]

    return pd.DataFrame(
        {
            "profit": profits,
            "default_rate": dr,
            "cac_per_account": cac,
            "funding_cost_rate": fc,
        }
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fintech Unit Econ + Risk Simulator", layout="wide")

st.title("Fintech Lending Unit Economics + Capital Risk Simulator")
st.caption("A lightweight risk-desk style decision tool: profitability, downside risk (VaR/ES), capital proxy, and RAROC.")

base = base_case()
default_field = "default_rate" if hasattr(base, "default_rate") else "annual_default_rate"

with st.sidebar:
    st.header("Model Inputs")

    apr = st.slider("APR", min_value=0.05, max_value=0.60, value=float(base.apr), step=0.005)
    default_rate = st.slider("Default rate", min_value=0.01, max_value=0.30, value=float(getattr(base, default_field)), step=0.005)
    cac = st.slider("CAC per account ($)", min_value=25.0, max_value=800.0, value=float(base.cac_per_account), step=5.0)
    funding = st.slider("Funding cost rate", min_value=0.00, max_value=0.25, value=float(base.funding_cost_rate), step=0.0025)

    st.divider()
    st.header("Simulation Controls")
    N = st.select_slider("Monte Carlo draws", options=[5000, 10000, 20000, 50000], value=10000)
    seed = st.number_input("Random seed", min_value=1, max_value=1_000_000, value=42, step=1)

    st.divider()
    st.header("Correlation Model")
    use_corr = st.toggle("Enable correlated macro shocks", value=True)
    rho = st.slider("Correlation strength (ρ)", 0.0, 1.5, 1.0, 0.05, disabled=not use_corr)

    with st.expander("Advanced (volatility + factor loadings)"):
        dr_sd = st.number_input("Default rate σ", value=0.015, step=0.001, format="%.3f")
        cac_sd = st.number_input("CAC σ ($)", value=40.0, step=5.0)
        fc_sd = st.number_input("Funding cost σ", value=0.01, step=0.001, format="%.3f")

        beta_dr = st.number_input("β default", value=1.0, step=0.1)
        beta_cac = st.number_input("β CAC", value=0.8, step=0.1)
        beta_fc = st.number_input("β funding", value=0.6, step=0.1)

    st.divider()
    st.header("Target Return")
    target_raroc = st.slider("Target RAROC", 0.00, 0.40, 0.15, 0.01)

    st.divider()
    save_outputs = st.toggle("Save figures + tables to /outputs", value=False)

# Apply user inputs to base assumptions
base_ui = base.model_copy(
    update={
        "apr": float(apr),
        default_field: float(default_rate),
        "cac_per_account": float(cac),
        "funding_cost_rate": float(funding),
    }
)

colA, colB = st.columns([1.1, 0.9], gap="large")

with colA:
    st.subheader("Deterministic Snapshot (Single Expected Case)")
    det = result_to_dict(run_deterministic_unit_econ(base_ui))

    det_table = pd.DataFrame(
        {
            "Metric": [
                "Accounts",
                "Annual Gross Revenue",
                "Expected Credit Loss",
                "Funding Cost",
                "Servicing Cost",
                "Acquisition Cost",
                "Contribution Profit",
                "Contribution Margin",
                "Profit per Account",
            ],
            "Value": [
                f"{det['accounts']:,.0f}",
                f"${det['annual_gross_revenue']:,.0f}",
                f"${det['expected_credit_loss']:,.0f}",
                f"${det['funding_cost']:,.0f}",
                f"${det['servicing_cost']:,.0f}",
                f"${det['acquisition_cost']:,.0f}",
                f"${det['contribution_profit']:,.0f}",
                f"{det['contribution_margin']:.1%}",
                f"${det['profit_per_account']:,.0f}",
            ],
        }
    )
    st.dataframe(det_table, use_container_width=True, hide_index=True)

    st.divider()
    run_btn = st.button("Run Monte Carlo Risk Simulation", type="primary")

with colB:
    st.subheader("Monte Carlo (Risk + Capital Proxy)")
    st.write("Run the simulation to see downside risk, required equity (ES proxy), and RAROC.")

if run_btn:
    with st.spinner("Simulating..."):
        sim = run_simulation(
            base=base_ui,
            default_field=default_field,
            N=int(N),
            seed=int(seed),
            use_correlated=bool(use_corr),
            rho=float(rho),
            dr_sd=float(dr_sd),
            cac_sd=float(cac_sd),
            fc_sd=float(fc_sd),
            beta_dr=float(beta_dr),
            beta_cac=float(beta_cac),
            beta_fc=float(beta_fc),
        )

    metrics = compute_risk_metrics(sim["profit"].to_numpy())

    # Break-even APR (deterministic)
    be_apr = solve_break_even_apr(base_ui, default_field=default_field)

    # APR needed for target RAROC (using the same draws for stability)
    rng = np.random.default_rng(int(seed))
    # Recreate draws consistent with current settings for pricing solve
    # (We use the sim draws directly to keep it aligned.)
    apr_for_target = solve_apr_for_target_raroc(
        base=base_ui.model_copy(update={"apr": float(base_ui.apr)}),  # start from current
        default_field=default_field,
        target_raroc=float(target_raroc),
        N=int(N),
        rng=rng,
        dr_draws=sim["default_rate"].to_numpy(),
        cac_draws=sim["cac_per_account"].to_numpy(),
        fc_draws=sim["funding_cost_rate"].to_numpy(),
        apr_low=0.05,
        apr_high=0.60,
        steps=60,
    )

    exec_table = format_exec_table(metrics, break_even_apr=be_apr, apr_for_target_raroc=apr_for_target)

    with colB:
        st.dataframe(exec_table, use_container_width=True, hide_index=True)

    # Plot profit distribution + VaR/ES lines
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(sim["profit"], bins=60, density=False)
    ax.axvline(metrics["VaR (5%)"], linestyle="--", linewidth=2)
    ax.axvline(metrics["Expected Shortfall (5%)"], linestyle=":", linewidth=2)
    ax.set_title("Profit Distribution (Monte Carlo)")
    ax.set_xlabel("Contribution Profit ($)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Optional: Save outputs
    if save_outputs:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        FIG_DIR = REPO_ROOT / "outputs" / "figures"
        TAB_DIR = REPO_ROOT / "outputs" / "tables"
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        TAB_DIR.mkdir(parents=True, exist_ok=True)

        fig_path = FIG_DIR / f"{ts}_app_profit_distribution.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")

        # Save exec summary (formatted + raw)
        exec_table.to_csv(TAB_DIR / f"{ts}_app_exec_summary_formatted.csv", index=False)

        raw = pd.DataFrame(
            {
                "Metric": list(metrics.keys()),
                "Value_raw": list(metrics.values()),
            }
        )
        raw.to_csv(TAB_DIR / f"{ts}_app_exec_summary_raw.csv", index=False)

        st.success(f"Saved outputs to outputs/: {fig_path.name}")

    # Quick driver correlation table (rank-ish signal)
    corr = sim[["profit", "default_rate", "cac_per_account", "funding_cost_rate"]].corr(numeric_only=True)["profit"].drop("profit")
    st.subheader("Driver Correlation vs Profit (sanity / importance)")
    st.write(corr.to_frame("corr_with_profit").sort_values("corr_with_profit"))

else:
    st.info("Set inputs on the left, then click **Run Monte Carlo Risk Simulation**.")