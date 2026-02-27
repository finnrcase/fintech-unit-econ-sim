from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import LendingAssumptions


@dataclass(frozen=True)
class UnitEconomicsResult:
    """
    Deterministic unit economics outputs for a one-year origination cohort.
    All values are annualized USD unless noted.
    """
    accounts: float

    # Revenue
    annual_interest_revenue: float
    annual_fee_revenue: float
    annual_gross_revenue: float

    # Costs
    expected_credit_loss: float
    funding_cost: float
    servicing_cost: float
    acquisition_cost: float

    # Profitability
    contribution_profit: float
    contribution_margin: float
    profit_per_account: float


def _get_default_rate(a: LendingAssumptions) -> float:
    """
    Return the annual default rate (PD) from assumptions.

    We standardize on `default_rate`. If your config still uses the older
    `annual_default_rate` name, this provides backward compatibility.
    """
    if hasattr(a, "default_rate"):
        return float(getattr(a, "default_rate"))
    if hasattr(a, "annual_default_rate"):
        return float(getattr(a, "annual_default_rate"))
    raise AttributeError("Assumptions must include `default_rate` (preferred) or `annual_default_rate`.")


def run_deterministic_unit_econ(a: LendingAssumptions) -> UnitEconomicsResult:
    """
    Compute expected unit economics for a consumer lending fintech.

    Modeling choices (intentionally transparent):
    - Interest revenue and funding cost use an 'average outstanding balance' approximation:
      avg_outstanding = origination_balance * avg_outstanding_balance_factor
    - Expected credit loss (ECL) = origination_balance * default_rate * LGD
    - Accounts = origination_balance / avg_loan_balance_per_account
    - CAC = accounts * cac_per_account
    """

    # How many accounts did we acquire to originate this balance?
    accounts = a.annual_origination_balance / a.avg_loan_balance_per_account

    # Approximate average outstanding principal through the year
    avg_outstanding = a.annual_origination_balance * a.avg_outstanding_balance_factor

    # Revenue
    annual_interest_revenue = avg_outstanding * a.apr
    annual_fee_revenue = a.annual_origination_balance * a.upfront_fee_rate
    annual_gross_revenue = annual_interest_revenue + annual_fee_revenue

    # Costs
    default_rate = _get_default_rate(a)
    expected_credit_loss = a.annual_origination_balance * default_rate * a.lgd
    funding_cost = avg_outstanding * a.funding_cost_rate
    servicing_cost = a.annual_origination_balance * a.servicing_cost_rate
    acquisition_cost = accounts * a.cac_per_account

    # Contribution profit (before overhead)
    contribution_profit = annual_gross_revenue - (
        expected_credit_loss + funding_cost + servicing_cost + acquisition_cost
    )

    contribution_margin = (
        float(contribution_profit / annual_gross_revenue) if annual_gross_revenue > 0 else np.nan
    )
    profit_per_account = float(contribution_profit / accounts) if accounts > 0 else np.nan

    return UnitEconomicsResult(
        accounts=float(accounts),
        annual_interest_revenue=float(annual_interest_revenue),
        annual_fee_revenue=float(annual_fee_revenue),
        annual_gross_revenue=float(annual_gross_revenue),
        expected_credit_loss=float(expected_credit_loss),
        funding_cost=float(funding_cost),
        servicing_cost=float(servicing_cost),
        acquisition_cost=float(acquisition_cost),
        contribution_profit=float(contribution_profit),
        contribution_margin=float(contribution_margin),
        profit_per_account=float(profit_per_account),
    )


def result_to_dict(r: UnitEconomicsResult) -> Dict[str, float]:
    """Convert results to a flat dict for DataFrame/table export."""
    return {
        "accounts": r.accounts,
        "annual_interest_revenue": r.annual_interest_revenue,
        "annual_fee_revenue": r.annual_fee_revenue,
        "annual_gross_revenue": r.annual_gross_revenue,
        "expected_credit_loss": r.expected_credit_loss,
        "funding_cost": r.funding_cost,
        "servicing_cost": r.servicing_cost,
        "acquisition_cost": r.acquisition_cost,
        "contribution_profit": r.contribution_profit,
        "contribution_margin": r.contribution_margin,
        "profit_per_account": r.profit_per_account,
    }