from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class LendingAssumptions(BaseModel):
    """
    Baseline assumptions for a consumer lending fintech unit economics model.
    """

    model_config = ConfigDict(frozen=True)

    # Portfolio scale
    annual_origination_balance: float = Field(
        default=100_000_000,
        ge=0,
        description="Total principal originated in a year (USD).",
    )

    # Pricing
    apr: float = Field(
        default=0.18,
        ge=0,
        le=1,
        description="Annual Percentage Rate charged to borrowers.",
    )

    upfront_fee_rate: float = Field(
        default=0.02,
        ge=0,
        le=0.2,
        description="Upfront origination fee as % of principal.",
    )

    # Credit risk
    annual_default_rate: float = Field(
        default=0.06,
        ge=0,
        le=1,
        description="Fraction of principal that defaults over 12 months.",
    )

    lgd: float = Field(
        default=0.70,
        ge=0,
        le=1,
        description="Loss given default (1 - recovery rate).",
    )

    # Funding + operating costs
    funding_cost_rate: float = Field(
        default=0.06,
        ge=0,
        le=1,
        description="Cost of funds as annual rate.",
    )

    servicing_cost_rate: float = Field(
        default=0.01,
        ge=0,
        le=0.2,
        description="Servicing cost as % of principal.",
    )

    # Acquisition economics
    cac_per_account: float = Field(
        default=250.0,
        ge=0,
        description="Customer acquisition cost per account.",
    )

    avg_loan_balance_per_account: float = Field(
        default=5_000.0,
        ge=1,
        description="Average principal per account.",
    )

    # Timing approximation
    avg_outstanding_balance_factor: float = Field(
        default=0.55,
        ge=0,
        le=1,
        description="Average outstanding balance as % of original principal.",
    )

def base_case() -> LendingAssumptions:
    """Returns a realistic synthetic baseline configuration."""
    return LendingAssumptions()