###  Live App  
 https://fintech-unit-econ-sim-y89oapbghqh6hs4ygbn68o.streamlit.app/

Lending Unit Economics & Capital Risk Simulator

Summary / Overview
This project is a simulator for structured credit risk and capital adequacy for a consumer lending product.

The model evaluates unit economics under uncertainty by using the following:
- Default Rates
- Volatility of acquisition cost
- Funding cost dynamics
- Stress testing under multiple scenarios
- Macro Shocks
- Required capital needed under expected shortfalls
- Risk-adjusted return analysis

The goal of this project is to determine if a lending product is economically viable after taking into account downside risk and capital constraints.

Model Stages

Deterministic Unit Economics

Calculates profit based on assumptions for:
- APR
- Default Rate
- Customer Acquisition Cost
- Funding Related Costs

Sensitivity Analysis

Measures the marginal impact of:
- Changes in default rate
- CAC changes
- Funding cost changes
- While identifying primary risk drivers.

Monte Carlo Risk Simulation

Ran simulations on 20,000 possible economic environments using truncated normal distributions.

- Outputs included:
- Expected profit
- Chance of loss
- 5% Value at Risk
- 5% Expected Shortfall

Scenario Based Stress Testing
Evaluates adverse environments:
- Competition pressure
- Rate spikes
- Credit changes (higher defaults)
- Economic recession

Capital & Correlation Sensitivity
Extended the simulation to include a macroeconomic factor introducing a correlation between:
- Default Rate
- CAC
- Funding Cost
Findings:
- Under a shock required equity increases
- Capital requirements increase 2.6x as macro correlation intensifies


Insights:
The biggest driver of risk is default rate
Ignoring macro correlations understates capital requirements
Capital adjusted return determines viability, simply looking at expected profitability is not a sufficient metric

| Metric | Result |
|--------|--------|
| Expected Profit | -$1,604,709 |
| Probability of Loss | 86.7% |
| VaR (5%) | -$3,949,445 |
| Expected Shortfall (5%) | -$4,569,268 |
| Required Equity | $4,569,268 |
| RAROC | -35.1% |
| Break-even APR | 20.9% |
| APR for 15% RAROC | 22.2% |

Project Structure

notebooks/
  01_deterministic_model.ipynb
  02_sensitivity_analysis.ipynb
  03_monte_carlo_simulation.ipynb
  04_scenario_stress_tests.ipynb
  05_break_even_frontier.ipynb

src/
  config.py
  model.py
  scenarios.py

outputs/
  figures/
  tables/
  reports/

APP: https://fintech-unit-econ-sim-y89oapbghqh6hs4ygbn68o.streamlit.app/

Running the Interactive App (Local)

Clone the repository and launch the Streamlit interface locally:

```powershell
git clone https://github.com/<your-username>/fintech-unit-econ-sim.git
cd fintech-unit-econ-sim

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app\streamlit_app.py
