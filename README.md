# Rossmann Store Sales Forecasting

## Business Problem
Managing labor, inventory, and logistics across a large retail chain is difficult when daily customer traffic fluctuates. For managers overseeing more than 1,100 drug stores, inaccurate sales estimates create two costly problems:

- **Understaffing & Stockouts**: Inability to meet customer demand during promotional peaks, resulting in lost revenue and degraded customer experience.
- **Overstaffing & Holding Costs**: Inflated payroll expenses and tied-up working capital during slower shopping periods.

To align with standard retail scheduling and distribution cycles, store leaders need reliable, store-specific daily revenue forecasts up to 6 weeks in advance.

## Solution
An automated sales forecasting service that predicts daily store-level revenue up to 42 days (6 weeks) ahead. In addition to daily expected revenue, the solution produces uncertainty intervals (10th and 90th percentiles), giving decision-makers clear upper and lower demand boundaries to plan for varying operational scenarios.

## Data
The solution leverages daily operational and store data across 1,115 Rossmann retail locations:

- **Operational Drivers**: Store open/closed status, customer promotions, school holidays, and state holidays.
- **Store Profiles**: Store format, product assortment level, distance to nearest competitor, and competitor opening timelines.
- **Calendar & Trends**: Day-of-week demand patterns, monthly seasonality, and historical sales velocity.

## Business Value
- **Optimized Workforce Scheduling**: Enables store managers to align staff shifts directly with anticipated revenue surges, reducing unnecessary labor costs while preventing understaffed peak hours.
- **Smarter Inventory Replenishment**: Informs warehouse and replenishment schedules around promotional events, lowering product stockout rates and cutting holding costs.
- **Demand Risk Management**: Upper and lower prediction intervals allow operations teams to hedge against unexpected demand spikes or downturns.
- **Automated Operational Guardrails**: Enforces zero sales on store closure days (Sundays and public holidays), preventing misallocated logistics and staffing resources.

## Machine Learning Approach
- **Problem Type**: Supervised tabular regression predicting daily sales in Euros.
- **Model**: An **XGBoost Regressor** trained on store characteristics, promotional schedules, and historical lag trends, combined with **LightGBM Quantile Regressors** for uncertainty bounds.

## Result
- **Accuracy**: Achieved an **11.3% RMSPE** (Root Mean Square Percentage Error) with an average error of approximately €565 on a strict 42-day future evaluation period.
- **Business Impact**: Outperformed traditional historical naive planning rules (which showed a 41.0% error rate) by **over 72%**, providing retail planners with dependable revenue estimates that significantly de-risk staffing and inventory decisions.
