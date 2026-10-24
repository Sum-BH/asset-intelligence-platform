# Model Card — RUL Baseline (gradient boosting / random forest)

**Version:** Phase 1 baseline · **Owner:** Sumedh Bhalerao · **Status:** Staging

## Intended use
Estimate remaining useful life (cycles) for turbofan-style degrading assets, to set a
maintenance planning window. Decision support only — a human approves all resulting work.

## Data
NASA C-MAPSS (public) or the repository's synthetic generator. Features: variance-selected
sensors with rolling mean/std (5/10/20-cycle windows) and first differences; min–max
scaling fitted on training data only.

## Metrics
RMSE (cycles) and the asymmetric NASA score. Current recorded values are in
`docs/business/NUMBERS.md`; sandbox figures come from synthetic data and must be
re-measured on real FD001 before external citation.

## Limitations
- Trained on run-to-failure data; underestimates life after partial repairs — reset
  baselines post-maintenance.
- Single-condition (FD001) scope; multi-condition sets need operating-regime clustering.
- RUL is a planning distribution, not a date: action windows should never be narrower
  than ±1 RMSE.

## Drift triggers
Feature PSI > 0.2 on any monitored feature, or KS test p < 0.01 on the prediction
distribution. On trigger: demote to Staging and re-run the eval harness.

## Ethical / operational notes
Outputs must not be used to evaluate technician performance.
