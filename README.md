## Bayesian Layer & Context Model

- **Beta–Binomial** for boundary rate (success = {4 or 6}). Prior Beta(α,β) updates with observed (successes, trials) to Posterior Beta(α+succ, β+fail).
- We convert posterior mean \( \hat{p}_B \) into a full categorical vector by:
  1) preserving non-boundary categories’ **relative** proportions, scaling them to \(1-\hat{p}_B\),
  2) allocating boundary mass \( \hat{p}_B \) between 4 and 6 in the same ratio as the scenario’s base split.

- **Player vs Bowler**: maintain per-pair counts; the simulator uses the pair-specific posterior every time you simulate.

## Scenario Persistence

- Load or export scenarios as **YAML or JSON** from the **Scenarios** tab.
- Each scenario includes: `name`, `probs` (7 numbers for [·,1,2,3,4,6,W]), and Beta prior `alpha_prior`, `beta_prior`.
