# Bayesian Cricket — Discrete Distributions Playground 🏏

An interactive **Streamlit** app to teach **discrete probability distributions** through a **cricket** storyline. Explore **Bernoulli, Binomial, Categorical, Multinomial, Poisson, Negative Binomial**, then layer on **Bayesian updates** (Beta–Binomial), **player vs bowler context**, and **scenario persistence** via YAML/JSON. Includes a **match simulator** (ball-by-ball).

## Quick Start

```bash
# 1) clone or create repo
git init bayes-cricket && cd bayes-cricket

# 2) (optional) create a venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) install deps
pip install -r requirements.txt

# 4) run the app
streamlit run app.py
```

## What’s Inside (App Navigation)

- **Learn**: See each distribution’s intuition, PMF, mean/variance, and a cricket-tied example.
- **Playground**: Sample from distributions and visualize empirical frequencies.
- **Match Simulator**: Simulate an innings with ball-by-ball outcomes; context-aware via posterior boundary rate.
- **Bayesian**: Beta–Binomial posterior explorer; convert posterior mean to a full categorical vector.
- **Scenarios**: Load/edit/save scenario files (YAML/JSON) with base probabilities and Beta priors.

## Repository Layout

```
app.py                      # Streamlit UI (tabs)
src/
  distributions.py          # Bernoulli, Binomial, Categorical, Multinomial, Poisson, Negative Binomial
  story_examples.py         # Cricket-specific defaults and descriptions
  simulator.py              # Over & match simulators (ball-by-ball)
  bayes.py                  # Beta–Binomial posterior + categorical adjustment helper
  scenarios.py              # Scenario dataclass, import/export (YAML/JSON), player–bowler pair stats
requirements.txt
README.md
```

---

## Discrete Distributions — with Cricket Intuition

Below, \(X\) is a random variable, \( \mathbb{E}[X] \) is the mean, and \( \mathrm{Var}(X) \) is the variance.

### 1) Bernoulli \((p)\)

- **Story:** One ball, “success” if it’s a **boundary** (4 or 6), otherwise failure.  
- **Support:** \(x \in \{0,1\}\)  
- **PMF:** \( \Pr(X=x) = p^x (1-p)^{1-x} \)  
- **Mean/Var:** \( \mathbb{E}[X]=p \), \( \mathrm{Var}(X)=p(1-p) \)  
- **When to use:** Single yes/no event per ball (boundary? wicket? wide?)  
- **Cricket example:** \(p=0.22\) means a 22% chance that the next ball is a boundary.

### 2) Binomial \((n,p)\)

- **Story:** Over of \(n=6\) balls; count **how many boundaries** occur.  
- **Support:** \(k \in \{0,1,\dots,n\}\)  
- **PMF:** \( \Pr(X=k) = \binom{n}{k} p^k (1-p)^{n-k} \)  
- **Mean/Var:** \( \mathbb{E}[X]=np \), \( \mathrm{Var}(X)=np(1-p) \)  
- **When to use:** Fixed number of independent balls with same boundary probability.  
- **Cricket example:** With \(p=0.22\), expected boundaries per over \(=6\times0.22=1.32\).

> **Bayesian conjugate:** Binomial likelihood + **Beta prior** \(\to\) **Beta posterior** on \(p\).

### 3) Categorical \((\boldsymbol{\pi})\)

- **Story:** One ball with multiple outcomes \(\{\text{·}, 1, 2, 3, 4, 6, W\}\).  
- **Params:** \( \boldsymbol{\pi}=(\pi_1,\ldots,\pi_K), \sum \pi_k = 1, \pi_k \ge 0 \)  
- **PMF:** \( \Pr(X=k) = \pi_k \)  
- **Mean (index):** \( \sum k \,\pi_k \) (index mean; often less interpretable directly)  
- **When to use:** Single event with **>2** mutually exclusive outcomes.  
- **Cricket example:** Base probabilities might be  
  \([0.32, 0.40, 0.06, 0.02, 0.12, 0.05, 0.03]\) for \([·,1,2,3,4,6,W]\).

> **Bayesian conjugate:** Categorical + **Dirichlet prior** \(\to\) **Dirichlet posterior** over \(\boldsymbol{\pi}\).

### 4) Multinomial \((n,\boldsymbol{\pi})\)

- **Story:** Over of \(n\) balls, each ball is Categorical; count outcomes per category.  
- **PMF:**  
  \( \Pr(\mathbf{X}=\mathbf{x}) = \dfrac{n!}{x_1!\cdots x_K!}\prod_{k=1}^K \pi_k^{x_k} \), \(\sum x_k = n\)  
- **Mean/Cov:** \( \mathbb{E}[X_k]=n\pi_k \);  
  \(\mathrm{Cov}(X_i,X_j)=
  \begin{cases}
  n\pi_k(1-\pi_k),&i=j\\
  -n\pi_i\pi_j,&i\ne j
  \end{cases}\)  
- **When to use:** Counts across multiple outcomes over a fixed number of trials (balls).  
- **Cricket example:** Expected \# of 4s in an over is \(6\pi_{\text{4}}\), etc.

> **Bayesian conjugate:** Multinomial + **Dirichlet prior** \(\to\) **Dirichlet posterior**.

### 5) Poisson \((\lambda)\)

- **Story:** Count of events in a **time/space interval**—e.g., **wides+no-balls per over**.  
- **Support:** \(k \in \{0,1,2,\dots\}\)  
- **PMF:** \( \Pr(X=k) = e^{-\lambda}\dfrac{\lambda^k}{k!} \)  
- **Mean/Var:** \( \mathbb{E}[X]=\lambda \), \( \mathrm{Var}(X)=\lambda \)  
- **When to use:** Events occur **independently** at a constant rate.  
- **Cricket example:** \(\lambda=0.8\) ⇒ ~0.8 incidental events per over on average.

> **Bayesian conjugate:** Poisson likelihood + **Gamma prior** on \(\lambda\) \(\to\) **Gamma posterior**.

### 6) Negative Binomial (Pascal form: \(r,p\), counts “failures before \(r\) successes”)

- **Story:** Balls until the \(r\)-th **boundary** occurs; count non-boundaries before the \(r\)-th boundary.  
- **Support:** \(k\in\{0,1,2,\dots\}\) (failures before \(r\) successes)  
- **PMF (SciPy’s `nbinom`):** \( \Pr(X=k)=\binom{k+r-1}{k}(1-p)^k p^r \)  
- **Mean/Var:** \( \mathbb{E}[X]=r\dfrac{1-p}{p} \), \( \mathrm{Var}(X)=r\dfrac{1-p}{p^2} \)  
- **When to use:** Over-dispersion (variance > mean), or “wait time” until \(r\) boundaries.  
- **Cricket example:** With \(r=2, p=0.22\), expected non-boundary balls before the 2nd boundary ≈ \(2(1-0.22)/0.22\).

> **Mixture view:** A **Poisson–Gamma** mixture yields a Negative Binomial (handy for over-dispersion).

---

## Bayesian Layer (used in the app)

### Beta–Binomial for Boundary Rate

We treat “boundary on a ball” as **Bernoulli** with unknown rate \(p_B\).  
- **Prior:** \( p_B \sim \mathrm{Beta}(\alpha,\beta) \)  
- **Data:** \( s \) boundaries in \( n \) balls  
- **Posterior:** \( p_B \mid s,n \sim \mathrm{Beta}(\alpha+s,\ \beta+n-s) \)  
- **Mean & 95% CI:** shown in the **Bayesian** tab and in the **Match Simulator** for a given player–bowler pair.

### Turning Posterior into Full Ball Outcome Vector

We adjust the categorical vector \(\boldsymbol{\pi}\) by:
1. Computing posterior mean \( \hat{p}_B \) for boundary mass (4 or 6 combined).
2. **Scaling non-boundary** categories proportionally to sum to \(1-\hat{p}_B\).
3. **Splitting \(\hat{p}_B\)** between \(\{4,6\}\) in the same ratio as the base scenario.

This preserves the scenario’s **shape** while reflecting updated boundary strength.

---

## Player vs Bowler Context

- We track per-pair counts **(successes = boundaries, trials = balls)** in memory.
- Each simulation uses the pair’s **posterior mean** \( \hat{p}_B \) to adjust the outcome probabilities **before** simulating.
- Add observations in “Match Simulator” → table shows the cumulative stats.

---

## Scenarios: YAML / JSON

Define base probabilities and priors at the scenario level.

**YAML example:**
```yaml
scenarios:
  - name: Baseline
    probs: [0.32, 0.40, 0.06, 0.02, 0.12, 0.05, 0.03]   # [·,1,2,3,4,6,W] must sum > 0 (we’ll normalize)
    alpha_prior: 2.0
    beta_prior: 8.0
  - name: Powerplay Aggressive
    probs: [0.24, 0.33, 0.07, 0.03, 0.18, 0.12, 0.03]
    alpha_prior: 3.5
    beta_prior: 6.5
```

Load these from the **Scenarios** tab; edit and export back to YAML or JSON.

---

## When to Use Which (Cheat Sheet)

| Goal | Distribution | Typical Cricket Use |
|---|---|---|
| Single yes/no on a ball | **Bernoulli** | boundary? wicket? wide? |
| #successes in fixed #balls | **Binomial** | #boundaries in an over |
| Single ball, many outcomes | **Categorical** | {·,1,2,3,4,6,W} |
| Counts across outcomes in n balls | **Multinomial** | over breakdown: how many 1s, 4s, wickets… |
| Counts in a time interval | **Poisson** | wides+no-balls per over |
| Over-dispersed counts / wait till r-th success | **Negative Binomial** | balls until r-th boundary; extra-variance scoring |

---

## Common Pitfalls & Tips

- **Negative Binomial parameterization:** SciPy’s `nbinom(r,p)` returns **failures before r successes** with success-prob \(p\). If you want “trials until r successes,” add \(r\).
- **Poisson rate vs mean:** \(\lambda\) is **both** mean and variance; if variance ≫ mean, consider Negative Binomial.
- **Probabilities must sum to 1:** The app normalizes your vector; still, aim for sensible inputs.
- **Independence assumptions:** Binomial/Poisson assume independence and stationarity; real cricket has context (overs, bowlers, field settings). Our **Bayesian context** helps but is still a simplification.
- **Small-sample caution:** Beta–Binomial posteriors can be **wide** with few balls—use sensible priors (e.g., Beta(2,8) ≈ mean 0.2).

---

## Extending the App

- **Dirichlet–Multinomial:** Bayesian updates for the **full** outcome vector (not just boundary mass).
- **Hierarchical models:** Share strength across players and bowlers (random effects).
- **Context features:** Separate scenarios for **powerplay**, **middle overs**, **death**, pitch types, etc.
- **Persistence:** Save/load player–bowler stats to file or database.

---

## Troubleshooting

- **Streamlit duplicate widget IDs:** If you duplicate a **label** for the same widget type in multiple places, add a unique `key="..."`. The provided `app.py` already includes unique keys for all widgets.
- **Nothing pushes to GitHub (`src refspec main does not match any`):** Make an initial commit first:  
  `git add -A && git commit -m "initial" && git branch -M main && git push -u origin main`

---