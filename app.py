# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.distributions import (
    BernoulliDist, BinomialDist, CategoricalDist, MultinomialDist,
    PoissonDist, NegativeBinomialDist
)
from src.story_examples import (
    bernoulli_cricket_example, binomial_cricket_example,
    categorical_cricket_example, multinomial_cricket_example,
    poisson_cricket_example, negbin_cricket_example
)
from src.simulator import simulate_over, simulate_match
from src.bayes import BetaBinomialPosterior, adjust_categorical_with_boundary
from src.scenarios import Scenario, PairStats, load_scenarios_text, dump_scenarios_text

st.set_page_config(page_title="Bayesian Cricket — Discrete Distributions", layout="wide")

# ---------- constants ----------
CATS = ["·", "1", "2", "3", "4", "6", "W"]
BOUNDARY_IDX = [CATS.index("4"), CATS.index("6")]  # success = 4 or 6

# ---------- session state ----------
if "pair_stats" not in st.session_state:
    st.session_state.pair_stats = PairStats()

if "scenarios" not in st.session_state:
    # default/baseline scenario
    st.session_state.scenarios = [
        Scenario(
            name="Baseline",
            probs=[0.32, 0.40, 0.06, 0.02, 0.12, 0.05, 0.03],
            alpha_prior=2.0,
            beta_prior=8.0
        )
    ]

st.title("Bayesian Cricket: Discrete Distributions Playground 🏏")
st.caption("Now with Bayesian updates, player–bowler context, and scenario persistence.")

tabs = st.tabs(["Learn", "Playground", "Match Simulator", "Bayesian", "Scenarios"])

# ----------------------- Learn Tab -----------------------
with tabs[0]:
    st.header("Learn the Distributions (Cricket Intuition)")

    with st.expander("Bernoulli"):
        info = bernoulli_cricket_example()
        p = st.slider("Boundary probability p", 0.0, 1.0, float(info["default_p"]), 0.01, key="learn_bern_p")
        dist = BernoulliDist(p)
        st.write(info["description"])
        st.markdown(f"**Mean:** {dist.mean:.3f}, **Var:** {dist.var:.3f}")
        ks = [0, 1]
        pmf_vals = [dist.pmf(k) for k in ks]
        fig, ax = plt.subplots()
        ax.bar([info["k_labels"][k] for k in ks], pmf_vals)
        ax.set_ylabel("PMF")
        st.pyplot(fig)

    with st.expander("Binomial"):
        info = binomial_cricket_example()
        n = st.number_input("Balls (n)", min_value=1, max_value=60, value=int(info["default_n"]), key="learn_binom_n")
        p = st.slider("Boundary probability p", 0.0, 1.0, float(info["default_p"]), 0.01, key="learn_binom_p")
        dist = BinomialDist(n, p)
        st.write(info["description"])
        st.markdown(f"**Mean:** {dist.mean:.3f}, **Var:** {dist.var:.3f}")
        ks = np.arange(0, n + 1)
        fig, ax = plt.subplots()
        ax.bar(ks, [dist.pmf(k) for k in ks])
        ax.set_xlabel("Number of boundaries")
        ax.set_ylabel("PMF")
        st.pyplot(fig)

    with st.expander("Categorical"):
        info = categorical_cricket_example()
        cols = st.columns(len(info["categories"]))
        p_vals = []
        for i, cat in enumerate(info["categories"]):
            with cols[i]:
                p_vals.append(st.number_input(f"{cat}", min_value=0.0, max_value=1.0,
                                              value=float(info["default_probs"][i]), step=0.01,
                                              key=f"learn_cat_{i}"))
        p_arr = np.array(p_vals)
        p_arr = p_arr / p_arr.sum() if p_arr.sum() > 0 else info["default_probs"]
        dist = CategoricalDist(p_arr)
        st.write(info["description"])
        st.markdown(f"**Mean (category index):** {dist.mean:.3f}")
        fig, ax = plt.subplots()
        ax.bar(info["categories"], p_arr)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

    with st.expander("Multinomial"):
        info = multinomial_cricket_example()
        n = st.number_input("Balls (n)", min_value=1, max_value=120, value=int(info["default_n"]), key="learn_multi_n")
        cols = st.columns(len(info["categories"]))
        p_vals = []
        for i, cat in enumerate(info["categories"]):
            with cols[i]:
                p_vals.append(st.number_input(f"{cat}", min_value=0.0, max_value=1.0,
                                              value=float(info["default_probs"][i]), step=0.01,
                                              key=f"learn_multi_{i}"))
        p_arr = np.array(p_vals)
        p_arr = p_arr / p_arr.sum() if p_arr.sum() > 0 else info["default_probs"]
        dist = MultinomialDist(n, p_arr)
        st.write(info["description"])
        st.markdown("**Mean counts:** " + ", ".join([f"{c}:{m:.2f}" for c, m in zip(info["categories"], dist.mean)]))
        fig, ax = plt.subplots()
        ax.bar(info["categories"], dist.mean)
        ax.set_ylabel("Expected count")
        st.pyplot(fig)

    with st.expander("Poisson"):
        info = poisson_cricket_example()
        lam = st.slider("λ (expected incidents per over)", 0.0, 5.0, float(info["default_lambda"]), 0.1, key="learn_poiss_lam")
        dist = PoissonDist(lam)
        st.write(info["description"])
        st.markdown(f"**Mean=Var:** {dist.mean:.2f}")
        ks = np.arange(0, 10)
        fig, ax = plt.subplots()
        ax.bar(ks, [dist.pmf(k) for k in ks])
        ax.set_xlabel("Incidents in an over")
        ax.set_ylabel("PMF")
        st.pyplot(fig)

    with st.expander("Negative Binomial"):
        info = negbin_cricket_example()
        r = st.number_input("r (successes to stop)", min_value=1, max_value=50, value=int(info["default_r"]), key="learn_nb_r")
        p = st.slider("p (success probability per ball)", 0.0, 1.0, float(info["default_p"]), 0.01, key="learn_nb_p")
        dist = NegativeBinomialDist(r, p)
        st.write(info["description"])
        st.markdown(f"**Mean:** {dist.mean:.2f}, **Var:** {dist.var:.2f}")
        ks = np.arange(0, 20)
        fig, ax = plt.subplots()
        ax.bar(ks, [dist.pmf(k) for k in ks])
        ax.set_xlabel("Failures before r-th success")
        ax.set_ylabel("PMF")
        st.pyplot(fig)

# ----------------------- Playground Tab -----------------------
with tabs[1]:
    st.header("Playground: Sample and See")
    sub = st.selectbox("Pick a distribution", ["Bernoulli", "Binomial", "Categorical", "Multinomial", "Poisson", "NegativeBinomial"], key="play_subselect")
    n_samples = st.number_input("Number of samples", 1, 10000, 1000, key="play_n_samples")
    rng = np.random.default_rng(42)

    if sub == "Bernoulli":
        p = st.slider("p", 0.0, 1.0, 0.22, 0.01, key="play_bern_p")
        dist = BernoulliDist(p)
        samples = dist.sample(size=n_samples, rng=rng)
        vals, counts = np.unique(samples, return_counts=True)
        fig, ax = plt.subplots()
        ax.bar([str(int(v)) for v in vals], counts / n_samples)
        ax.set_ylabel("Empirical probability")
        st.pyplot(fig)

    elif sub == "Binomial":
        n = st.number_input("n", 1, 60, 6, key="play_binom_n")
        p = st.slider("p", 0.0, 1.0, 0.22, 0.01, key="play_binom_p")
        dist = BinomialDist(n, p)
        samples = dist.sample(size=n_samples, rng=rng)
        vals, counts = np.unique(samples, return_counts=True)
        fig, ax = plt.subplots()
        ax.bar(vals, counts / n_samples)
        ax.set_xlabel("k successes")
        ax.set_ylabel("Empirical probability")
        st.pyplot(fig)

    elif sub == "Categorical":
        cols = st.columns(len(CATS))
        p_vals = []
        base = np.array([0.32, 0.40, 0.06, 0.02, 0.12, 0.05, 0.03])
        for i, c in enumerate(CATS):
            with cols[i]:
                p_vals.append(st.number_input(c, 0.0, 1.0, float(base[i]), 0.01, key=f"play_cat_{i}"))
        p_arr = np.array(p_vals)
        p_arr = p_arr / p_arr.sum() if p_arr.sum() > 0 else base
        dist = CategoricalDist(p_arr)
        samples = dist.sample(size=n_samples, rng=rng)
        vals, counts = np.unique(samples, return_counts=True)
        fig, ax = plt.subplots()
        ax.bar([CATS[int(v)] for v in vals], counts / n_samples)
        ax.set_ylabel("Empirical probability")
        st.pyplot(fig)

    elif sub == "Multinomial":
        n = st.number_input("n (trials)", 1, 120, 6, key="play_multi_n")
        cols = st.columns(len(CATS))
        p_vals = []
        base = np.array([0.32, 0.40, 0.06, 0.02, 0.12, 0.05, 0.03])
        for i, c in enumerate(CATS):
            with cols[i]:
                p_vals.append(st.number_input(c, 0.0, 1.0, float(base[i]), 0.01, key=f"play_m_{i}"))
        p_arr = np.array(p_vals)
        p_arr = p_arr / p_arr.sum() if p_arr.sum() > 0 else base
        dist = MultinomialDist(n, p_arr)
        samples = dist.sample(size=n_samples, rng=rng)
        mean_counts = samples.mean(axis=0)
        fig, ax = plt.subplots()
        ax.bar(CATS, mean_counts)
        ax.set_ylabel("Average counts across samples")
        st.pyplot(fig)

    elif sub == "Poisson":
        lam = st.slider("λ", 0.0, 5.0, 0.8, 0.1, key="play_poiss_lam")
        dist = PoissonDist(lam)
        samples = dist.sample(size=n_samples, rng=rng)
        vals, counts = np.unique(samples, return_counts=True)
        fig, ax = plt.subplots()
        ax.bar(vals, counts / n_samples)
        ax.set_xlabel("Count in interval")
        ax.set_ylabel("Empirical probability")
        st.pyplot(fig)

    else:  # NegativeBinomial
        r = st.number_input("r", 1, 50, 2, key="play_nb_r")
        p = st.slider("p", 0.0, 1.0, 0.22, 0.01, key="play_nb_p")
        dist = NegativeBinomialDist(r, p)
        samples = dist.sample(size=n_samples, rng=rng)
        vals, counts = np.unique(samples, return_counts=True)
        fig, ax = plt.subplots()
        ax.bar(vals, counts / n_samples)
        ax.set_xlabel("Failures before r-th success")
        ax.set_ylabel("Empirical probability")
        st.pyplot(fig)

# ----------------------- Match Simulator Tab (context-aware) -----------------------
with tabs[2]:
    st.header("Match Simulator (Ball-by-Ball, Context-Aware)")

    # Scenario picker
    scen_names = [s.name for s in st.session_state.scenarios]
    scen_idx = st.selectbox("Scenario", list(range(len(scen_names))), format_func=lambda i: scen_names[i], key="sim_scen_idx")
    scenario = st.session_state.scenarios[scen_idx]

    # Player vs Bowler
    st.subheader("Player vs Bowler context")
    c1, c2 = st.columns(2)
    with c1:
        batter = st.text_input("Batter", "Batter A", key="sim_batter")
    with c2:
        bowler = st.text_input("Bowler", "Bowler X", key="sim_bowler")

    # Record observations
    st.markdown("Record observations for this pair (success = boundary: 4 or 6).")
    oc1, oc2, oc3 = st.columns([1, 1, 1])
    with oc1:
        new_successes = st.number_input("Boundary successes", 0, 36, 0, key="sim_new_successes")
    with oc2:
        new_trials = st.number_input("Balls (trials)", 0, 300, 0, key="sim_new_trials")
    with oc3:
        if st.button("Add to Pair Stats", key="sim_add_pair"):
            st.session_state.pair_stats.record(batter, bowler, int(new_successes), int(new_trials))
            st.success("Recorded.")

    # Current pair stats and posterior
    s_pair, t_pair = st.session_state.pair_stats.get(batter, bowler)
    st.info(f"Current stats for **{batter} vs {bowler}** — successes={s_pair}, trials={t_pair}")

    bb = BetaBinomialPosterior(scenario.alpha_prior, scenario.beta_prior)
    a_post, b_post = bb.update(s_pair, t_pair)
    p_mean = bb.mean(a_post, b_post)
    lo, hi = bb.ci(a_post, b_post, level=0.95)

    st.markdown(f"**Posterior boundary rate (mean)**: {p_mean:.3f}  \n"
                f"**95% CI**: [{lo:.3f}, {hi:.3f}]  \n"
                f"_Prior_ Beta({scenario.alpha_prior:.1f}, {scenario.beta_prior:.1f}) → _Posterior_ Beta({a_post:.1f}, {b_post:.1f})")

    base = np.array(scenario.probs, dtype=float)
    p_adj = adjust_categorical_with_boundary(base, p_mean, BOUNDARY_IDX)

    # Compare base vs posterior-adjusted categorical probabilities
    fig, ax = plt.subplots()
    width = 0.4
    x = np.arange(len(CATS))
    ax.bar(x - width/2, base / base.sum(), width, label="Base")
    ax.bar(x + width/2, p_adj, width, label="Posterior-adjusted")
    ax.set_xticks(x, CATS)
    ax.set_ylabel("Probability")
    ax.legend()
    st.pyplot(fig)

    overs = st.number_input("Overs", 1, 50, 5, key="sim_overs")
    if st.button("Simulate innings (context-aware)", key="sim_run"):
        df = simulate_match(int(overs), p_adj)
        st.dataframe(df, use_container_width=True)
        final_runs = int(df['total_runs'].iloc[-1])
        final_wkts = int(df['total_wkts'].iloc[-1])
        balls = int(df['ball_global'].iloc[-1])
        st.success(f"Final score: {final_runs}/{final_wkts} in {balls//6}.{balls%6} overs")

        # Runs per over
        df["over_label"] = df["over"]
        runs_per_over = df.groupby("over_label")["runs_this_ball"].sum()
        fig2, ax2 = plt.subplots()
        ax2.bar(runs_per_over.index.astype(str), runs_per_over.values)
        ax2.set_xlabel("Over")
        ax2.set_ylabel("Runs")
        st.pyplot(fig2)

    st.subheader("Current Player–Bowler Table")
    rows = st.session_state.pair_stats.table()
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ----------------------- Bayesian Tab (manual what-if) -----------------------
with tabs[3]:
    st.header("Bayesian: Beta–Binomial Posterior for Boundary Rate")
    c1, c2, c3 = st.columns(3)
    with c1:
        alpha_prior = st.number_input("Prior alpha (Beta)", 0.1, 1000.0, 2.0, step=0.1, key="bayes_alpha")
    with c2:
        beta_prior = st.number_input("Prior beta (Beta)", 0.1, 1000.0, 8.0, step=0.1, key="bayes_beta")
    with c3:
        trials = st.number_input("Observed balls", 0, 10000, 30, key="bayes_trials")

    successes = st.number_input("Observed boundaries (4 or 6)", 0, 10000, 6, key="bayes_successes")
    bb = BetaBinomialPosterior(alpha_prior, beta_prior)
    a_post, b_post = bb.update(int(successes), int(trials))
    mean_post = bb.mean(a_post, b_post)
    lo, hi = bb.ci(a_post, b_post)

    st.markdown(f"**Posterior** Beta({a_post:.1f}, {b_post:.1f})  —  mean={mean_post:.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")

    # Adjust any base categorical vector with this posterior
    st.subheader("Adjust a categorical vector with posterior boundary rate")
    base_str = st.text_input("Comma-separated base probs for [·,1,2,3,4,6,W]", "0.32,0.40,0.06,0.02,0.12,0.05,0.03", key="bayes_base_str")
    try:
        base = np.array([float(x.strip()) for x in base_str.split(",")])
        p_adj = adjust_categorical_with_boundary(base, mean_post, BOUNDARY_IDX)
        fig, ax = plt.subplots()
        x = np.arange(len(CATS))
        ax.bar(x - 0.4/2, base / base.sum(), 0.4, label="Base")
        ax.bar(x + 0.4/2, p_adj, 0.4, label="Posterior-adjusted")
        ax.set_xticks(x, CATS)
        ax.legend()
        st.pyplot(fig)
        st.code("Adjusted probs: " + ", ".join([f"{v:.3f}" for v in p_adj]))
    except Exception as e:
        st.warning(f"Parse error: {e}")

# ----------------------- Scenarios Tab (YAML/JSON persistence) -----------------------
with tabs[4]:
    st.header("Scenarios: load / edit / save")

    # Show current scenarios
    st.subheader("In-memory scenarios")
    for i, s in enumerate(st.session_state.scenarios):
        probs_norm = np.array(s.probs) / np.sum(s.probs)
        st.markdown(f"- **{i}** — **{s.name}** | prior Beta({s.alpha_prior},{s.beta_prior}) | probs={', '.join([f'{x:.2f}' for x in probs_norm])}")

    st.divider()

    # Import
    st.subheader("Import scenarios (YAML or JSON)")
    file = st.file_uploader("Upload .yaml/.yml or .json", type=["yaml", "yml", "json"], key="sc_upload")
    if file is not None:
        data_bytes = file.read()
        is_yaml = file.name.lower().endswith((".yaml", ".yml"))
        try:
            scenarios = load_scenarios_text(data_bytes.decode("utf-8"), is_yaml=is_yaml)
            st.session_state.scenarios = scenarios
            st.success(f"Loaded {len(scenarios)} scenario(s).")
        except Exception as e:
            st.error(f"Failed to load: {e}")

    # Edit one scenario
    st.subheader("Edit scenario")
    if st.session_state.scenarios:
        idx = st.number_input("Scenario index", 0, len(st.session_state.scenarios)-1, 0, key="sc_idx")
        s = st.session_state.scenarios[int(idx)]
        name = st.text_input("Name", s.name, key="sc_name")
        alpha = st.number_input("Prior alpha", 0.1, 1000.0, float(s.alpha_prior), key="sc_alpha")
        beta = st.number_input("Prior beta", 0.1, 1000.0, float(s.beta_prior), key="sc_beta")
        cols = st.columns(len(CATS))
        probs = []
        base = np.array(s.probs, dtype=float)
        for i, c in enumerate(CATS):
            with cols[i]:
                probs.append(st.number_input(c, 0.0, 1.0, float(base[i]), 0.01, key=f"sc_{i}"))
        if st.button("Apply edits", key="sc_apply"):
            s.name = name
            s.alpha_prior = float(alpha)
            s.beta_prior = float(beta)
            s.probs = [float(x) for x in probs]
            st.success("Scenario updated in memory.")

    # Export
    st.subheader("Export scenarios")
    fmt = st.selectbox("Format", ["YAML", "JSON"], key="sc_fmt")
    is_yaml = fmt == "YAML"
    text = dump_scenarios_text(st.session_state.scenarios, is_yaml=is_yaml)
    st.download_button("Download", data=text, file_name=f"scenarios.{ 'yaml' if is_yaml else 'json'}", mime="text/plain", key="sc_download")
