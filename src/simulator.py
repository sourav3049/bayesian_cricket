# src/simulator.py
import numpy as np
import pandas as pd

CATS = ["·", "1", "2", "3", "4", "6", "W"]

def simulate_over(probs, rng=None):
    """
    Simulate one over (6 balls) given category probabilities over CATS.
    Returns a DataFrame with ball-by-ball outcomes and cumulative score/wickets.
    """
    rng = rng or np.random.default_rng()
    probs = np.asarray(probs) / np.sum(probs)
    outcomes_idx = rng.choice(len(CATS), size=6, p=probs)
    outcomes = [CATS[i] for i in outcomes_idx]

    runs_map = {"·": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "W": 0}
    runs = [runs_map[o] for o in outcomes]
    wickets = [1 if o == "W" else 0 for o in outcomes]

    df = pd.DataFrame({
        "ball": np.arange(1, 7),
        "outcome": outcomes,
        "runs": runs,
        "wicket": wickets
    })
    df["cum_runs"] = df["runs"].cumsum()
    df["cum_wkts"] = df["wicket"].cumsum()
    return df

def simulate_match(overs, probs, rng=None):
    """
    Simulate a T-overs innings with given ball outcome probabilities.
    Stops if 10 wickets fall.
    """
    rng = rng or np.random.default_rng()
    all_rows = []
    total_wkts = 0
    total_runs = 0
    ball_global = 0

    for over_idx in range(1, overs + 1):
        over_df = simulate_over(probs, rng=rng)
        for _, row in over_df.iterrows():
            ball_global += 1
            total_runs += int(row["runs"])
            total_wkts += int(row["wicket"])
            all_rows.append({
                "over": over_idx,
                "ball_in_over": row["ball"],
                "ball_global": ball_global,
                "outcome": row["outcome"],
                "runs_this_ball": row["runs"],
                "total_runs": total_runs,
                "total_wkts": total_wkts
            })
            if total_wkts >= 10:
                return pd.DataFrame(all_rows)

    return pd.DataFrame(all_rows)
