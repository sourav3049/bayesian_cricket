# src/story_examples.py
import numpy as np

# Categories for a ball outcome: dot,1,2,3,4,6,wicket
CATEGORIES = ["·", "1", "2", "3", "4", "6", "W"]

def bernoulli_cricket_example():
    """
    Bernoulli: single ball boundary? (1=yes boundary, 0=no boundary)
    """
    return {
        "description": "Is this ball a boundary? Success=boundary.",
        "k_labels": {0: "No boundary", 1: "Boundary"},
        "default_p": 0.22
    }

def binomial_cricket_example():
    """
    Binomial: number of boundaries in an over (6 balls).
    """
    return {
        "description": "How many boundaries in an over?",
        "default_n": 6,
        "default_p": 0.22
    }

def categorical_cricket_example():
    """
    Categorical: outcome of a single ball across 7 categories.
    """
    probs = np.array([0.32, 0.40, 0.06, 0.02, 0.12, 0.05, 0.03])
    probs = probs / probs.sum()
    return {
        "description": "Outcome on a single ball.",
        "categories": CATEGORIES,
        "default_probs": probs
    }

def multinomial_cricket_example():
    """
    Multinomial: outcome counts over an over (6 balls) across categories.
    """
    probs = np.array([0.32, 0.40, 0.06, 0.02, 0.12, 0.05, 0.03])
    probs = probs / probs.sum()
    return {
        "description": "Outcome breakdown over an over (6 balls).",
        "default_n": 6,
        "categories": CATEGORIES,
        "default_probs": probs
    }

def poisson_cricket_example():
    """
    Poisson: counts per over (e.g., wides+no-balls total incidents per over).
    """
    return {
        "description": "Incidental events per over (e.g., wides+no-balls).",
        "default_lambda": 0.8
    }

def negbin_cricket_example():
    """
    Negative Binomial: balls until r boundaries occur (over-dispersion friendly).
    """
    return {
        "description": "How many non-boundary balls before r-th boundary?",
        "default_r": 2,
        "default_p": 0.22
    }
