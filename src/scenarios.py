# src/scenarios.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
import json
import yaml

@dataclass
class Scenario:
    name: str
    probs: List[float]         # [·,1,2,3,4,6,W]
    alpha_prior: float = 2.0   # Beta prior alpha for boundary rate
    beta_prior: float = 8.0    # Beta prior beta for boundary rate

    def as_dict(self):
        d = asdict(self)
        d["probs"] = [float(x) for x in self.probs]
        return d

def load_scenarios_text(text: str, is_yaml: bool = True) -> List[Scenario]:
    data = yaml.safe_load(text) if is_yaml else json.loads(text)
    scenarios = []
    for item in data.get("scenarios", []):
        scenarios.append(Scenario(**item))
    return scenarios

def dump_scenarios_text(scenarios: List[Scenario], is_yaml: bool = True) -> str:
    payload = {"scenarios": [s.as_dict() for s in scenarios]}
    return yaml.safe_dump(payload, sort_keys=False) if is_yaml else json.dumps(payload, indent=2)

# Player–Bowler stats map: {(batter, bowler): (successes, trials)}
class PairStats:
    def __init__(self):
        self._map: Dict[Tuple[str, str], Tuple[int, int]] = {}

    def record(self, batter: str, bowler: str, successes: int, trials: int):
        key = (batter.strip(), bowler.strip())
        s0, t0 = self._map.get(key, (0, 0))
        self._map[key] = (s0 + int(successes), t0 + int(trials))

    def get(self, batter: str, bowler: str):
        return self._map.get((batter.strip(), bowler.strip()), (0, 0))

    def table(self):
        rows = []
        for (ba, bo), (s, t) in sorted(self._map.items()):
            rows.append({"batter": ba, "bowler": bo, "boundary_successes": s, "balls": t})
        return rows
