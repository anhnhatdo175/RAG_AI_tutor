from collections import namedtuple

import pandas as pd

from lmsrag.sim.policies import ProposedPolicy

Obs = namedtuple("Obs", "t item_id correct")
CANDS = pd.DataFrame({"item_id": ["e", "m", "h"], "skill_tag": "S", "difficulty": [0.1, 0.5, 0.9]})


def test_chooses_harder_after_right_answers_and_easier_after_wrong():
    p = ProposedPolicy(theta0=0.0, k0=1.0, k_min=0.5, scale=1.0, repeat_penalty=0.0)
    first = p.choose([], CANDS)
    up = p.choose([Obs(i, first, 1) for i in range(8)], CANDS)
    down = p.choose([Obs(i, first, 0) for i in range(8)], CANDS)
    order = {"e": 0, "m": 1, "h": 2}
    assert order[up] >= order[first] >= order[down] and order[up] > order[down]


def test_from_config_and_unknown_parameter():
    p = ProposedPolicy.from_config({"closed_loop": {"proposed_params": {"scale": 0.5}}})
    assert p.p["scale"] == 0.5 and p.choose([], CANDS) in set(CANDS["item_id"])
    try:
        ProposedPolicy(bogus=1)
    except ValueError:
        return
    raise AssertionError("unknown parameter accepted")
