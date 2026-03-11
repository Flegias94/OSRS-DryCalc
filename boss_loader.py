import json

from encounters import Encounter, DropTarget, CompletionRule
from math_engine import GeometricDist, NegBinomDist


def load_encounters(path="bosses.json"):
    with open(path) as f:
        data = json.load(f)

    encounters = {}

    for key, enc in data.items():
        targets = []

        for t in enc["targets"]:
            if t["distribution"] == "geometric":
                dist = GeometricDist(p=t["p"])
            elif t["distribution"] == "negbinom":
                dist = NegBinomDist(r=t["required_count"], p=t["p"])
            else:
                raise ValueError("Unknown distribution")

            targets.append(
                DropTarget(
                    key=t["key"],
                    display_name=t["display_name"],
                    distribution=dist,
                    required_count=t["required_count"],
                )
            )

        encounters[key] = Encounter(
            key=key,
            display_name=enc["display_name"],
            rule=CompletionRule(enc["rule"]),
            targets=tuple(targets),
        )

    return encounters