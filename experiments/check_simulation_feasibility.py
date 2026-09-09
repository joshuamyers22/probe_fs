"""Check population endpoint headroom before fitting a frozen simulation design."""

import argparse
import itertools
import json
from pathlib import Path

from pgfs.study import write_json


def check(config):
    sim = config["simulation"]
    betas, size = sim["betas"], sim["class_size"]
    if len(betas) != sim["n_classes"]:
        raise ValueError("Class strengths must match the number of classes")
    rows = []
    for rho in sim["rho"]:
        if not 0 < rho < 1:
            raise ValueError("This noisy-proxy preflight requires 0 < rho < 1")
        reference = sum(b*b*(1-rho)/(1-rho+size*rho) for b in betas)
        allocations = []
        for counts in itertools.product(range(size+1), repeat=len(betas)):
            k = sum(counts)
            if k not in sim["candidate_k"] or k > sim["k_max"]:
                continue
            residual = sum(b*b*(1-rho)/(1-rho+r*rho) for b, r in zip(betas, counts))
            allocations.append({"counts": counts, "k": k,
                                "class_recall": sum(r > 0 for r in counts)/len(betas),
                                "excess_mse": residual-reference,
                                "within_margin": residual-reference <= sim["noninferiority_margin"]})
        feasible = [a for a in allocations if a["within_margin"]]
        if not any(a["class_recall"] == 1 for a in feasible):
            raise ValueError(f"No feasible full-coverage allocation at rho={rho}")
        if not any(0 < a["class_recall"] < 1 for a in feasible):
            raise ValueError(f"No feasible partial-coverage allocation at rho={rho}")
        rows.append({"rho": rho, "allocations": allocations,
                     "feasible_recall_values": sorted({a["class_recall"] for a in feasible})})
    return {"name": config["name"], "betas": betas,
            "interpretation": "Population Bayes excess MSE versus all signal proxies; noise cancels. This checks headroom, not finite-sample Ridge or learned-ranking performance.",
            "regimes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Feasibility passed for {len(result['regimes'])} correlation regimes")


if __name__ == "__main__":
    main()
