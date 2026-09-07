"""Preview a study, or export resolved YAMLs without starting jobs.

uv run --locked python study.py --config configs/studies/baseline.yaml
uv run --locked python study.py --config configs/studies/splits.yaml --task coco
uv run --locked python study.py --config configs/studies/splits.yaml --output runs/studies/splits
"""
import argparse

from jscc.studies import expand_study, export_study


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="study YAML")
    parser.add_argument("--task", choices=("coco", "hellaswag"), help="optional task filter")
    parser.add_argument("--output", help="new directory for manifest and resolved configs")
    args = parser.parse_args()
    plan = expand_study(args.config, args.task)
    print(f"Study {plan.name}: {len(plan.runs)} training configurations; no jobs started")
    print("index  task       experiment   seed  split                  bottleneck  FiLM")
    for index, run in enumerate(plan.runs):
        split = run.config["split"]
        where = f"{split['stack']}:{split['where']}"
        if split["where"] == "after_layer":
            where += f":{split['index']}"
        print(f"{index:5}  {run.task:10} {run.experiment:12} {run.seed:4}  {where:23} "
              f"{run.config['codec']['bottleneck_dim']:10}  {run.config['codec']['snr_film']}")
    if args.output:
        print(f"Prepared: {export_study(plan, args.output)}")


if __name__ == "__main__":
    main()
