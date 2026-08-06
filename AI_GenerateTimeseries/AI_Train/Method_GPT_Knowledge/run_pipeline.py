from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
AI_TRAIN = HERE.parent
sys.path.insert(0, str(AI_TRAIN))

from pipeline_common import choose_operation, load_json, parser, run, wants_evaluate, wants_train  # noqa: E402


def resolve_from_config(config_path: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def validate_retrieval_contract(build_path: pathlib.Path, eval_path: pathlib.Path) -> list[str]:
    build = load_json(build_path)
    evaluate = load_json(eval_path)
    failures = []
    roots = [resolve_from_config(build_path, item) for item in build.get("dataset_roots", [])]
    if build.get("split") != "train":
        failures.append("knowledge index must use split=train")
    if len(roots) != 1 or roots[0].name != "Topo_HouseGAN":
        failures.append("knowledge index must use only canonical Topo_HouseGAN, not mixed datasets")
    eval_root = resolve_from_config(eval_path, evaluate.get("dataset_root", ""))
    if evaluate.get("split") != "test" or eval_root.name != "Topo_HouseGAN":
        failures.append("evaluation must use the canonical Topo_HouseGAN test split")
    build_output = resolve_from_config(build_path, build.get("knowledge_output_dir", ""))
    eval_knowledge = resolve_from_config(eval_path, evaluate.get("knowledge_dir", ""))
    if build_output != eval_knowledge:
        failures.append("build knowledge_output_dir and evaluation knowledge_dir do not match")
    return failures


def main() -> None:
    cli = parser("Build or evaluate GPT-Assisted Knowledge Retrieval and Geometric Transfer.")
    cli.add_argument("--config-train", type=pathlib.Path, default=HERE / "config_build.json")
    cli.add_argument("--config-test", type=pathlib.Path, default=HERE / "config_validate.json")
    args = cli.parse_args()
    build_path = args.config_train.resolve()
    eval_path = args.config_test.resolve()
    if len(sys.argv) == 1:
        action = choose_operation(
            "GPT-Assisted Knowledge Retrieval and Geometric Transfer",
            supports_smoke=False,
            retrieval=True,
        )
        if action == "exit":
            return
        if action == "runs":
            build = load_json(build_path)
            evaluate = load_json(eval_path)
            build_output = resolve_from_config(build_path, build.get("knowledge_output_dir", ""))
            eval_knowledge = resolve_from_config(eval_path, evaluate.get("knowledge_dir", ""))
            print("\nConfigured artifacts:")
            print(f"  knowledge index : {build_output} exists={build_output.exists()}")
            print(f"  evaluation input: {eval_knowledge} exists={eval_knowledge.exists()}")
            return
        args.stage = {"check": "plan", "train": "train", "evaluate": "evaluate", "all": "all"}[action]
    failures = validate_retrieval_contract(build_path, eval_path)
    print(f"[pipeline] method=GPT-Assisted Knowledge Retrieval and Geometric Transfer stage={args.stage}")
    if failures:
        for failure in failures:
            print(f"[pipeline] CONTRACT FAILURE: {failure}")
        if args.stage != "plan" and not args.dry_run:
            raise RuntimeError("retrieval pipeline contract failed; refusing a leakage-prone run")
    if wants_train(args.stage):
        run([args.python, HERE / "build_knowledge.py", "--config", build_path], cwd=HERE, dry_run=args.dry_run)
    if wants_evaluate(args.stage):
        run([args.python, HERE / "validate_gpt_knowledge.py", "--config", eval_path], cwd=HERE, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
