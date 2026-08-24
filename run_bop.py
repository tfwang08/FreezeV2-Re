from __future__ import annotations

import argparse
import json
from pathlib import Path

from freezev2.bop import (
    BOP_TOOLKIT_COMMIT,
    REFERENCE_SUBMISSIONS,
    download_reference_submission,
    evaluate_reference,
    prepare_bop_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="FreeZeV2 BOP reproduction utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Download/extract the BOP19 challenge subset")
    prepare.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    prepare.add_argument("--bop-root", type=Path, default=Path("data/bop"))

    download = subparsers.add_parser("download-reference", help="Download the authors' public FreeZeV2.1 result CSV")
    download.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    download.add_argument("--output-dir", type=Path, default=Path("data/reference"))

    evaluate = subparsers.add_parser("evaluate-reference", help="Run the official BOP19 pose evaluator")
    evaluate.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    evaluate.add_argument("--bop-root", type=Path, default=Path("data/bop"))
    evaluate.add_argument("--result-csv", type=Path)
    evaluate.add_argument("--results-dir", type=Path, default=Path("data/reference"))
    evaluate.add_argument("--bop-toolkit", type=Path, default=Path("external/bop_toolkit"))
    evaluate.add_argument("--eval-root", type=Path, default=Path("outputs/bop_eval"))
    evaluate.add_argument("--num-workers", type=int, default=10)

    args = parser.parse_args()

    if args.command == "prepare-data":
        print(prepare_bop_dataset(args.dataset, args.bop_root))
        return

    if args.command == "download-reference":
        print(download_reference_submission(args.dataset, args.output_dir))
        return

    spec = REFERENCE_SUBMISSIONS[args.dataset]
    result_csv = args.result_csv or (args.results_dir / spec["filename"])
    score_path = evaluate_reference(
        dataset=args.dataset,
        bop_root=args.bop_root,
        result_csv=result_csv,
        bop_toolkit=args.bop_toolkit,
        eval_root=args.eval_root,
        num_workers=args.num_workers,
    )
    scores = json.loads(score_path.read_text())
    print(json.dumps({
        "dataset": args.dataset,
        "scores": scores,
        "expected": spec["expected"],
        "bop_toolkit_commit": BOP_TOOLKIT_COMMIT,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
