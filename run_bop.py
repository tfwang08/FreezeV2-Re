from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from freezev2.bop import (
    BOP_TOOLKIT_COMMIT,
    REFERENCE_SUBMISSIONS,
    download_reference_submission,
    evaluate_reference,
    prepare_bop_dataset,
)
from freezev2.gedi_bridge import GEDI_REPO_COMMIT, GEDI_SCALES, GediExtractor


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

    gedi = subparsers.add_parser(
        "extract-query-gedi",
        help="Extract two-scale GeDi descriptors for an onboarded CAD object",
    )
    gedi.add_argument("--dataset", required=True, choices=sorted(REFERENCE_SUBMISSIONS))
    gedi.add_argument("--obj-id", type=int, required=True)
    gedi.add_argument("--bop-root", type=Path, default=Path("data/bop"))
    gedi.add_argument("--onboarding-cache", type=Path)
    gedi.add_argument("--gedi-root", type=Path, default=Path("external/gedi"))
    gedi.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("external/gedi/data/chkpts/3dmatch/chkpt.tar"),
    )
    gedi.add_argument("--output", type=Path)
    gedi.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "prepare-data":
        print(prepare_bop_dataset(args.dataset, args.bop_root))
        return

    if args.command == "download-reference":
        print(download_reference_submission(args.dataset, args.output_dir))
        return

    if args.command == "extract-query-gedi":
        if args.obj_id <= 0:
            raise ValueError("--obj-id must be positive")

        models_info_path = (
            args.bop_root / args.dataset / "models" / "models_info.json"
        )
        models_info = json.loads(models_info_path.read_text())
        object_info = models_info.get(str(args.obj_id))
        if object_info is None or "diameter" not in object_info:
            raise KeyError(
                f"object {args.obj_id} has no diameter in {models_info_path}"
            )
        diameter = float(object_info["diameter"])
        if diameter <= 0:
            raise ValueError("object diameter must be positive")

        cache_path = args.onboarding_cache or (
            Path("outputs/onboard")
            / f"{args.dataset}_obj_{args.obj_id:06d}"
            / "onboarding.npz"
        )
        with np.load(cache_path, allow_pickle=False) as cache:
            if "query_points" not in cache:
                raise KeyError(f"query_points missing from {cache_path}")
            query_points = np.asarray(cache["query_points"], dtype=np.float32)

        extractor = GediExtractor(
            checkpoint=args.checkpoint,
            gedi_root=args.gedi_root,
            seed=args.seed,
        )
        geometric = extractor.encode(
            query_points,
            query_points,
            object_diameter=diameter,
        )
        if geometric.shape != (len(query_points), 64):
            raise RuntimeError(f"unexpected fused GeDi shape: {geometric.shape}")

        radii = np.asarray(
            [scale * diameter for scale in GEDI_SCALES],
            dtype=np.float32,
        )
        output = args.output or (
            Path("outputs/features")
            / f"{args.dataset}_obj_{args.obj_id:06d}_gedi.npz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            query_points=query_points,
            geometric_features_30=geometric[:, :32],
            geometric_features_40=geometric[:, 32:],
            geometric_features=geometric,
            scales=np.asarray(GEDI_SCALES, dtype=np.float32),
            radii=radii,
            diameter=np.float32(diameter),
            seed=np.int32(args.seed),
            gedi_commit=np.array(GEDI_REPO_COMMIT),
        )
        print(json.dumps({
            "dataset": args.dataset,
            "obj_id": args.obj_id,
            "diameter": diameter,
            "query_points": list(query_points.shape),
            "geometric_features": list(geometric.shape),
            "radii": radii.tolist(),
            "finite": bool(np.isfinite(geometric).all()),
            "output": str(output),
        }, indent=2, sort_keys=True))
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
