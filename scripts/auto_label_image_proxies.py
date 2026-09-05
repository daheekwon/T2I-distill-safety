#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.image_proxies import auto_label_image_proxies


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill conservative auto-proxy labels for image quality, refusal, and MemBench copy similarity.")
    parser.add_argument("--labels", default="data/mats20/label_template_existing_images.csv")
    parser.add_argument("--output-labels", default="data/mats20/auto_proxy_labels.csv")
    parser.add_argument("--output-updates", default="data/mats20/auto_proxy_updates.csv")
    parser.add_argument("--features-csv", default="results/mats20/auto_proxy/image_features.csv")
    parser.add_argument("--summary-json", default="results/mats20/auto_proxy/auto_proxy_summary.json")
    parser.add_argument("--copy-scores-csv", default="results/mats20/auto_proxy/membench_copy_scores.csv")
    parser.add_argument("--reference-index", default="data/mats20/membench_reference_index.json")
    parser.add_argument("--sscd-checkpoint", default="/DATA0/dahee/.cache/torch/hub/checkpoints/sscd_disc_mixup.torchscript.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    reference_index = Path(args.reference_index) if args.reference_index else None
    sscd_checkpoint = Path(args.sscd_checkpoint) if args.sscd_checkpoint else None
    result = auto_label_image_proxies(
        labels_path=Path(args.labels),
        output_labels_path=Path(args.output_labels),
        output_updates_path=Path(args.output_updates),
        feature_csv_path=Path(args.features_csv),
        summary_json_path=Path(args.summary_json),
        copy_scores_csv_path=Path(args.copy_scores_csv) if args.copy_scores_csv else None,
        reference_index_path=reference_index,
        sscd_checkpoint=sscd_checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
