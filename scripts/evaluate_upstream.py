#!/usr/bin/env python3
"""Compare J-lens and logit-lens pass@k on upstream evaluation prompts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from jlens_panel.config import load_config
from jlens_panel.modeling import load_model_bundle
from jlens_panel.provenance import build_manifest, sha256_file, write_json_atomic
from jlens_panel.upstream_eval import best_rank, pass_at, token_variants


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("evaluate_upstream")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dirty_run.yaml")
    parser.add_argument("--lens", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    bundle = load_model_bundle(
        model_name=model_config["name"],
        revision=model_config["revision"],
        dtype=model_config["dtype"],
        device_map=model_config["device_map"],
        lens_path=args.lens,
    )
    payload = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    items = payload["items"]
    layers = bundle.lens.source_layers

    records: list[dict[str, Any]] = []
    for item_index, item in enumerate(items, start=1):
        logger.info("item %d/%d: %s", item_index, len(items), item.get("name"))
        j_logits, _, _ = bundle.lens.apply(
            bundle.lens_model,
            item["prompt"],
            layers=layers,
            positions=[-1],
            use_jacobian=True,
        )
        l_logits, _, _ = bundle.lens.apply(
            bundle.lens_model,
            item["prompt"],
            layers=layers,
            positions=[-1],
            use_jacobian=False,
        )
        for intermediate in item["intermediates"]:
            token_ids = token_variants(bundle.tokenizer, intermediate)
            records.append(
                {
                    "item": item.get("name"),
                    "intermediate": intermediate,
                    "token_ids": token_ids,
                    "jlens_rank": best_rank(
                        (j_logits[layer][0] for layer in layers), token_ids
                    ),
                    "logit_lens_rank": best_rank(
                        (l_logits[layer][0] for layer in layers), token_ids
                    ),
                }
            )

    ks = (1, 5, 10, 50)
    jlens_pass = {str(k): pass_at(records, "jlens_rank", k) for k in ks}
    logit_pass = {str(k): pass_at(records, "logit_lens_rank", k) for k in ks}

    summary = {
        "evaluation": str(Path(args.eval).resolve()),
        "n_items": len(items),
        "n_targets": len(records),
        "source_layers": layers,
        "pass_at": {
            "jlens": jlens_pass,
            "logit_lens": logit_pass,
            "jlens_minus_logit_lens": {
                str(k): jlens_pass[str(k)] - logit_pass[str(k)] for k in ks
            },
        },
        "records": records,
    }
    output = Path(args.output)
    write_json_atomic(output, summary)
    write_json_atomic(
        output.with_suffix(".manifest.json"),
        build_manifest(
            config_path=args.config,
            project_root=args.project_root,
            extra={
                "lens": str(Path(args.lens).resolve()),
                "lens_sha256": sha256_file(args.lens),
                "eval": summary["evaluation"],
                "eval_sha256": sha256_file(args.eval),
            },
        ),
    )
    print(json.dumps(summary["pass_at"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
