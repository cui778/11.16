#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_segments(parsed_inp_path, baseline_stats_path, output_segment_list, output_static_csv, output_map_json):
    parsed = load_json(parsed_inp_path)
    baseline = load_json(baseline_stats_path) if Path(baseline_stats_path).exists() else {}

    junctions = parsed.get("junctions", {})
    outfalls = parsed.get("outfalls", {})
    storage = parsed.get("storage", {})
    conduits = parsed.get("conduits", {})
    xsections = parsed.get("xsections", {})
    link_flow_stats = baseline.get("links", {})
    link_properties = baseline.get("link_properties", {})

    node_elev = {}
    for source in (junctions, outfalls, storage):
        for node_id, item in source.items():
            node_elev[str(node_id)] = float(item.get("elevation", 0.0))

    segments = []
    static_rows = []
    node_to_primary_out = {}
    outgoing_per_node = {}

    for segment_idx, (link_id, link_info) in enumerate(conduits.items()):
        link_id = str(link_id)
        from_node = str(link_info.get("from_node", ""))
        to_node = str(link_info.get("to_node", ""))
        length = float(link_info.get("length", 0.0))
        xsec = xsections.get(link_id, {})
        diameter = float(xsec.get("geom1", 0.0))
        shape = str(xsec.get("shape", "UNKNOWN"))
        material = "UNKNOWN"
        elev_from = float(node_elev.get(from_node, 0.0))
        elev_to = float(node_elev.get(to_node, 0.0))
        slope = (elev_from - elev_to) / max(length, 1e-6) if length > 0 else 0.0
        baseline_mean_flow = float(link_flow_stats.get(link_id, {}).get("mean", 0.0))

        segment_id = f"seg_{segment_idx:04d}"
        segment = {
            "segment_id": segment_id,
            "segment_idx": int(segment_idx),
            "link_id": link_id,
            "from_node": from_node,
            "to_node": to_node,
        }
        segments.append(segment)

        static_rows.append(
            {
                "segment_id": segment_id,
                "segment_idx": int(segment_idx),
                "link_id": link_id,
                "from_node": from_node,
                "to_node": to_node,
                "length": length,
                "diameter": diameter,
                "slope": slope,
                "shape": shape,
                "material": material,
                "baseline_mean_flow": baseline_mean_flow,
            }
        )

        outgoing_per_node.setdefault(from_node, []).append(
            {
                "segment_id": segment_id,
                "link_id": link_id,
                "mean_flow": baseline_mean_flow,
                "length": length,
            }
        )

    for node_id, items in outgoing_per_node.items():
        items = sorted(items, key=lambda item: (item["mean_flow"], item["length"]), reverse=True)
        node_to_primary_out[node_id] = items[0]["segment_id"]

    payload = {
        "segments": segments,
        "node_to_primary_out_segment": node_to_primary_out,
        "meta": {
            "n_segments": len(segments),
            "source_parsed_inp": str(parsed_inp_path),
            "source_baseline_stats": str(baseline_stats_path),
        },
    }
    with open(output_segment_list, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    pd.DataFrame(static_rows).to_csv(output_static_csv, index=False, encoding="utf-8-sig")

    pipe_segment_map = {
        "link_id_to_segment_id": {item["link_id"]: item["segment_id"] for item in segments},
        "node_to_primary_out_segment": node_to_primary_out,
        "segment_id_to_link_id": {item["segment_id"]: item["link_id"] for item in segments},
        "segment_id_to_index": {item["segment_id"]: item["segment_idx"] for item in segments},
    }
    with open(output_map_json, "w", encoding="utf-8") as f:
        json.dump(pipe_segment_map, f, ensure_ascii=False, indent=2)

    return payload, pd.DataFrame(static_rows), pipe_segment_map


def main():
    import sys

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from config import Config

    cfg = Config()
    build_segments(
        parsed_inp_path=cfg.parsed_inp_data_file,
        baseline_stats_path=cfg.baseline_flow_stats_file,
        output_segment_list=cfg.segment_list_file,
        output_static_csv=cfg.edge_static_features_file,
        output_map_json=cfg.pipe_segment_map_file,
    )
    print(f"[OK] wrote {cfg.segment_list_file}")
    print(f"[OK] wrote {cfg.edge_static_features_file}")
    print(f"[OK] wrote {cfg.pipe_segment_map_file}")


if __name__ == "__main__":
    main()
