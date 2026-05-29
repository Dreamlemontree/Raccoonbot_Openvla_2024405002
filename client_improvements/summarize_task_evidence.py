import argparse
import csv
import json
from pathlib import Path


def summarize_csv(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {"file": str(path), "rows": 0}

    def floats(key):
        vals = []
        for row in rows:
            value = row.get(key, "")
            if value not in ("", None):
                try:
                    vals.append(float(value))
                except ValueError:
                    pass
        return vals

    request_sent = [int(float(row.get("request_sent", 0) or 0)) for row in rows]
    latency = floats("latency_ms")
    step_total = floats("step_total_ms")
    lift = floats("target_object_lift")
    move_xy = floats("target_object_move_xy")

    summary = {
        "file": str(path),
        "rows": len(rows),
        "task": rows[0].get("task", ""),
        "instruction": rows[0].get("instruction", ""),
        "target_color": rows[0].get("target_color", ""),
        "object_shape": rows[0].get("object_shape", ""),
        "stages": sorted({row.get("stage", "") for row in rows}),
        "server_requests": sum(request_sent),
        "request_ratio": sum(request_sent) / max(1, len(rows)),
        "avg_nonzero_latency_ms": (
            sum(v for v in latency if v > 0) / max(1, len([v for v in latency if v > 0]))
        ),
        "avg_step_total_ms": sum(step_total) / max(1, len(step_total)),
        "max_lift_m": max(lift) if lift else 0.0,
        "max_move_xy_m": max(move_xy) if move_xy else 0.0,
        "last_lift_m": lift[-1] if lift else 0.0,
        "last_move_xy_m": move_xy[-1] if move_xy else 0.0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("--out_json", default="evidence/task_evidence_summary.json")
    args = parser.parse_args()

    summaries = [summarize_csv(Path(path)) for path in args.csv_paths]
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
