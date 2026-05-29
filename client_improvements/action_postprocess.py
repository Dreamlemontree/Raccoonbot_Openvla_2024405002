import csv
import json
import time
from pathlib import Path

import numpy as np


class ActionPostProcessor:
    def __init__(self, max_delta_xyz=0.012, alpha=0.35):
        self.max_delta_xyz = float(max_delta_xyz)
        self.alpha = float(alpha)
        self.prev_delta_xyz = None

    def process(self, action7):
        raw = np.asarray(action7, dtype=np.float32).reshape(-1)

        if raw.shape[0] < 7:
            padded = np.zeros(7, dtype=np.float32)
            padded[: raw.shape[0]] = raw
            raw = padded

        processed = raw.copy()

        clipped_delta = np.clip(raw[:3], -self.max_delta_xyz, self.max_delta_xyz)

        if self.prev_delta_xyz is None:
            smoothed_delta = clipped_delta
        else:
            smoothed_delta = self.alpha * clipped_delta + (1.0 - self.alpha) * self.prev_delta_xyz

        self.prev_delta_xyz = smoothed_delta

        processed[:3] = smoothed_delta
        processed[6] = np.clip(processed[6], -1.0, 1.0)

        info = {
            "raw_action7": raw.tolist(),
            "processed_action7": processed.tolist(),
            "pitch_hint": float(raw[4]),
            "raw_delta_norm": float(np.linalg.norm(raw[:3])),
            "processed_delta_norm": float(np.linalg.norm(processed[:3])),
            "raw_gripper": float(raw[6]),
            "processed_gripper": float(processed[6]),
        }
        return processed.tolist(), info


class InferenceLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "step",
                "time_sec",
                "instruction",
                "target_color",
                "latency_ms",
                "raw_action7",
                "processed_action7",
                "pitch_hint",
                "raw_delta_norm",
                "processed_delta_norm",
                "raw_gripper",
                "processed_gripper",
                "exec_success",
                "final_delta_xyz",
                "actual_move_xyz",
                "retry_count",
            ],
        )
        self.writer.writeheader()

    def write(self, step, instruction, target_color, latency_ms, info, exec_info=None):
        exec_info = exec_info or {}
        row = {
            "step": step,
            "time_sec": time.time(),
            "instruction": instruction,
            "target_color": target_color,
            "latency_ms": latency_ms,
            "raw_action7": json.dumps(info["raw_action7"]),
            "processed_action7": json.dumps(info["processed_action7"]),
            "pitch_hint": info["pitch_hint"],
            "raw_delta_norm": info["raw_delta_norm"],
            "processed_delta_norm": info["processed_delta_norm"],
            "raw_gripper": info["raw_gripper"],
            "processed_gripper": info["processed_gripper"],
            "exec_success": exec_info.get("success", False),
            "final_delta_xyz": json.dumps(exec_info.get("final_delta_xyz", [])),
            "actual_move_xyz": json.dumps(exec_info.get("actual_move_xyz", [])),
            "retry_count": exec_info.get("retry_count", ""),
        }
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()