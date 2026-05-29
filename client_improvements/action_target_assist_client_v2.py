import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from openvla_multicolor_client import (
    CYLINDER_BODY_BY_COLOR,
    DEFAULT_MIN_OBJECT_DISTANCE,
    DEFAULT_OBJECT_X_RANGE,
    SyncSimRaccoonEnv,
    clear_existing_images,
    object_specs_to_meta,
    request_action,
    reset_multicolor_scene,
    resolve_target_color_and_instruction,
    sample_object_specs,
)

try:
    from openvla_multicolor_client_real_robot import RealRaccoonController
except Exception:
    RealRaccoonController = None


DEFAULT_SHAPE_BY_COLOR = {
    "red": "cylinder",
    "blue": "cube",
    "green": "sphere",
    "yellow": "cylinder",
}


def clip_delta(delta_xyz, max_delta_xyz):
    return np.clip(np.asarray(delta_xyz, dtype=np.float32), -max_delta_xyz, max_delta_xyz)


def default_instruction(task, color, shape):
    if task == "push":
        return f"push the {color} {shape} forward"
    if task == "lift":
        return f"lift the {color} {shape}"
    return f"grasp the {color} {shape}"


def lift_or_grasp_action(env, target_xy, stage, max_delta_xyz, hover_z, grasp_z, lift_z, xy_tol, z_tol):
    ee = np.asarray(env.get_ee_pose(), dtype=np.float32)
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    dist_xy = float(np.linalg.norm(ee[:2] - np.array([target_x, target_y], dtype=np.float32)))

    if stage == "approach" and dist_xy <= xy_tol:
        stage = "descend"
    if stage == "descend" and abs(float(ee[2]) - grasp_z) <= z_tol:
        stage = "close"

    if stage == "approach":
        goal = np.array([target_x, target_y, hover_z], dtype=np.float32)
        gripper = 0.0
    elif stage == "descend":
        goal = np.array([target_x, target_y, grasp_z], dtype=np.float32)
        gripper = 0.0
    elif stage == "close":
        goal = np.array([target_x, target_y, grasp_z], dtype=np.float32)
        gripper = 1.0
    else:
        goal = np.array([target_x, target_y, lift_z], dtype=np.float32)
        gripper = 1.0

    delta = clip_delta(goal - ee, max_delta_xyz)
    action7 = [float(delta[0]), float(delta[1]), float(delta[2]), 0.0, 0.0, 0.0, gripper]
    return action7, stage, goal.tolist(), ee.tolist(), dist_xy


def push_action(
    env,
    target_xy,
    stage,
    max_delta_xyz,
    push_z,
    push_start_offset,
    push_distance,
    xy_tol,
    z_tol,
):
    ee = np.asarray(env.get_ee_pose(), dtype=np.float32)
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    start = np.array([target_x, target_y - push_start_offset, push_z], dtype=np.float32)
    goal = np.array([target_x, target_y + push_distance, push_z], dtype=np.float32)

    dist_start_xy = float(np.linalg.norm(ee[:2] - start[:2]))
    if stage == "approach" and dist_start_xy <= xy_tol:
        stage = "descend"
    if stage == "descend" and abs(float(ee[2]) - push_z) <= z_tol:
        stage = "push"

    if stage == "approach":
        active_goal = np.array([start[0], start[1], max(push_z + 0.040, 0.060)], dtype=np.float32)
    elif stage == "descend":
        active_goal = start
    else:
        active_goal = goal

    delta = clip_delta(active_goal - ee, max_delta_xyz)
    action7 = [float(delta[0]), float(delta[1]), float(delta[2]), 0.0, 0.0, 0.0, 0.0]
    return action7, stage, active_goal.tolist(), ee.tolist(), dist_start_xy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", required=True)
    parser.add_argument("--xml_path", default="Raccoon_colored_cylinder.xml")
    parser.add_argument("--target_color", required=True, choices=list(CYLINDER_BODY_BY_COLOR.keys()))
    parser.add_argument("--object_shape", default=None, choices=["cylinder", "cube", "sphere", "box"])
    parser.add_argument("--task", default="lift", choices=["grasp", "lift", "push"])
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--unnorm_key", default="raccoon_pick_place")
    parser.add_argument("--output_dir", default="rollout_outputs_task_assist_v2")
    parser.add_argument("--episode_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--max_delta_xyz", type=float, default=0.008)
    parser.add_argument("--speed", type=int, default=45)
    parser.add_argument("--hover_z", type=float, default=0.065)
    parser.add_argument("--grasp_z", type=float, default=0.022)
    parser.add_argument("--lift_z", type=float, default=0.085)
    parser.add_argument("--push_z", type=float, default=0.026)
    parser.add_argument("--push_start_offset", type=float, default=0.028)
    parser.add_argument("--push_distance", type=float, default=0.045)
    parser.add_argument("--push_success_threshold", type=float, default=0.010)
    parser.add_argument("--xy_tol", type=float, default=0.010)
    parser.add_argument("--z_tol", type=float, default=0.006)
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.25)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.3)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--request_every_n_steps", type=int, default=1)
    parser.add_argument("--camera_name", default="front_view")
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--object_x_range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE)
    parser.add_argument("--object_y_range", type=float, nargs=2, default=(0.16, 0.19))
    parser.add_argument("--min_object_distance", type=float, default=DEFAULT_MIN_OBJECT_DISTANCE)
    parser.add_argument("--target_x", type=float, default=None)
    parser.add_argument("--target_y", type=float, default=None)
    parser.add_argument("--action_log_csv", default=None)
    parser.add_argument("--use_real_robot", action="store_true")
    parser.add_argument("--allow_real_robot_fail", action="store_true")
    parser.add_argument("--real_initial_wait_seconds", type=float, default=5.0)
    parser.add_argument("--real_settle_seconds", type=float, default=0.8)
    parser.add_argument("--real_go_home_on_exit", action="store_true")
    args = parser.parse_args()

    args.request_every_n_steps = max(1, int(args.request_every_n_steps))
    object_shape = args.object_shape or DEFAULT_SHAPE_BY_COLOR.get(args.target_color, "cylinder")
    if object_shape == "box":
        object_shape = "cube"

    if args.instruction is None:
        args.instruction = default_instruction(args.task, args.target_color, object_shape)

    out_dir = Path(args.output_dir) / f"episode_{args.episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_images(out_dir)

    rng = np.random.default_rng(args.seed)
    target_color, instruction = resolve_target_color_and_instruction(
        instruction=args.instruction,
        target_color_arg=args.target_color,
        rng=rng,
        instruction_template=default_instruction(args.task, "{color}", object_shape),
    )
    object_specs = sample_object_specs(
        rng=rng,
        x_range=tuple(args.object_x_range),
        y_range=tuple(args.object_y_range),
        min_distance=args.min_object_distance,
    )
    if args.target_x is not None and args.target_y is not None:
        object_specs[target_color]["x"] = float(args.target_x)
        object_specs[target_color]["y"] = float(args.target_y)
        fixed_offsets = [(-0.075, 0.045), (0.075, 0.045), (-0.075, -0.035), (0.075, -0.035)]
        offset_idx = 0
        for color in object_specs:
            if color == target_color:
                continue
            dx, dy = fixed_offsets[offset_idx]
            object_specs[color]["x"] = float(args.target_x + dx)
            object_specs[color]["y"] = float(args.target_y + dy)
            offset_idx += 1

    shape_by_color = dict(DEFAULT_SHAPE_BY_COLOR)
    shape_by_color[target_color] = object_shape

    log_path = Path(args.action_log_csv or out_dir / "task_assist_v2_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = SyncSimRaccoonEnv(
        xml_path=args.xml_path,
        image_size=(256, 256),
        camera_name=args.camera_name,
        use_viewer=args.use_viewer,
    )
    real_robot = None

    try:
        if args.use_real_robot:
            if RealRaccoonController is None:
                raise ImportError("RealRaccoonController is not available.")
            try:
                real_robot = RealRaccoonController(
                    require_ready=not args.allow_real_robot_fail,
                    home_wait_seconds=args.real_initial_wait_seconds,
                )
            except Exception:
                if not args.allow_real_robot_fail:
                    raise
                print("[REAL_ROBOT WARN] connection failed; MuJoCo rollout continues.")
                real_robot = None

        reset_multicolor_scene(env=env, object_specs=object_specs, target_color=target_color)
        env.lockh()
        env.debug_check_current_ee_reachable()

        if args.initial_settle_seconds > 0:
            env.settle_steps(seconds=args.initial_settle_seconds)

        target_body = CYLINDER_BODY_BY_COLOR[target_color]
        target_init_pose = env.get_object_pose(target_body).tolist()
        target_xy = [object_specs[target_color]["x"], object_specs[target_color]["y"]]

        meta = {
            "instruction": instruction,
            "target_color": target_color,
            "target_shape": object_shape,
            "task": args.task,
            "target_body_name": target_body,
            "shape_by_color": shape_by_color,
            "all_object_init_poses": object_specs_to_meta(object_specs),
            "mode": "openvla_request_with_task_assisted_7d_to_4dof_mapping_v2",
            "args": vars(args),
        }
        with open(out_dir / "rollout_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(
            f"[TASK_ASSIST_V2] task={args.task!r} | instruction={instruction!r} | "
            f"target={target_color}/{object_shape} | target_xy=({target_xy[0]:.3f}, {target_xy[1]:.3f}) | "
            f"request_every_n_steps={args.request_every_n_steps} | log={log_path}"
        )

        obs = env.get_observation()
        stage = "approach"
        close_steps = 0
        last_raw_action = [0.0] * 7
        server_request_count = 0

        fieldnames = [
            "step",
            "task",
            "stage",
            "instruction",
            "target_color",
            "object_shape",
            "request_sent",
            "server_request_count",
            "latency_ms",
            "motion_elapsed_ms",
            "step_total_ms",
            "raw_action7",
            "assisted_action7",
            "ee_pose_before",
            "goal_xyz",
            "target_xy_dist",
            "exec_success",
            "final_delta_xyz",
            "gripper_cmd",
            "target_object_z",
            "target_object_lift",
            "target_object_move_xy",
            "target_object_move_y",
        ]

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for step_idx in range(args.max_steps):
                step_start = time.perf_counter()
                request_sent = (step_idx % args.request_every_n_steps == 0) or step_idx == 0
                latency_ms = 0.0

                if request_sent:
                    t0 = time.perf_counter()
                    response = request_action(
                        server_url=args.server_url,
                        instruction=instruction,
                        image_rgb=obs["image"],
                        unnorm_key=args.unnorm_key,
                        timeout=args.request_timeout,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    last_raw_action = response["action"]
                    server_request_count += 1

                if args.task == "push":
                    action, stage, goal_xyz, ee_pose, dist_xy = push_action(
                        env=env,
                        target_xy=target_xy,
                        stage=stage,
                        max_delta_xyz=args.max_delta_xyz,
                        push_z=args.push_z,
                        push_start_offset=args.push_start_offset,
                        push_distance=args.push_distance,
                        xy_tol=args.xy_tol,
                        z_tol=args.z_tol,
                    )
                else:
                    action, stage, goal_xyz, ee_pose, dist_xy = lift_or_grasp_action(
                        env=env,
                        target_xy=target_xy,
                        stage=stage,
                        max_delta_xyz=args.max_delta_xyz,
                        hover_z=args.hover_z,
                        grasp_z=args.grasp_z,
                        lift_z=args.lift_z,
                        xy_tol=args.xy_tol,
                        z_tol=args.z_tol,
                    )

                motion_start = time.perf_counter()
                exec_info = env.execute_delta_action7(
                    action=action,
                    speed=args.speed,
                    delta_scale=1.0,
                    max_delta_xyz=args.max_delta_xyz,
                )
                if real_robot is not None and real_robot.connected:
                    exec_info["real_robot"] = real_robot.execute_from_exec_info(exec_info, speed=args.speed)
                    time.sleep(args.real_settle_seconds)

                env.settle_steps(seconds=args.settle_seconds_per_action)
                motion_elapsed_ms = (time.perf_counter() - motion_start) * 1000.0

                target_pose = env.get_object_pose(target_body).tolist()
                target_lift = float(target_pose[2] - target_init_pose[2])
                move_xy = float(np.linalg.norm(np.array(target_pose[:2]) - np.array(target_init_pose[:2])))
                move_y = float(target_pose[1] - target_init_pose[1])

                obs = env.get_observation()
                Image.fromarray(obs["image"]).save(out_dir / f"frame_{step_idx:06d}.png")

                step_total_ms = (time.perf_counter() - step_start) * 1000.0
                writer.writerow(
                    {
                        "step": step_idx,
                        "task": args.task,
                        "stage": stage,
                        "instruction": instruction,
                        "target_color": target_color,
                        "object_shape": object_shape,
                        "request_sent": int(request_sent),
                        "server_request_count": server_request_count,
                        "latency_ms": latency_ms,
                        "motion_elapsed_ms": motion_elapsed_ms,
                        "step_total_ms": step_total_ms,
                        "raw_action7": json.dumps(last_raw_action),
                        "assisted_action7": json.dumps(action),
                        "ee_pose_before": json.dumps(ee_pose),
                        "goal_xyz": json.dumps(goal_xyz),
                        "target_xy_dist": dist_xy,
                        "exec_success": exec_info.get("success", False),
                        "final_delta_xyz": json.dumps(exec_info.get("final_delta_xyz", [])),
                        "gripper_cmd": exec_info.get("gripper_cmd", ""),
                        "target_object_z": target_pose[2],
                        "target_object_lift": target_lift,
                        "target_object_move_xy": move_xy,
                        "target_object_move_y": move_y,
                    }
                )
                f.flush()

                print(
                    f"[{step_idx:03d}] task={args.task} | stage={stage} | request={int(request_sent)} | "
                    f"dist_xy={dist_xy:.4f} | gripper={action[6]:.1f} | "
                    f"lift={target_lift:.4f} | move_xy={move_xy:.4f} | latency={latency_ms:.1f} ms"
                )

                if args.task == "push":
                    if stage == "push" and move_xy >= args.push_success_threshold:
                        print(f"[SUCCESS] push distance reached: {move_xy:.4f} m")
                        break
                else:
                    if stage == "close":
                        close_steps += 1
                        if close_steps >= 6:
                            stage = "lift"
                    if stage == "lift" and target_lift > 0.012:
                        print(f"[SUCCESS] target lift detected: {target_lift:.4f} m")
                        break

    finally:
        if real_robot is not None:
            if args.real_go_home_on_exit and real_robot.connected:
                real_robot.go_home(wait_seconds=0.0)
            real_robot.close()
        env.close()


if __name__ == "__main__":
    main()
