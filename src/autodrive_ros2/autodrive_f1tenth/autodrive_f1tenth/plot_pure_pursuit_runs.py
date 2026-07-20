#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "package.xml").exists():
            return parent
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path(__file__).resolve().parents[1]


def latest_two_runs(run_dir: Path) -> tuple[Path, Path]:
    candidates = sorted(run_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if len(candidates) < 2:
        raise FileNotFoundError(f"Need at least two .npz runs in {run_dir}")
    return candidates[-2], candidates[-1]


def as_state_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    if arr.shape[0] <= 10:
        return arr
    return arr.T


def as_input_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    if arr.shape[0] <= 4:
        return arr
    return arr.T


def as_xy_path(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D path array, got {arr.shape}")
    if arr.shape[1] == 2:
        return arr
    if arr.shape[0] == 2:
        return arr.T
    raise ValueError(f"Expected Nx2 or 2xN path array, got {arr.shape}")


def compute_ex_ey(states: np.ndarray, path_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = states[0]
    y = states[1]
    ex = np.zeros_like(x)
    ey = np.zeros_like(y)

    for i in range(len(x)):
        pos = np.array([x[i], y[i]], dtype=float)
        dists = np.linalg.norm(path_xy - pos, axis=1)
        idx = int(np.argmin(dists))

        prev_idx = (idx - 1) % len(path_xy)
        next_idx = (idx + 1) % len(path_xy)
        tangent = path_xy[next_idx] - path_xy[prev_idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12:
            tangent = np.array([1.0, 0.0], dtype=float)
        else:
            tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        err = pos - path_xy[idx]
        ex[i] = float(np.dot(err, tangent))
        ey[i] = float(np.dot(err, normal))

    return ex, ey


def load_run(path: Path) -> dict[str, np.ndarray | str | float]:
    with np.load(path, allow_pickle=True) as data:
        states = as_state_matrix(data["states"])
        inputs = as_input_matrix(data["inputs"])
        path_xy = as_xy_path(data["path"])
        time = np.asarray(data["time"], dtype=float).reshape(-1)
        model_name = str(data["model_name"])
        target_speed = float(np.asarray(data["target_speed"]).item())

    n = min(states.shape[1], inputs.shape[1], len(time))
    states = states[:, :n]
    inputs = inputs[:, :n]
    time = time[:n]
    if len(time) > 0:
        time = time - time[0]

    ex, ey = compute_ex_ey(states, path_xy)
    vx = states[3]
    vy = states[4]
    speed_mag = np.sqrt(vx**2 + vy**2)

    return {
        "label": path.stem,
        "model_name": model_name,
        "target_speed": target_speed,
        "time": time,
        "states": states,
        "inputs": inputs,
        "path_xy": path_xy,
        "ex": ex,
        "ey": ey,
        "vx": vx,
        "vy": vy,
        "speed_mag": speed_mag,
    }


def plot_runs(run_a: dict, run_b: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    signal_lw = 1.2
    color_a = "tab:blue"
    color_b = "tab:red"
    diff_fill = "0.7"

    trajectory_ax = axes[0, 0]
    trajectory_ax.plot(run_a["path_xy"][:, 0], run_a["path_xy"][:, 1], "k--", lw=1.0, alpha=0.7, label="reference")
    trajectory_ax.plot(
        run_a["states"][0],
        run_a["states"][1],
        lw=signal_lw,
        color=color_a,
        label=run_a["label"],
    )
    trajectory_ax.plot(
        run_b["states"][0],
        run_b["states"][1],
        lw=signal_lw,
        color=color_b,
        label=run_b["label"],
    )
    trajectory_ax.set_title("Trajectory")
    trajectory_ax.set_xlabel("x")
    trajectory_ax.set_ylabel("y")
    trajectory_ax.axis("equal")
    trajectory_ax.grid(True, alpha=0.25)
    trajectory_ax.legend(loc="best")

    plots = [
        (axes[0, 1], "Cross-Track Along Tangent $e_x$", "ex", "m"),
        (axes[1, 0], "Cross-Track Along Normal $e_y$", "ey", "m"),
        (axes[1, 1], "Longitudinal Velocity $v_x$", "vx", "m/s"),
        (axes[2, 0], "Lateral Velocity $v_y$", "vy", "m/s"),
        (axes[2, 1], "Throttle", "inputs0", ""),
        (axes[3, 0], "Steering Angle", "inputs1", "rad"),
        (axes[3, 1], "Speed Magnitude", "speed_mag", "m/s"),
    ]

    for ax, title, key, ylabel in plots:
        if key == "inputs0":
            y_a = run_a["inputs"][0]
            y_b = run_b["inputs"][0]
        elif key == "inputs1":
            y_a = run_a["inputs"][1]
            y_b = run_b["inputs"][1]
        else:
            y_a = run_a[key]
            y_b = run_b[key]

        ax.plot(run_a["time"], y_a, lw=signal_lw, color=color_a, label=run_a["label"])
        ax.plot(run_b["time"], y_b, lw=signal_lw, color=color_b, label=run_b["label"])

        shared_t_end = min(float(run_a["time"][-1]), float(run_b["time"][-1]))
        if shared_t_end > 0.0:
            shared_t = np.linspace(0.0, shared_t_end, 400)
            y_a_interp = np.interp(shared_t, run_a["time"], y_a)
            y_b_interp = np.interp(shared_t, run_b["time"], y_b)
            ax.fill_between(
                shared_t,
                y_a_interp,
                y_b_interp,
                color=diff_fill,
                alpha=0.18,
                linewidth=0.0,
                zorder=0,
            )
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle(f"Pure Pursuit Run Comparison: {run_a['label']} vs {run_b['label']}", fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(out_path)


def main() -> None:
    repo_root = get_repo_root()
    default_run_dir = repo_root / "output" / "ros2_pure_pursuit_runs"

    parser = argparse.ArgumentParser(description="Compare two ROS2 pure pursuit run logs.")
    parser.add_argument("run_a", nargs="?", help="Path to first .npz run log")
    parser.add_argument("run_b", nargs="?", help="Path to second .npz run log")
    parser.add_argument(
        "--run-dir",
        default=str(default_run_dir),
        help="Directory to search when run paths are not provided",
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "output" / "ros2_pure_pursuit_runs" / "comparison.png"),
        help="Output figure path",
    )
    args = parser.parse_args()

    if args.run_a and args.run_b:
        run_a_path = Path(args.run_a).expanduser()
        run_b_path = Path(args.run_b).expanduser()
    else:
        run_a_path, run_b_path = latest_two_runs(Path(args.run_dir).expanduser())

    run_a = load_run(run_a_path)
    run_b = load_run(run_b_path)
    plot_runs(run_a, run_b, Path(args.out).expanduser())


if __name__ == "__main__":
    main()
