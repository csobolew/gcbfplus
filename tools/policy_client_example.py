#!/usr/bin/env python3
"""
Interactive client that repeatedly queries the policy server and visualises agent, goal, and obstacle
positions inside a square workspace (default: unit square with coordinates in [0, 1]).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms
import numpy as np

PAYLOAD_WAIT = 0.1  # seconds to wait between requests when the server is unreachable


@dataclass
class AgentState:
    pos: np.ndarray  # shape (N, 2)
    yaw: np.ndarray  # shape (N,)


def build_payload(state: AgentState, goals: np.ndarray, obstacles: np.ndarray, area_size: float) -> dict:
    if area_size <= 0:
        raise ValueError("area_size must be positive.")
    agents_norm = np.concatenate([state.pos / area_size, state.yaw[:, None]], axis=1)
    goals_norm = goals.copy()
    goals_norm[:, :2] = goals_norm[:, :2] / area_size
    obstacles_norm = obstacles.copy()
    obstacles_norm[:, :2] = obstacles_norm[:, :2] / area_size
    obstacles_norm[:, 2:4] = obstacles_norm[:, 2:4] / area_size

    return {
        "agents": agents_norm.tolist(),
        "goals": goals_norm.tolist(),
        "obstacles": obstacles_norm.tolist(),
    }


def send_request(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def normalise_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def step_dynamics(state: AgentState, action: np.ndarray, dt: float,
                  lin_scale: float, yaw_scale: float, area_size: float) -> AgentState:
    if action.ndim != 2 or action.shape[0] != state.pos.shape[0]:
        raise ValueError("Action has unexpected shape.")

    vx = lin_scale * np.clip(action[:, 0], -1.0, 1.0)
    if action.shape[1] >= 2:
        vy = lin_scale * np.clip(action[:, 1], -1.0, 1.0)
    else:
        vy = np.zeros_like(vx)
    if action.shape[1] >= 3:
        yaw_rate = yaw_scale * np.clip(action[:, 2], -1.0, 1.0)
    else:
        yaw_rate = np.zeros_like(vx)

    cos_yaw = np.cos(state.yaw)
    sin_yaw = np.sin(state.yaw)

    dx = (vx * cos_yaw - vy * sin_yaw) * dt
    dy = (vx * sin_yaw + vy * cos_yaw) * dt

    pos = state.pos + np.stack([dx, dy], axis=1)
    yaw = normalise_angle(state.yaw + yaw_rate * dt)

    pos = np.clip(pos, 0.0, area_size)
    return AgentState(pos=pos, yaw=yaw)


def create_artists(ax, state: AgentState, goals: np.ndarray, obstacles: np.ndarray, arrow_len: float):
    agent_scatter = ax.scatter(state.pos[:, 0], state.pos[:, 1], c="tab:blue", s=60, label="Agent")
    goal_scatter = ax.scatter(goals[:, 0], goals[:, 1], c="tab:green", marker="*", s=80, label="Goal")
    quiver = ax.quiver(
        state.pos[:, 0],
        state.pos[:, 1],
        np.cos(state.yaw) * arrow_len,
        np.sin(state.yaw) * arrow_len,
        color="tab:blue",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.01 * arrow_len,
        pivot="middle",
    )

    rects = []
    for idx, obstacle in enumerate(obstacles):
        cx, cy, width, height, yaw = obstacle
        rect = patches.Rectangle(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            linewidth=1.5,
            edgecolor="tab:red",
            facecolor="tab:red",
            alpha=0.3,
            label="Obstacle" if idx == 0 else None,
        )
        rotation = transforms.Affine2D().rotate_around(cx, cy, yaw)
        rect.set_transform(rotation + ax.transData)
        ax.add_patch(rect)
        rects.append((rect, rotation))

    return agent_scatter, goal_scatter, quiver, rects


def update_artists(artists, state: AgentState, arrow_len: float):
    agent_scatter, goal_scatter, quiver, rects = artists
    agent_scatter.set_offsets(state.pos)
    quiver.set_offsets(state.pos)
    quiver.set_UVC(np.cos(state.yaw) * arrow_len, np.sin(state.yaw) * arrow_len)
    return agent_scatter, goal_scatter, quiver, rects


def run_visualisation(url: str, state: AgentState, goals: np.ndarray, obstacles: np.ndarray,
                      dt: float, lin_scale: float, yaw_scale: float, area_size: float):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0.0, area_size)
    ax.set_ylim(0.0, area_size)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(f"Policy Server Interaction (area size {area_size})")

    arrow_len = 0.1 * area_size
    artists = create_artists(ax, state, goals, obstacles, arrow_len=arrow_len)
    ax.legend(loc="upper right")
    status_text = ax.text(0.01, 1.02, "", transform=ax.transAxes)

    last_response = {"action": np.zeros((state.pos.shape[0], 2)), "step": None}

    def animation_step(frame_idx):
        nonlocal state, last_response
        payload = build_payload(state, goals, obstacles, area_size)

        try:
            response = send_request(url, payload)
            last_response = response
        except urllib.error.URLError as exc:
            status_text.set_text(f"Connection failed ({exc}). Retrying...")
            time.sleep(PAYLOAD_WAIT)
            return update_artists(artists, state, arrow_len)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            status_text.set_text(f"HTTP {exc.code}: {body}")
            time.sleep(PAYLOAD_WAIT)
            return update_artists(artists, state, arrow_len)

        action = np.asarray(last_response.get("action", []), dtype=np.float32)
        if action.ndim != 2:
            status_text.set_text("Invalid action shape from server")
            time.sleep(PAYLOAD_WAIT)
            return update_artists(artists, state, arrow_len)

        state = step_dynamics(state, action, dt, lin_scale, yaw_scale, area_size=area_size)
        update_artists(artists, state, arrow_len)

        step_info = last_response.get("step", "N/A")
        status_text.set_text(f"Server step: {step_info}  |  action[0]={np.array2string(action[0], precision=3)}")
        return (*artists, status_text)

    anim = animation.FuncAnimation(
        fig,
        animation_step,
        interval=dt * 1000,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    return anim


def initialise_state(num_agents: int, area_size: float) -> Tuple[AgentState, np.ndarray]:
    if num_agents <= 0:
        raise ValueError("num_agents must be positive.")
    x_start = 0.2 * area_size
    x_end = 0.3 * area_size
    y_start = 0.2 * area_size
    y_end = y_start + 0.05 * area_size * max(0, num_agents - 1)
    x_positions = np.linspace(x_start, x_end, num_agents)
    y_positions = np.linspace(y_start, y_end, num_agents)
    pos = np.stack([x_positions, y_positions], axis=1).astype(np.float32)
    yaw = np.zeros(num_agents, dtype=np.float32)
    state = AgentState(pos=pos.astype(np.float32), yaw=yaw)

    goal_x = np.linspace(0.8 * area_size, 0.7 * area_size, num_agents)
    goal_y = np.linspace(0.8 * area_size, (0.8 - 0.05 * max(0, num_agents - 1)) * area_size, num_agents)
    goals = np.stack([goal_x, goal_y, np.zeros(num_agents, dtype=np.float32)], axis=1)
    return state, goals


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise repeated interactions with the policy server.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port (default: 8000).")
    parser.add_argument("--num-agents", type=int, default=1, help="Number of agents in the simulation.")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation time step (seconds).")
    parser.add_argument("--lin-scale", type=float, default=0.3,
                        help="Scale factor that maps action components to linear velocity (units/s).")
    parser.add_argument("--yaw-scale", type=float, default=1.0,
                        help="Scale factor that maps action component to yaw rate (rad/s).")
    parser.add_argument("--area-size", type=float, default=1.0,
                        help="Physical size of the square workspace (default: 1.0).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/act"

    state, goals = initialise_state(args.num_agents, args.area_size)
    obstacles = np.array([[0.5 * args.area_size,
                           0.5 * args.area_size,
                           0.5 * args.area_size,
                           0.5 * args.area_size,
                           np.pi/4]], dtype=np.float32)

    anim = run_visualisation(
        url,
        state,
        goals,
        obstacles,
        args.dt,
        args.lin_scale,
        args.yaw_scale,
        area_size=args.area_size,
    )
    # keep reference so animation is not garbage-collected immediately
    _ = anim
    return 0


if __name__ == "__main__":
    sys.exit(main())
