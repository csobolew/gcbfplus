import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pathlib
import functools as ft
import matplotlib.pyplot as plt
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve

from .base import MultiAgentEnv, RolloutResult
from ..utils.typing import State, Array, AgentState, Action, Reward, Cost, Done, Info, Pos2d
from ..utils.graph import GraphsTuple, EdgeBlock, GetGraph
from .obstacle import Obstacle, Rectangle
from typing import NamedTuple, Tuple, Optional
from ..utils.utils import merge01, jax_vmap
from .plot import render_video
from .utils import get_node_goal_rng, inside_obstacles, get_lidar

class QuadrupedAccel(MultiAgentEnv):
    
    AGENT = 0
    GOAL = 1
    OBS = 2
    ENV_SIZE = 48.0

    class EnvState(NamedTuple):
        agent: State
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]
        
    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.55/ENV_SIZE,
        "comm_radius": 5.5/ENV_SIZE,
        "n_rays": 32,
        "obs_len_range": [1.0/ENV_SIZE, 4.0/ENV_SIZE],
        "n_obs": 1,
        "time_horizon": 50,
        "ilqr_iterations": 1,
        "action_lower": np.array([-0.75/ENV_SIZE, -0.75/ENV_SIZE, -1.0]),
        "action_upper": np.array([0.75/ENV_SIZE, 0.75/ENV_SIZE, 1.0])
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int=512,
            max_travel: float=None,
            dt: float=0.03,
            params: dict=None
    ):
        super(QuadrupedAccel, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.trajs = []
        self.prev_u_ref = None
        self.prev_x_ref = None
        self.create_obstacles = jax_vmap(Rectangle.create)

    @property
    def state_dim(self) -> int:
        return 5 # x, y, yaw, vx, vy
    
    @property
    def node_dim(self) -> int:
        return 3 # indicator: agent: 001, goal: 010, obstacle: 100
    
    @property
    def edge_dim(self) -> int:
        return 2 # x_rel, y_rel
    
    @property
    def action_dim(self) -> int:
        return 3 # ax, ax, yawrate
    
    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * jnp.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # Randomly generate agent and goal position (x, y)
        pos_key, key = jr.split(key, 2)
        states_pos, goals_pos = get_node_goal_rng(
            pos_key, self.area_size, 2, obstacles, self.num_agents, 4 * self.params["car_radius"], self.max_travel
        )

        # Randomly generate agent and goal orientation (yaw)
        agent_yaw_key, key = jr.split(key, 2)
        agent_orientations = jr.uniform(agent_yaw_key, (self.num_agents, 1), minval=-jnp.pi, maxval=jnp.pi)

        goal_yaw_key, key = jr.split(key, 2)
        goal_orientations = jr.uniform(goal_yaw_key, (self.num_agents, 1), minval=-jnp.pi, maxval=jnp.pi)

        agent_v_x = jnp.zeros((self.num_agents, 1))
        agent_v_y = jnp.zeros((self.num_agents, 1))

        states = jnp.concatenate([states_pos, agent_orientations, agent_v_x, agent_v_y], axis=1)
        goals = jnp.concatenate([goals_pos, goal_orientations, agent_v_x, agent_v_y], axis=1)
        env_states = self.EnvState(states, goals, obstacles)

        return self.get_graph(env_states)
    
    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        # Scale normalized actions to actual physical ranges
        scaled_action = 0.5 * ((action + 1.0) * (self.PARAMS["action_upper"] - self.PARAMS["action_lower"])) + self.PARAMS["action_lower"]

        # Unpack
        ax = scaled_action[:, 0]
        ay = scaled_action[:, 1]
        yaw_rate = scaled_action[:, 2]
        vx = agent_states[:, 3]
        vy = agent_states[:, 4]
        yaw = agent_states[:, 2]

        # Compute global velocity derivatives
        x_dot = vx * jnp.cos(yaw) - vy * jnp.sin(yaw)
        y_dot = vx * jnp.sin(yaw) + vy * jnp.cos(yaw)

        x_dot_dot = ax
        y_dot_dot = ay
        
        yaw_dot = yaw_rate

        return jnp.stack([x_dot, y_dot, yaw_dot, x_dot_dot, y_dot_dot], axis=1)
    
    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        x_dot = self.agent_xdot(agent_states, action)
        n_state_agent_new = agent_states + x_dot * self.dt
        n_state_agent_new = n_state_agent_new.at[:, 2].set(
            jnp.arctan2(jnp.sin(n_state_agent_new[:, 2]), jnp.cos(n_state_agent_new[:, 2]))
        )
        n_state_agent_new = self.clip_state(n_state_agent_new)
        return n_state_agent_new
    
    def step(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        
        # Calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        
        # Clip action to reasonable values
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)


        # Episode ends when reaching max episode steps
        done = jnp.array(False)

        # Compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=-1) ** 2).mean()

        cost = self.get_cost(graph)


        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goals, obstacles)

        info = {}
        if get_eval_info:
            # Collision between agents and obstacles
            agent_pos = agent_states[:, :2]
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])

        return self.get_graph(next_state), reward, cost, done, info
    
    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()

        return cost
    
    def render_video(
            self,
            rollout: RolloutResult,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict=None,
            dpi: int=100,
            **kwargs
    ) -> None:
        
        # Add orientation visualization
        if viz_opts is None:
            viz_opts = {}

        # Enable orientation arrows (added to plot.py)
        viz_opts['show_orientation'] = True
        viz_opts['arrow_length'] = self.params["car_radius"] * 2.0

        new_trajs = np.array(self.trajs)
        new_trajs = new_trajs[::2]

        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            trajs=new_trajs,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: Pos2d) -> list[EdgeBlock]:
        # Total number of lidar hits

        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent[:, :2] # Only use position, not orientation
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # pairwise difference in locations
        dist = jnp.linalg.norm(pos_diff, axis=-1) # Compute distances between all agents
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1) # Prevent agent from connecting to itself
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"]) # Boolean if within communication radius (True if so)
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(pos_diff, agent_agent_mask, id_agent, id_agent) # Create EdgeBlock to define these connections

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2) 
        agent_goal_mask = jnp.eye(self.num_agents) # Since each agent has its own goal, mask is identity (connect only to corresponding goal)
        agent_goal_feats = state.agent[:, None, :2] - state.goal[None, :, :2] # Compute relative positions from agent to goal
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats ** 2, axis=-1, keepdims=True)) # Compute distance
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius) # Clip distance to communication radius if longer
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats * coef
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        ) # Each agent connects to it's goal with some clipped relative direction (it knows the direction, but only up to comm_radius away)

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_feats = agent_pos[i, :2] - lidar_data[id_hits, :2]
            lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1) # Compute distance from agent to each lidar hit
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1) # Only consider hits that are within comm_radius
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar) # Mask active hits
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges
    
    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2

        vx = state[:, 3]
        vy = state[:, 4]
        yaw = state[:, 2]


        # f is 0, no drift
        f = jnp.zeros_like(state)
        f = f.at[:, 0].set(vx * jnp.cos(yaw) - vy * jnp.sin(yaw))
        f = f.at[:, 1].set(vx * jnp.sin(yaw) + vy * jnp.cos(yaw))

        # define g
        g = jnp.zeros((state.shape[0], self.state_dim, self.action_dim))

        # dx
        g = g.at[:, 3, 0].set(1.0)
        g = g.at[:, 4, 1].set(1.0)

        # dyaw
        g = g.at[:, 2, 2].set(1.0)

        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g
    
    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        # Use only x, y position
        edge_feats = state[graph.receivers, :2] - state[graph.senders, :2]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats * coef

        return graph._replace(edges=edge_feats, states=state)
    
    def get_graph(self, state: EnvState) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        # node type
        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(QuadrupedAccel.GOAL)
        node_type = node_type.at[-n_hits:].set(QuadrupedAccel.OBS)

        # edge blocks
        get_lidar_vmap = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )

        lidar_data = merge01(get_lidar_vmap(state.agent[:, :2]))  # Lidar only uses x, y
        # Pad lidar data with zeros for orientation to match state dimension
        lidar_data_padded = jnp.concatenate([lidar_data, jnp.zeros((lidar_data.shape[0], 1)), jnp.zeros((lidar_data.shape[0], 1)), jnp.zeros((lidar_data.shape[0], 1))], axis=1)
        edge_blocks = self.edge_blocks(state, lidar_data)
        
        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data_padded], axis=0),
        ).to_padded()
    
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.pi, -jnp.inf, -jnp.inf])
        upper_lim = jnp.array([jnp.inf, jnp.inf, jnp.pi, jnp.inf, jnp.inf])
        return lower_lim, upper_lim
    
    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(3) * -1.0
        upper_lim = jnp.ones(3)
        return lower_lim, upper_lim

    @ft.partial(jax.jit, static_argnums=(0,))
    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal  = graph.type_states(type_idx=1, n_type=self.num_agents)

        Q = jnp.expand_dims(jnp.diag(jnp.array([10,10,5,5,5])), 0).repeat(self.num_agents, 0)
        Qf = jnp.expand_dims(jnp.diag(jnp.array([100,100,50,50,50])), 0).repeat(self.num_agents, 0)
        R = jnp.expand_dims(jnp.diag(jnp.array([30,30,10])), 0).repeat(self.num_agents, 0)

        def forward_roll(x, u):
            # JIT these helpers too
            return self.agent_step_euler(x, u)

        def ilqr_iter(carry, _):
            x_ref, u_ref = carry

            # ----- backward pass over time with lax.scan -----
            def bwd_step(carry_b, k):
                Vx, Vxx = carry_b
                vx = x_ref[:, 3, k]; vy = x_ref[:, 4, k]; yaw = x_ref[:, 2, k]

                state_error = x_ref[:, :, k] - goal
                angle_error = state_error[:, 2]
                angle_error = jnp.arctan2(jnp.sin(angle_error), jnp.cos(angle_error))
                state_error = state_error.at[:, 2].set(angle_error)

                lx  = Q @ state_error[..., None]
                lu  = R @ u_ref[:, :, k][..., None]
                lxx = Q
                luu = R
                lxu = jnp.zeros((self.num_agents, self.state_dim, self.action_dim))

                fx = jnp.zeros((self.num_agents, self.state_dim, self.state_dim))
                fx = fx.at[:, 0, 2].set(-vx * jnp.sin(yaw) - vy * jnp.cos(yaw))
                fx = fx.at[:, 0, 3].set(jnp.cos(yaw))
                fx = fx.at[:, 0, 4].set(-jnp.sin(yaw))
                fx = fx.at[:, 1, 2].set(vx * jnp.cos(yaw) - vy * jnp.sin(yaw))
                fx = fx.at[:, 1, 3].set(jnp.sin(yaw))
                fx = fx.at[:, 1, 4].set(jnp.cos(yaw))
                fx = jnp.eye(self.state_dim).reshape((1, self.state_dim, self.state_dim)).repeat(self.num_agents, axis=0) + (fx * self.dt)

                fu = jnp.zeros((self.num_agents, self.state_dim, self.action_dim))
                fu = fu.at[:, 2, 2].set(1.0)
                fu = fu.at[:, 3, 0].set(1.0)
                fu = fu.at[:, 4, 1].set(1.0)
                fu = fu * self.dt

                Qx  = lx + jnp.transpose(fx, (0,2,1)) @ Vx
                Qu  = lu + jnp.transpose(fu, (0,2,1)) @ Vx
                Qxx = lxx + jnp.transpose(fx, (0,2,1)) @ Vxx @ fx
                Quu = luu + jnp.transpose(fu, (0,2,1)) @ Vxx @ fu + (jnp.eye(self.action_dim).reshape(1, self.action_dim, self.action_dim).repeat(self.num_agents, axis=0) * 0.01)
                # jax.debug.print("{x}", x=Quu)
                Qxu = lxu + jnp.transpose(fx, (0,2,1)) @ Vxx @ fu

                # batch Cholesky solve for stability
                def solve_batch(Quu_i, b_i):
                    L, lower = cho_factor(Quu_i)
                    return cho_solve((L, lower), b_i)

                k_ff = -jax.vmap(solve_batch)(Quu, Qu)
                k_fb = -jax.vmap(solve_batch)(Quu, jnp.transpose(Qxu, (0,2,1)))

                Vx_next  = Qx + Qxu @ k_ff
                Vxx_next = Qxx + Qxu @ k_fb

                return (Vx_next, Vxx_next), (k_ff.squeeze(-1), k_fb)  # save gains

            state_error_T = x_ref[:, :, -1] - goal
            angle_error_T = state_error_T[:, 2]
            angle_error_T = jnp.arctan2(jnp.sin(angle_error_T), jnp.cos(angle_error_T))
            state_error_T = state_error_T.at[:, 2].set(angle_error_T)

            Vx_T = (Qf @ state_error_T[..., None]).astype(jnp.float32)
            Vxx_T = Qf.astype(jnp.float32)
            ks, gains = lax.scan(
                f=bwd_step,
                init=(Vx_T, Vxx_T),
                xs=jnp.arange(self.PARAMS["time_horizon"]-1, -1, -1)
            )
            ff_gains, fb_gains = gains  # shapes: [T, N, A] and [T, N, A, X]
            ff_gains = ff_gains[::-1]; fb_gains = fb_gains[::-1]  # reverse
            # # ----- forward pass with lax.scan -----
            # def fwd_step(carry_f, k):
            #     xk = carry_f
            #     du = 0.1 * ff_gains[k] + (fb_gains[k] @ (xk - x_ref[:, :, k])[..., None]).squeeze(-1)
            #     uk = u_ref[:, :, k] + du
            #     xk1 = forward_roll(xk, uk)
            #     return xk1, (uk, xk1)
            alphas = jnp.logspace(0, -5, 30)

            def rollout_with_alpha(alpha):
                def fwd_step(xk, k):
                    du = alpha * ff_gains[k] + (fb_gains[k] @ (xk - x_ref[:, :, k])[..., None]).squeeze(-1)
                    uk = u_ref[:, :, k] + du
                    xk1 = forward_roll(xk, uk)
                    return xk1, (uk, xk1)

                x0 = x_ref[:, :, 0]
                _, (u_new_seq, x_new_seq) = lax.scan(fwd_step, x0, jnp.arange(self.PARAMS["time_horizon"]))
                x_ref_new = jnp.concatenate([x0[..., None], jnp.transpose(x_new_seq, (1,2,0))], axis=-1)
                u_ref_new = jnp.transpose(u_new_seq, (1,2,0))
                return x_ref_new, u_ref_new


            def compute_cost(xr, ur):
                # assume single agent for now (index 0)
                g0 = goal[0]  # shape (state_dim,)

                def cstep(carry, k):
                    j = carry
                    xk = xr[0, :, k] - g0       # distance to goal
                    uk = ur[0, :, k]

                    # Q, R are (5,5) and (3,3) or use Q[0], R[0] if batched
                    stage = xk.T @ Q[0] @ xk + uk.T @ R[0] @ uk
                    stage = jnp.squeeze(stage)
                    return j + stage, None

                j0 = 0.0
                T = ur.shape[-1]
                j, _ = lax.scan(cstep, j0, jnp.arange(T))

                xT = xr[0, :, T-1] - g0
                terminal = jnp.squeeze(xT.T @ Qf[0] @ xT)

                return j + terminal


            def try_alpha(alpha):
                xr, ur = rollout_with_alpha(alpha)
                cost = compute_cost(xr, ur)
                return cost, xr, ur

            costs, xs, us = jax.vmap(try_alpha)(alphas)

            best_idx = jnp.argmin(costs)
            x_ref_new = xs[best_idx]
            u_ref_new = us[best_idx]
            best_alpha = alphas[best_idx]
            # jax.debug.print("Chosen alpha = {a}", a=best_alpha)

            # x0 = x_ref[:, :, 0]
            # _, (u_new_seq, x_new_seq) = lax.scan(fwd_step, x0, jnp.arange(self.PARAMS["time_horizon"]))
            # x_ref_new = jnp.concatenate([x0[..., None], jnp.transpose(x_new_seq, (1,2,0))], axis=-1)
            # u_ref_new = jnp.transpose(u_new_seq, (1,2,0))
            
            j_new = 0.0
            g0 = goal[0]
            for k in range(self.PARAMS["time_horizon"]):
                x_err = x_ref_new[0, :, k] - g0
                u_k   = u_ref_new[0, :, k]
                j_new = j_new + x_err.T @ Q[0] @ x_err + u_k.T @ R[0] @ u_k

            x_err_T = x_ref_new[0, :, self.PARAMS["time_horizon"]-1] - g0
            j_new = j_new + x_err_T.T @ Qf[0] @ x_err_T
            # jax.debug.print("{x}", x=j_new)

            return (x_ref_new, u_ref_new), None

        T = self.PARAMS["time_horizon"]
        N = self.num_agents

        # check if warmstart exists
        if self.prev_x_ref is None:
            # cold start (zeros)
            u0 = jnp.zeros((N, self.action_dim, T))
            # rollout initial x0 using zero controls
            def init_rollout(xk, k):
                xk1 = self.agent_step_euler(xk, jnp.zeros((N, self.action_dim)))
                return xk1, xk1
            _, x_traj = lax.scan(init_rollout, agent, jnp.arange(T))
            x0 = jnp.concatenate([agent[..., None], jnp.transpose(x_traj, (1,2,0))], axis=-1)

        else:
            # warm start
            # Shift the previous optimized trajectory/control by 1 step
            prev_x = self.prev_x_ref
            prev_u = self.prev_u_ref

            # state warm-start: use prev_x[t+1] → prev_x[1:], pad last with last
            x0 = jnp.concatenate(
                [
                    agent[..., None],
                    prev_x[:, :, 1:],  # shift
                ],
                axis=-1
            )

            # action warm-start: use prev_u[t+1] → prev_u[1:], pad last with zeros
            u0 = jnp.concatenate(
                [
                    prev_u[:, :, 1:],                    # shifted warm start
                    jnp.zeros((N, self.action_dim, 1)),  # last control is zero
                ],
                axis=-1
            )
        (x_ref, u_ref), _ = lax.scan(ilqr_iter, (x0, u0), None, length=self.PARAMS["ilqr_iterations"])
        # jax.debug.print("{a}", a=x_ref[:,:3,:])
        # jax.debug.print("done")

        jax.debug.callback(self.add_traj, x_ref)
        # scale to normalized action space using solve instead of ad-hoc inv
        # breakpoint()
        u_norm0 = 2.0 * (u_ref[:, :, 0] - self.PARAMS["action_lower"]) / (self.PARAMS["action_upper"] - self.PARAMS["action_lower"]) - 1.0
        return self.clip_action(u_norm0)
    
    def add_traj(self, x_ref):
        x_ref_new = jax.device_get(x_ref)
        self.trajs.append(x_ref_new)
        # np.save('trajs.npy', np.array(self.trajs))

    # def u_ref(self, graph: GraphsTuple) -> Action:
    #     agent = graph.type_states(type_idx=0, n_type=self.num_agents) # num_agents, state_dim
    #     goal = graph.type_states(type_idx=1, n_type=self.num_agents) # num_agents, state_dim

    #     u_ref = jnp.zeros((self.num_agents, self.action_dim, self.PARAMS["time_horizon"])) # num_agents, action_dim, time_horizon
    #     x_ref = jnp.zeros((self.num_agents, self.state_dim, self.PARAMS["time_horizon"])) # num_agents, state_dim, time_horizon

    #     x_ref = x_ref.at[:, :, 0].set(agent)

    #     for k in range(self.PARAMS["time_horizon"]):
    #         x_ref = x_ref.at[:, :, k+1].set(self.agent_step_euler(x_ref[:, :, k], u_ref[:, :, k]))

    #     # Position error in global frame
    #     # state_error = jnp.zeros((self.num_agents, self.state_dim, self.PARAMS["time_horizon"])) # num_agents, state_dim, time_horizon
    #     # state_error = state_error.at[:, :, 0].set(agent - goal)

    #     Q = jnp.expand_dims(jnp.diag(jnp.array([100, 100, 1, 10, 10])), 0).repeat(self.num_agents, axis=0) # num_agents, state_dim, state_dim
    #     R = jnp.expand_dims(jnp.diag(jnp.array([100, 100, 100])), 0).repeat(self.num_agents, axis=0) # num_agents, action_dim, action_dim
        
    #     for i in range(self.PARAMS["ilqr_iterations"]):

    #         # V = 0.5 * jnp.transpose(jnp.expand_dims(x_ref[:, :, -1], -1), (0, 2, 1)) @ Q @ jnp.expand_dims(x_ref[:, :, -1], -1) # num_agents

    #         Vx = Q @ jnp.expand_dims(x_ref[:, :, -1] - goal, -1) # num_agents, state_dim, 1

    #         Vxx = Q # num_agents, state_dim, state_dim

    #         feedforward_gain = jnp.zeros((self.num_agents, self.action_dim, self.PARAMS["time_horizon"]))
    #         feedback_gain = jnp.zeros((self.num_agents, self.action_dim, self.state_dim, self.PARAMS["time_horizon"]))

    #         for k in range(self.PARAMS["time_horizon"] - 1, -1, -1):

    #             vx = x_ref[:, 3, k] # num_agents
    #             vy = x_ref[:, 4, k] # num_agents
    #             yaw = x_ref[:, 2, k] #num_agents

    #             lx = Q @ jnp.expand_dims(x_ref[:, :, k] - goal, -1) # num_agents, state_dim, 1
    #             lu = R @ jnp.expand_dims(u_ref[:, :, k], -1) # num_agents, action_dim
    #             lxx = Q # num_agents, state_dim, state_dim
    #             luu = R # num_agents, action_dim, action_dim
    #             lxu = jnp.zeros((self.num_agents, self.state_dim, self.action_dim)) # num_agents, state_dim, action_dim

    #             fx = jnp.zeros((self.num_agents, self.state_dim, self.state_dim)) # num_agents, state_dim, state_dim
    #             fx = fx.at[:, 0, 2].set(-vx * jnp.sin(yaw) - vy * jnp.cos(yaw))
    #             fx = fx.at[:, 0, 3].set(jnp.cos(yaw))
    #             fx = fx.at[:, 0, 4].set(-jnp.sin(yaw))

    #             fx = fx.at[:, 1, 2].set(vx * jnp.cos(yaw) - vy * jnp.sin(yaw))
    #             fx = fx.at[:, 1, 3].set(jnp.sin(yaw))
    #             fx = fx.at[:, 1, 4].set(jnp.cos(yaw))

    #             fu = jnp.zeros((self.num_agents, self.state_dim, self.action_dim)) # num_agents, state_dim, action_dim
    #             fu = fu.at[:, 2, 2].set(1)
    #             fu = fu.at[:, 3, 0].set(1)
    #             fu = fu.at[:, 4, 1].set(1)

    #             Qx = lx + jnp.transpose(fx, (0, 2, 1)) @ Vx # num_agents, state_dim, 1
    #             Qu = lu + jnp.transpose(fu, (0, 2, 1)) @ Vx # num_agents, action_dim, 1
    #             Qxx = lxx + jnp.transpose(fx, (0, 2, 1)) @ Vxx @ fx # num_agents, state_dim, state_dim
    #             Quu = luu + jnp.transpose(fu, (0, 2, 1)) @ Vxx @ fu # num_agents, action_dim, action_dim
    #             Qxu = lxu + jnp.transpose(fx, (0, 2, 1)) @ Vxx @ fu # num_agents, state_dim, action_dim

    #             # Vnext = -0.5 * jnp.transpose(Qu, (0, 2, 1)) @ jnp.linalg.inv(Quu) @ Qu # num_agents
    #             Vxnext = Qx - Qxu @ jnp.linalg.inv(Quu) @ Qu # num_agents, state_dim
    #             Vxxnext = Qxx - Qxu @ jnp.linalg.inv(Quu) @ jnp.transpose(Qxu, (0, 2, 1)) #num_agents, state_dim, state_dim

    #             feedforward_gain = feedforward_gain.at[:, :, k].set(jnp.squeeze(-jnp.linalg.inv(Quu) @ Qu, -1)) # num_agents, action_dim
    #             feedback_gain = feedback_gain.at[:, :, :, k].set(-jnp.linalg.inv(Quu) @ jnp.transpose(Qxu, (0, 2, 1))) # num_agents, action_dim, state_dim

    #             # V = Vnext 
    #             Vx = Vxnext
    #             Vxx = Vxxnext

    #         x_ref_new = jnp.zeros((self.num_agents, self.state_dim, self.PARAMS["time_horizon"])) # num_agents, state_dim, time_horizon
    #         x_ref_new = x_ref_new.at[:, :, 0].set(agent)
    #         # jax.debug.print('Ff_gain:{x}', x=feedforward_gain[:, :, 0])
    #         for k in range(self.PARAMS["time_horizon"]):
    #             u_ref = u_ref.at[:, :, k].set(u_ref[:, :, k] + feedforward_gain[:, :, k] + jnp.squeeze(feedback_gain[:, :, :, k] @ jnp.expand_dims(x_ref_new[:, :, k] - x_ref[:, :, k], -1), -1))
    #             x_ref_new = x_ref_new.at[:, :, k+1].set(self.agent_step_euler(x_ref_new[:, :, k], u_ref[:, :, k]))
            
    #         x_ref = x_ref_new

    #     # u_ref = u_ref.at[:, :, k - 1].set(jnp.squeeze(feedforward_gain + feedback_gain @ jnp.expand_dims(x_ref[:, :, k], -1), -1)) # num_agents, action_dim
    #     # breakpoint()
    #     lower, upper = jnp.array([-0.5 / 48.0, -0.5 / 48.0, -1.0]), jnp.array([0.5 / 48.0, 0.5 / 48.0, 1.0])
    #     u_norm = 2.0 * (u_ref[:, :, 0]  - lower) / (upper - lower) - 1.0

    #     # jax.debug.callback(self.save_plot, x_ref_new)
    #     # jax.debug.print('Action:{x}', x=u_ref)
    #     # jax.debug.print('State:{x}', x=x_ref)
    #     return self.clip_action(u_norm)
    
    def save_plot(self, x_ref):
        # breakpoint()
        x_ref_new = jax.device_get(x_ref)
        fig, axs = plt.subplots()
        axs.plot(x_ref_new[0, 0, :].transpose(), x_ref_new[0, 1, :].transpose())
        fig.savefig('test.png')
    
    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)

        return next_graph

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 2.5)
        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        unsafe_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return unsafe_mask

    def collision_mask(self, graph: GraphsTuple) -> Array:
        return self.unsafe_mask(graph)

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach
