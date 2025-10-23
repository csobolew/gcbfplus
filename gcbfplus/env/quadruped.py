import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pathlib
import functools as ft

from .base import MultiAgentEnv, RolloutResult
from ..utils.typing import State, Array, AgentState, Action, Reward, Cost, Done, Info, Pos2d
from ..utils.graph import GraphsTuple, EdgeBlock, GetGraph
from .obstacle import Obstacle, Rectangle
from typing import NamedTuple, Tuple, Optional
from ..utils.utils import merge01, jax_vmap
from .plot import render_video
from .utils import get_node_goal_rng, inside_obstacles, get_lidar

class Quadruped(MultiAgentEnv):
    
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: State
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]
        
    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.6],
        "n_obs": 8,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int=256,
            max_travel: float=None,
            dt: float=0.03,
            params: dict=None
    ):
        super(Quadruped, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)

        self.create_obstacles = jax_vmap(Rectangle.create)

    @property
    def state_dim(self) -> int:
        return 3 # x, y, yaw
    
    @property
    def node_dim(self) -> int:
        return 3 # indicator: agent: 001, goal: 010, obstacle: 100
    
    @property
    def edge_dim(self) -> int:
        return 2 # x_rel, y_rel
    
    @property
    def action_dim(self) -> int:
        return 3 # vx, vy, yawrate
    
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

        states = jnp.concatenate([states_pos, agent_orientations], axis=1)
        goals = jnp.concatenate([goals_pos, goal_orientations], axis=1)

        env_states = self.EnvState(states, goals, obstacles)

        return self.get_graph(env_states)
    
    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        # Get current state information
        x = agent_states[:, 0]
        y = agent_states[:, 1]
        yaw = agent_states[:, 2]

        # Get action information
        vx = action[:, 0]
        vy = action[:, 1]
        yaw_rate = action[:, 2]

        # Kinematics
        x_dot = vx * jnp.cos(yaw) - vy * jnp.sin(yaw)
        y_dot = vx * jnp.sin(yaw) + vy * jnp.cos(yaw)

        # Euler integration
        x_new = x + x_dot * self.dt
        y_new = y + y_dot * self.dt
        yaw_new = yaw + yaw_rate * self.dt

        # Clip to [-pi, pi]
        yaw_new = jnp.arctan2(jnp.sin(yaw_new), jnp.cos(yaw_new))

        # Clip position to area
        n_state_agent_new = jnp.stack((x_new, y_new, yaw_new), axis=1)
        n_state_agent_new = self.clip_state(n_state_agent_new)
        
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)

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

        yaw = state[:, 2]

        # f is 0, no drift
        f = jnp.zeros_like(state)

        # define g
        g = jnp.zeros((state.shape[0], self.state_dim, self.action_dim))

        # dx
        g = g.at[:, 0, 0].set(jnp.cos(yaw))
        g = g.at[:, 0, 1].set(-jnp.sin(yaw))

        # dy
        g = g.at[:, 1, 0].set(jnp.sin(yaw))
        g = g.at[:, 1, 1].set(jnp.cos(yaw))

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
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(Quadruped.GOAL)
        node_type = node_type.at[-n_hits:].set(Quadruped.OBS)

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
        lidar_data_padded = jnp.concatenate([lidar_data, jnp.zeros((lidar_data.shape[0], 1))], axis=1)
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
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.pi])
        upper_lim = jnp.array([jnp.inf, jnp.inf, jnp.pi])
        return lower_lim, upper_lim
    
    def action_lim(self, action: Optional[Action] = None) -> Tuple[Action, Action]:
        lower_lim = jnp.array([-2.0, -1.5, -2.0])   
        upper_lim = jnp.array([3.0, 1.5, 2.0])     
        return lower_lim, upper_lim
    
    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        
        # Position error in global frame
        pos_error = goal[:, :2] - agent[:, :2]
        
        # Clip error to avoid too large references (similar to original)
        error_norm = jnp.linalg.norm(pos_error, axis=-1, keepdims=True)
        error_max = self._params["comm_radius"]
        error_clipped = jnp.clip(pos_error, -error_max, error_max)
        
        # Compute desired velocity in global frame
        vel_global = error_clipped
        
        # Transform to body frame
        psi = agent[:, 2]
        cos_psi = jnp.cos(psi)
        sin_psi = jnp.sin(psi)
        
        vx_desired = vel_global[:, 0] * cos_psi + vel_global[:, 1] * sin_psi
        vy_desired = -vel_global[:, 0] * sin_psi + vel_global[:, 1] * cos_psi
        
        # Orientation control: align with movement direction or goal orientation
        # If moving, align with velocity direction; otherwise align with goal
        vel_mag = jnp.sqrt(vel_global[:, 0]**2 + vel_global[:, 1]**2 + 1e-6)
        desired_heading = jnp.arctan2(vel_global[:, 1], vel_global[:, 0])
        
        # Use goal orientation when close to goal, movement direction otherwise
        weight = jnp.clip(vel_mag / (self._params["car_radius"] * 4), 0.0, 1.0)
        target_psi = weight * desired_heading + (1 - weight) * goal[:, 2]
        
        psi_error = target_psi - agent[:, 2]
        psi_error = jnp.arctan2(jnp.sin(psi_error), jnp.cos(psi_error))
        psi_dot_desired = psi_error * 2.0
        
        action = jnp.stack([vx_desired, vy_desired, psi_dot_desired], axis=1)
        return self.clip_action(action)
    
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