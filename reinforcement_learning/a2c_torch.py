
import gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim

from typing import Any, List, Sequence, Tuple


class A2C:
    def __init__(self, env: gym.Env, reward_type: str, gamma=0.95):
        self.env = env
        self.huber_loss = nn.HuberLoss(reduction="sum")  # loss for critic
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma: float = gamma    

    def get_expected_return(self, rewards: torch.Tensor,
                            standardize: bool = True) -> torch.Tensor:
        """Compute expected returns per timestep."""

        n = rewards.shape[0]
        returns = torch.zeros_like(rewards)

        # Start from the end of `rewards` and accumulate reward sums into the `returns` array
        rewards = torch.flip(rewards, dims=(0,))
        discounted_sum = torch.Tensor([0.])
        for i in torch.arange(0, n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            returns[i] = discounted_sum
        returns = torch.flip(returns, dims=(0,))

        if standardize:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)

        return returns


class A2CDiscrete(A2C):
    def __init__(self, env: gym.Env, n_actions, reward_type: str, gamma=0.95):
        super().__init__(env, reward_type, gamma)
        self.model = ActorCriticModelDiscrete(env.observation_space.shape, n_actions=n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

    def run_episode(self, initial_state: torch.Tensor,
                    max_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs a single episode to collect training data."""

        action_probs = torch.Tensor()
        values = torch.Tensor()
        rewards = torch.Tensor()

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in torch.arange(0, max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = torch.unsqueeze(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = self.model(state)

            # Sample next action from the action probability distribution           
            action = torch.distributions.Categorical(logits=action_logits_t).sample().item()
            action_probs_t = nn.Softmax(dim=1)(action_logits_t)

            # Apply action to the environment to get next state and reward
            state, reward, done, _, _ = self.env.step(action)
            state = torch.reshape(torch.from_numpy(state), initial_state_shape)

            # Store critic values
            values = torch.cat((values, value[0]), dim=0)
            
            # Store log probability of the action chosen
            action_probs = torch.cat((action_probs, action_probs_t[0, [action]]), dim=0)

            # Store reward
            rewards = torch.cat((rewards, torch.Tensor([reward])), dim=0)

            if done:
                break

        return action_probs, values, rewards

    def compute_loss(self, action_probs: torch.Tensor,
                     values: torch.Tensor,
                     returns: torch.Tensor) -> torch.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = torch.log(action_probs)
        actor_loss = -torch.sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    def train_step(self, initial_state: torch.Tensor,
                   max_steps_per_episode: int) -> torch.Tensor:
        """Runs a model training step."""

        # Run the model for one episode to collect training data
        action_probs, values, rewards = self.run_episode(torch.from_numpy(initial_state), max_steps_per_episode)

        # Calculate expected returns
        returns = self.get_expected_return(rewards, self.gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [torch.unsqueeze(x, 1) for x in [action_probs, values, returns]]

        # Calculating loss values to update our network
        loss = self.compute_loss(action_probs, values, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        episode_reward = torch.sum(rewards)

        return episode_reward


class A2CContinuous(A2C):
    def __init__(self, env: gym.Env, n_actions, reward_type: str, gamma=0.95):
        super().__init__(env, reward_type, gamma)
        self.model = ActorCriticModelContinuous(env.observation_space.shape, n_actions=n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def run_episode(self, initial_state: torch.Tensor,
                    max_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs a single episode to collect training data."""
        
        actions = torch.Tensor()
        mus = torch.Tensor()
        sigmas = torch.Tensor()
        values = torch.Tensor()
        rewards = torch.Tensor()

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in torch.arange(0, max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = torch.unsqueeze(state, 0)

            # Run the model and to get action probabilities and critic value
            action, norm_dist, value = self.model(state)

            # Store critic values
            values = torch.cat((values, value[0]), dim=0)

            # Store log probability of the action chosen
            actions = torch.cat((actions, action[0]), dim=0)

            # Store mu and sigma of the chosen action
            mus = torch.cat((mus, torch.Tensor([norm_dist.loc])), dim=0)
            sigmas = torch.cat((sigmas, torch.Tensor([norm_dist.scale])), dim=0)

            # Apply action to the environment to get next state and reward
            state, reward, done, _, _ = self.env.step(action)
            state = torch.reshape(torch.from_numpy(state), initial_state_shape)

            # Store reward
            rewards = torch.cat((rewards, torch.Tensor([reward])), dim=0)

            if done:
                break

        return actions, mus, sigmas, values, rewards

    def compute_loss(self, actions: torch.Tensor,
                     mus: torch.Tensor,
                     sigmas: torch.Tensor,
                     values: torch.Tensor,
                     returns: torch.Tensor) -> torch.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        norm_dists = torch.distributions.Normal(loc=mus, scale=sigmas)#.sample()
        actor_loss = -torch.sum((norm_dists.log_prob(actions) + 1e-8) * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss


    def train_step(self, initial_state: torch.Tensor,
                   max_steps_per_episode: int) -> torch.Tensor:
        """Runs a model training step."""

        # Run the model for one episode to collect training data
        actions, mus, sigmas, values, rewards = self.run_episode(torch.from_numpy(initial_state), max_steps_per_episode)

        # Calculate expected returns
        returns = self.get_expected_return(rewards, self.gamma)

        # Convert training data to appropriate TF tensor shapes
        actions, mus, sigmas, values, returns = [torch.unsqueeze(x, 1) for x in [actions, mus, sigmas, values, returns]]

        # Calculating loss values to update our network
        loss = self.compute_loss(actions, mus, sigmas, values, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        episode_reward = torch.sum(rewards)

        return episode_reward


class ActorCriticModelDiscrete(nn.Module):
    """Combined actor-critic network."""

    def __init__(self, input_dims, n_actions: int):
        super().__init__()
        
        self.hiddens = nn.Sequential(
            nn.Linear(input_dims[0], 128),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
            
        x = self.hiddens(state)

        return self.actor(x), self.critic(x)


class ActorCriticModelContinuous(nn.Module):
    """Combined actor-critic network."""

    def __init__(self, input_dims, n_actions: int):
        super().__init__()

        self.hiddens = nn.Sequential(
            nn.Linear(input_dims[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(32, n_actions),
            nn.Tanh(),
        )
        self.sigma = nn.Sequential(
            nn.Linear(32, n_actions),
            nn.Softplus(),
        )
        
        self.critic = nn.Linear(32, 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        x = self.hiddens(inputs)
        
        mu = self.mu(x)
        sigma = self.sigma(x) + 1e-8

        norm_dist = torch.distributions.Normal(loc=mu, scale=sigma)
        
        # action_tf_var = torch.squeeze(norm_dist.sample(), dim=0)
        action_tf_var = norm_dist.sample()
        action_tf_var = torch.clip(action_tf_var, -1., 1.)
        
        return action_tf_var, norm_dist, self.critic(x)
    