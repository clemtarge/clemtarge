
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

from typing import Any, List, Sequence, Tuple


class A2C:
    def __init__(self, env: gym.Env, reward_type: str):
        self.env = env
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)  # loss for critic
        self.eps = np.finfo(np.float32).eps.item()
        self.reward_types_available = {"int32": (np.int32, tf.int32),
                                       "float32": (np.float32, tf.float32),
                                       "int16": (np.int16, tf.int16),
                                       "float16": (np.float16, tf.float16)}
        self.reward_types = self.reward_types_available[reward_type]  # = (np.dtype, tf.dtype)

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, self.reward_types[0]),
                np.array(done, np.int32))

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action], [tf.float32, self.reward_types[1], tf.int32])

    def get_expected_return(self, rewards: tf.Tensor,
                            gamma: float,
                            standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.eps)

        return returns


class A2CDiscrete(A2C):
    def __init__(self, env: gym.Env, reward_type: str):
        super().__init__(env, reward_type)

    def run_episode(self, initial_state: tf.Tensor,
                    model: tf.keras.Model,
                    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Runs a single episode to collect training data."""

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=self.reward_types[1], size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = model(state)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state = tf.reshape(state, initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def compute_loss(self, action_probs: tf.Tensor,
                     values: tf.Tensor,
                     returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    @tf.function
    def train_step(self, initial_state: tf.Tensor,
                   model: tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   gamma: float,
                   max_steps_per_episode: int) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(initial_state, model, max_steps_per_episode)

            # Calculate expected returns
            returns = self.get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward


class A2CContinuous(A2C):
    def __init__(self, env: gym.Env, reward_type: str):
        super().__init__(env, reward_type)

    def run_episode(self, initial_state: tf.Tensor,
                    model: tf.keras.Model,
                    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Runs a single episode to collect training data."""

        actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        mus = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        sigmas = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=self.reward_types[1], size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action, norm_dist, value = model(state)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            actions = actions.write(t, action[0, 0])

            # Store mu and sigma of the chosen action
            mus = mus.write(t, norm_dist.loc)
            sigmas = sigmas.write(t, norm_dist.scale)

            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)

            state = tf.reshape(state, initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        actions = actions.stack()
        mus = mus.stack()
        sigmas = sigmas.stack()
        values = values.stack()
        rewards = rewards.stack()

        return actions, mus, sigmas, values, rewards

    def compute_loss(self, actions: tf.Tensor,
                     mus: tf.Tensor,
                     sigmas: tf.Tensor,
                     values: tf.Tensor,
                     returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        norm_dists = tfp.distributions.Normal(loc=mus, scale=sigmas)
        actor_loss = -tf.math.reduce_sum((norm_dists.log_prob(actions) + 1e-8) * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    @tf.function
    def train_step(self, initial_state: tf.Tensor,
                   model: tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   gamma: float,
                   max_steps_per_episode: int) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            actions, mus, sigmas, values, rewards = self.run_episode(initial_state, model, max_steps_per_episode)

            # Calculate expected returns
            returns = self.get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            actions, mus, sigmas, values, returns = [tf.expand_dims(x, 1) for x in [actions, mus, sigmas, values, returns]]

            # Calculating loss values to update our network
            loss = self.compute_loss(actions, mus, sigmas, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward


class ActorCriticModelDiscrete(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int):
        super().__init__()

        self.hiddens = [layers.Dense(128, activation="relu"),
                        ]

        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = inputs
        for hidden in self.hiddens:
            x = hidden(x)

        return self.actor(x), self.critic(x)


class ActorCriticModelContinuous(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int):
        super().__init__()

        self.hiddens = [layers.Dense(32, activation="relu"),
                        layers.Dense(32, activation="relu")
                        ]

        self.mu = layers.Dense(num_actions, activation="tanh")
        self.sigma = layers.Dense(num_actions, activation="softplus")

        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = inputs
        for hidden in self.hiddens:
            x = hidden(x)
        mu = self.mu(x)
        sigma = self.sigma(x) + 1e-8

        norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)

        action_tf_var = tf.clip_by_value(action_tf_var, -1., 1.)
        return action_tf_var, norm_dist, self.critic(x)
