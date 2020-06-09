"""Implementation of the DDPG algorithm."""

import copy

import gym
import numpy as np
import torch
from torch import nn
from torch import optim


class ReplayBuffer:

    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self._state = np.zeros((max_size, state_dim), dtype=np.float32)
        self._action = np.zeros((max_size, action_dim), dtype=np.float32)
        self._reward = np.zeros(max_size, dtype=np.float32)
        self._next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self._done = np.zeros(max_size, dtype=np.float32)
        self._ptr = 0
        self.size, self.max_size =  0, max_size

    def put(self, state, action, reward, next_state, done):
        self._state[self._ptr] = state
        self._action[self._ptr] = action
        self._reward[self._ptr] = reward
        self._next_state[self._ptr] = next_state
        self._done[self._ptr] = done
        self._ptr = (self._ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {
                'state': self._state[idxs],
                'action': self._action[idxs],
                'reward': self._reward[idxs], 
                'next_state': self._next_state[idxs],
                'done': self._done[idxs]
        }
        return {k: torch.from_numpy(v) for k, v in batch.items()}


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, output_activation_fn):
        super().__init__()
        self._net = nn.Sequential(
                nn.Linear(input_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, output_dim),
                output_activation_fn())

    def forward(self, state, action=None):
        x = state
        if action is not None:
            x = torch.cat((state, action), dim=1)
        return self._net(x)


def do_not_compute_gradients(parameters):
    for p in parameters:
        p.requires_grad = False


def do_compute_gradients(parameters):
    for p in parameters:
        p.requires_grad = True


def moving_avg(old, new, tau):
    for o, n in zip(old, new):
        o.data.copy_(tau * n.data + (1 - tau) * o.data)


def run_ddpg(
        env, 
        gamma=.99,
        action_noise=.1,
        tau=.005,
        batch_size=100, 
        episodes=1000, 
        max_episode_steps=1000,
        replay_buffer_size=int(1e6)):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = MLP(state_dim, action_dim, nn.Tanh)
    target_actor = copy.deepcopy(actor)
    do_not_compute_gradients(target_actor.parameters())
    actor_opt = optim.Adam(actor.parameters())

    critic = MLP(state_dim + action_dim, 1, nn.Identity)
    target_critic = copy.deepcopy(critic)
    do_not_compute_gradients(target_critic.parameters())
    critic_loss_fn = nn.MSELoss()
    critic_opt = optim.Adam(critic.parameters())

    replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_size)

    def get_action(state, action_noise):
        # Do not compute gradients when acting in the environment since we are
        # not training in this case.
        with torch.no_grad():
            state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            action = actor(state).numpy().reshape(action_dim,)
            action += np.random.normal(0., action_noise, size=(action_dim,))
            action *= env.action_space.high
            action = np.clip(action, env.action_space.low, env.action_space.high)
            return action

    def test():
        state, episode_steps, done, eps_reward = env.reset(), 0, False, 0 
        while not done:
            env.render()
            state, reward, done, _ = env.step(get_action(state, 0.))
            eps_reward += reward
            episode_steps += 1
            done = True if episode_steps >= max_episode_steps else done
        return eps_reward

    for episode in range(episodes):
        state, episode_steps, done, eps_reward = env.reset(), 0, False, 0.
        while not done:
            # Act in the environment and store transition in the replay buffer.
            action = get_action(state, action_noise)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.put(state, action, reward, next_state, done)
            state = next_state

            eps_reward += reward
            episode_steps += 1
            # End the episode if we've reached the max number of steps. N.B. 
            # that we should only update 'done' once we've already stored it in
            # the replay buffer.
            done = True if episode_steps >= max_episode_steps else done

            # Do not update the policy until we have taken at least batch_size
            # steps.
            if replay_buffer.size < batch_size * 2:
                continue

            # Update the actor and critic networks using an off-policy batch.
            batch = replay_buffer.sample_batch(batch_size)

            # Update cirtic.
            target_actions = target_actor(batch['next_state'])
            target_q = target_critic(batch['next_state'], target_actions).squeeze()
            targets = batch['reward'] + gamma * (1 - batch['done']) * target_q

            critic_opt.zero_grad()
            critic_predictions = critic(batch['state'], batch['action']).squeeze()
            critic_loss = critic_loss_fn(critic_predictions, targets)
            critic_loss.backward()
            critic_opt.step()

            # Update actor.
            # Freeze Q netowrk. We want to only update the parameters of the 
            # actor network while keeping the critic network parameters fixed.
            do_not_compute_gradients(critic.parameters())
            actor_opt.zero_grad()
            actor_loss = -1 * critic(batch['state'], actor(batch['state'])).mean()
            actor_loss.backward()
            actor_opt.step()
            do_compute_gradients(critic.parameters())

            # Update target networks.
            with torch.no_grad():
                moving_avg(target_actor.parameters(), actor.parameters(), tau)
                moving_avg(target_critic.parameters(), critic.parameters(), tau)

        test_reward = test()
        print(f'[{episode}]: train_reward={eps_reward} '
              f'test_reward={test_reward}  eps_steps={episode_steps}')


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    run_ddpg(env)
