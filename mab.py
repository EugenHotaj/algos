"""Multi Armed Bandits."""

from absl import app
import numpy as np

NUM_STEPS = 10000


class MultiArmedBandit(object):
  """A Multi Armed Bandit implementation using sample averages."""

  def __init__(self, num_arms, eps=0.1):
    """Constructor."""
    self._num_arms = num_arms
    self._eps = eps
    self._q = np.zeros(num_arms)
    self._n = np.zeros(num_arms)

  def act(self):
    """Returns the epsilon-greedy action."""
    if np.random.uniform() < self._eps:
      return np.random.randint(self._num_arms)
    return np.argmax(self._q)

  def update(self, action, reward):
    """Updates the Q values."""
    self._n[action] += 1
    self._q[action] += 1 / self._n[action] * (reward - self._q[action])


def main(argv):
  del argv  # Unused

  rewards = np.random.normal(0, 1, size=(4,))
  total_reward = 0
  mab = MultiArmedBandit(4)
  for i in range(NUM_STEPS):
    action = mab.act()
    reward = rewards[action]
    total_reward += reward
    mab.update(action, reward)

  print("Rewards: ", rewards)
  print("Action Histogram:", mab._n)
  print("Q Values:", mab._q)
  print("Avg Reward:", total_reward / NUM_STEPS)


if __name__ == "__main__":
  app.run(main)
