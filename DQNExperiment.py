import numpy as np
import gymnasium as gym
from Helper import LearningCurvePlot, smooth
from DQNAgent import DQNAgent


def run_repetitions(n_repetitions, n_episodes, epsilon=0.1, alpha=0.1, gamma=1, update_freq=1):
    print("Running repetitions with the following settings:")
    print(locals())

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    episode_returns = np.zeros((n_repetitions, n_episodes))

    for rep in range(n_repetitions):
        agent = DQNAgent(n_actions, n_states, epsilon, alpha, gamma, update_freq)
        for ep in range(n_episodes):
            s, info = env.reset()
            if ep % 100 == 0:
                print(f"Running repitition {rep+1:2}, Finished {ep:4} episodes", end="\r")
            done = False
            while not done:
                a = agent.select_action(s)
                s_next, r, done, trunc, info = env.step(a)  # Simulate environment
                agent.update(s, a, r, s_next, done)
                episode_returns[rep, ep] += r
                s = s_next

    # Compute average evaluations over all repetitions
    mean_episode_returns = np.mean(episode_returns, axis=0)
    return mean_episode_returns


def experiment_1():
    n_repetitions = 5
    n_episodes = 1000
    gamma = 1
    epsilon = 0.1
    update_freq = 4

    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    smoothing_window = 31
    plot = LearningCurvePlot("Naive DQN learning curve")
    for alpha in learning_rates:
        eval_returns = run_repetitions(n_repetitions, n_episodes, epsilon, alpha, gamma, update_freq)
        plot.add_curve(range(1, n_episodes+1), smooth(eval_returns, window=smoothing_window), label=f"α = {alpha}")
    plot.save(name="naive_dqn_learning_curve")


if __name__ == '__main__':
    experiment_1()
