import numpy as np
import gymnasium as gym
from Helper import LearningCurvePlot, smooth
from DQNAgent import DQNAgent
import os


def run_repetitions(outfile, n_repetitions, n_episodes, epsilon=0.1, alpha=0.1, gamma=1, update_freq=1, hidden_dim=128):
    print("Running repetitions with the following settings:")
    print(locals())

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    episode_returns = np.zeros((n_repetitions, n_episodes))

    for rep in range(n_repetitions):
        agent = DQNAgent(n_actions, n_states, epsilon, alpha, gamma, update_freq, hidden_dim)
        for ep in range(n_episodes):
            s, info = env.reset()
            if ep % 100 == 0:
                env_steps = int(episode_returns[rep].sum())
                print(f"Running repitition {rep+1:2}, Finished {ep:4} episodes, "
                      f"Environment steps: {env_steps:7}", end="\r")
            done = False
            while not done:
                a = agent.select_action(s)
                s_next, r, done, trunc, info = env.step(a)  # Simulate environment
                agent.update(s, a, r, s_next, done)
                episode_returns[rep, ep] += r
                s = s_next

    np.savetxt(outfile, episode_returns, delimiter=",")


def experiment_1():
    n_repetitions = 2
    n_episodes = 1000
    gamma = 1
    epsilon = 0.1
    update_freq = 10
    hidden_dim = 128

    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    smoothing_window = 31
    plot = LearningCurvePlot("Naive DQN learning curve")

    outdir = "naive_alpha"
    os.makedirs(outdir, exist_ok=True)

    for alpha in learning_rates:
        outfile = os.path.join(outdir, f"alpha_{alpha}.csv")
        if not os.path.exists(outfile):
            run_repetitions(outfile, n_repetitions, n_episodes, epsilon, alpha, gamma, update_freq, hidden_dim)
        episode_returns = np.loadtxt(outfile, delimiter=",", ndmin=2)
        mean_episode_returns = np.mean(episode_returns, axis=0)

        plot.add_curve(range(1, n_episodes+1), smooth(mean_episode_returns,
                       window=smoothing_window), label=f"α = {alpha}")

    plot.save(name="naive_dqn_learning_curve_alpha")


def experiment_2():
    n_repetitions = 2
    n_episodes = 1000
    gamma = 1
    epsilon = 0.1
    alpha = 0.001
    hidden_dim = 128

    update_freqs = [1, 10, 100]

    smoothing_window = 31
    plot = LearningCurvePlot("Naive DQN learning curve")

    outdir = "naive_update_freq"
    os.makedirs(outdir, exist_ok=True)

    for update_freq in update_freqs:
        outfile = os.path.join(outdir, f"update_freq_{update_freq}.csv")
        if not os.path.exists(outfile):
            run_repetitions(outfile, n_repetitions, n_episodes, epsilon, alpha, gamma, update_freq, hidden_dim)
        episode_returns = np.loadtxt(outfile, delimiter=",", ndmin=2)
        mean_episode_returns = np.mean(episode_returns, axis=0)

        plot.add_curve(range(1, n_episodes+1), smooth(mean_episode_returns,
                       window=smoothing_window), label=f"update freq = {update_freq}")

    plot.save(name="naive_dqn_learning_curve_freq")


if __name__ == '__main__':
    experiment_1()
    experiment_2()
    # experiment_3()
