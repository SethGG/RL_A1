import numpy as np
import gymnasium as gym
from Helper import LearningCurvePlot, smooth
from DQNAgent import DQNAgent
import os
import torch.multiprocessing as mp


def evaluation(agent: DQNAgent):
    env = gym.make('CartPole-v1')
    s, info = env.reset()
    done = False
    trunc = False
    episode_return = 0
    while not done and not trunc:
        a = agent.select_action(s, -1)  # greedy evaluation
        s_next, r, done, trunc, info = env.step(a)
        episode_return += r
        s = s_next
    return episode_return


def run_single_repetition(args):
    rep_id, n_envsteps, eval_internal, epsilon, alpha, gamma, update_freq, hidden_dim = args
    # Create a new environment and agent for each repetition.
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    eval_returns = np.zeros(int(n_envsteps / eval_internal))
    agent = DQNAgent(n_actions, n_states, alpha, gamma, update_freq, hidden_dim)

    done = True
    trunc = True
    eval_num = 0
    for step in range(1, n_envsteps+1):
        if done or trunc:
            s, info = env.reset()
        a = agent.select_action(s, epsilon)
        s_next, r, done, trunc, info = env.step(a)
        agent.update(s, a, r, s_next, done)
        s = s_next

        if step % eval_internal == 0:
            eval_return = evaluation(agent)
            eval_returns[eval_num] = eval_return
            eval_num += 1

            print(f"Running repitition {rep_id+1:2}, Environment steps: {step:7}, Eval return: {eval_return:3}")

    return eval_returns


def run_repetitions_multiprocessing(outfile, n_processes, n_repetitions, n_envsteps, eval_interval, epsilon=0.1,
                                    alpha=0.1, gamma=1, update_freq=1, hidden_dim=128):
    print("Running repetitions with the following settings:")
    print(locals())

    # Create a list of argument tuples for each repetition.
    args_list = [(rep, n_envsteps, eval_interval, epsilon, alpha, gamma, update_freq, hidden_dim)
                 for rep in range(n_repetitions)]

    # Use a multiprocessing Pool to run each repetition in parallel.
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(run_single_repetition, args_list)

    eval_returns = np.array(results)
    np.savetxt(outfile, eval_returns, delimiter=",")


def experiment_1(n_repetitions, n_envsteps, eval_interval):
    gamma = 1
    epsilon = 0.1
    update_freq = 4
    hidden_dim = 128

    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    smoothing_window = 31
    plot = LearningCurvePlot("Naive DQN learning curve")

    outdir = "evaluations"
    os.makedirs(outdir, exist_ok=True)

    n_processes = 3  # set the number of processes for parallel execution

    for alpha in learning_rates:
        outfile = os.path.join(outdir, f"a_{alpha}_uf_{update_freq}_e_{epsilon}_hd_{hidden_dim}.csv")
        if not os.path.exists(outfile):
            run_repetitions_multiprocessing(outfile, n_processes, n_repetitions, n_envsteps, eval_interval, epsilon,
                                            alpha, gamma, update_freq, hidden_dim)
        eval_returns = np.loadtxt(outfile, delimiter=",", ndmin=2)
        mean_eval_returns = np.mean(eval_returns, axis=0)
        conf_eval_returns = np.std(eval_returns, axis=0) / np.sqrt(n_repetitions)

        plot.add_curve(range(eval_interval, n_envsteps+eval_interval, eval_interval), smooth(mean_eval_returns,
                       window=smoothing_window), smooth(conf_eval_returns, window=smoothing_window), label=f"α = {alpha}")

    plot.save(name="naive_dqn_learning_curve_alpha")


def experiment_2(n_repetitions, n_envsteps, eval_interval):
    gamma = 1
    epsilon = 0.1
    alpha = 0.001
    update_freq = 4
    hidden_dim = 128

    update_freqs = [4, 16, 64]

    smoothing_window = 31
    plot = LearningCurvePlot("Naive DQN learning curve")

    outdir = "evaluations"
    os.makedirs(outdir, exist_ok=True)

    n_processes = 3  # set the number of processes for parallel execution

    for update_freq in update_freqs:
        outfile = os.path.join(outdir, f"a_{alpha}_uf_{update_freq}_e_{epsilon}_hd_{hidden_dim}.csv")
        if not os.path.exists(outfile):
            run_repetitions_multiprocessing(outfile, n_processes, n_repetitions, n_envsteps, eval_interval, epsilon,
                                            alpha, gamma, update_freq, hidden_dim)
        eval_returns = np.loadtxt(outfile, delimiter=",", ndmin=2)
        mean_eval_returns = np.mean(eval_returns, axis=0)
        conf_eval_returns = np.std(eval_returns, axis=0) / np.sqrt(n_repetitions)

        plot.add_curve(range(eval_interval, n_envsteps+eval_interval, eval_interval), smooth(mean_eval_returns,
                       window=smoothing_window), smooth(conf_eval_returns, window=smoothing_window),
                       label=f"update_freq = {update_freq}")

    plot.save(name="naive_dqn_learning_curve_update_freq")


def experiment_3(n_repetitions, n_envsteps, eval_interval):
    gamma = 1
    epsilon = 0.1
    alpha = 0.001
    update_freq = 4
    hidden_dim = 128

    epsilons = [0.05, 0.1, 0.2, 0.5]

    smoothing_window = 31
    plot = LearningCurvePlot("Naive DQN learning curve")

    outdir = "evaluations"
    os.makedirs(outdir, exist_ok=True)

    n_processes = 3  # set the number of processes for parallel execution

    for epsilon in epsilons:
        outfile = os.path.join(outdir, f"a_{alpha}_uf_{update_freq}_e_{epsilon}_hd_{hidden_dim}.csv")
        if not os.path.exists(outfile):
            run_repetitions_multiprocessing(outfile, n_processes, n_repetitions, n_envsteps, eval_interval, epsilon,
                                            alpha, gamma, update_freq, hidden_dim)
        eval_returns = np.loadtxt(outfile, delimiter=",", ndmin=2)
        mean_eval_returns = np.mean(eval_returns, axis=0)
        conf_eval_returns = np.std(eval_returns, axis=0) / np.sqrt(n_repetitions)

        plot.add_curve(range(eval_interval, n_envsteps+eval_interval, eval_interval), smooth(mean_eval_returns,
                       window=smoothing_window), smooth(conf_eval_returns, window=smoothing_window),
                       label=f"ϵ = {epsilon}")

    plot.save(name="naive_dqn_learning_curve_epsilon")


if __name__ == '__main__':
    n_repetitions = 5
    n_envsteps = 200000
    eval_interval = 1000

    experiment_1(n_repetitions, n_envsteps, eval_interval)
    experiment_2(n_repetitions, n_envsteps, eval_interval)
    experiment_3(n_repetitions, n_envsteps, eval_interval)
