import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import SepsisEnv

# DQN training script for ICU sepsis environment
# runs 10 different hyperparameter setups and compares performance

SAVE_DIR = "models/dqn"
LOG_DIR = "logs/dqn"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 100_000
N_EVAL_EPISODES = 20

# create monitored environment
def make_env():
    return Monitor(SepsisEnv())

# different hyperparameter combinations to test
HYPERPARAMS = [
    {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 100_000, "exploration_fraction": 0.15,
     "target_update_interval": 200, "train_freq": 1},

    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 100_000, "exploration_fraction": 0.15,
     "target_update_interval": 200, "train_freq": 1},

    {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 100_000, "exploration_fraction": 0.15,
     "target_update_interval": 200, "train_freq": 1},

    {"learning_rate": 5e-4, "gamma": 0.95, "batch_size": 64,
     "buffer_size": 100_000, "exploration_fraction": 0.15,
     "target_update_interval": 200, "train_freq": 1},

    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 256,
     "buffer_size": 100_000, "exploration_fraction": 0.15,
     "target_update_interval": 200, "train_freq": 1},

    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 100_000, "exploration_fraction": 0.10,
     "target_update_interval": 200, "train_freq": 1},

    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 100_000, "exploration_fraction": 0.25,
     "target_update_interval": 200, "train_freq": 1},

    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 128,
     "buffer_size": 200_000, "exploration_fraction": 0.15,
     "target_update_interval": 100, "train_freq": 1},

    {"learning_rate": 3e-4, "gamma": 0.99, "batch_size": 128,
     "buffer_size": 100_000, "exploration_fraction": 0.12,
     "target_update_interval": 50, "train_freq": 1},

    {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 128,
     "buffer_size": 150_000, "exploration_fraction": 0.12,
     "target_update_interval": 100, "train_freq": 1},
]

def train_dqn(run_id, params):
    print(f"\n{'-'*65}")
    print(f"Run {run_id} | LR={params['learning_rate']:.0e} | "
          f"gamma={params['gamma']} | batch={params['batch_size']} | "
          f"buffer={params['buffer_size']} | expl={params['exploration_fraction']}")
    print(f"{'-'*65}")

    env = make_env()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        buffer_size=params["buffer_size"],
        exploration_fraction=params["exploration_fraction"],
        exploration_final_eps=0.02,
        target_update_interval=params["target_update_interval"],
        train_freq=params["train_freq"],
        learning_starts=1000,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=42,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    mean_r, std_r = evaluate_policy(
        model,
        make_env(),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True
    )

    path = os.path.join(SAVE_DIR, f"dqn_run{run_id:02d}")
    model.save(path)

    print(f"Mean reward: {mean_r:.2f} ± {std_r:.2f}")

    env.close()

    return {
        "run": run_id,
        **params,
        "mean_reward": round(float(mean_r), 2),
        "std_reward": round(float(std_r), 2),
        "model_path": path,
    }

# plot comparison of all runs
def plot_results(results):
    best = max(results, key=lambda x: x["mean_reward"])

    runs = [r["run"] for r in results]
    means = [r["mean_reward"] for r in results]
    stds = [r["std_reward"] for r in results]

    colours = ["gold" if r["run"] == best["run"] else "steelblue"
               for r in results]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(runs, means, yerr=stds, capsize=5,
                  color=colours, edgecolor="white")

    ax.set_xlabel("Run Number")
    ax.set_ylabel("Mean Reward")
    ax.set_title("DQN Hyperparameter Comparison (Sepsis)")

    ax.axhline(0, linestyle="--")
    ax.bar_label(bars, fmt="%.1f")

    path = os.path.join(LOG_DIR, "dqn_comparison.png")
    fig.savefig(path)
    plt.close()

    print(f"Plot saved to {path}")

def main():
    print("\nStarting DQN training on ICU Sepsis environment\n")

    results = []
    for i, params in enumerate(HYPERPARAMS):
        results.append(train_dqn(i + 1, params))

    with open(os.path.join(LOG_DIR, "dqn_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x["mean_reward"])

    print("\nSummary\n")
    for r in results:
        mark = " *" if r["run"] == best["run"] else ""
        print(f"Run {r['run']} | reward={r['mean_reward']} ± {r['std_reward']}{mark}")

    print(f"\nBest run: {best['run']} with reward {best['mean_reward']}")

    plot_results(results)

    print("\nTraining complete")

if __name__ == "__main__":
    main()