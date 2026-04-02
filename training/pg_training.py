import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import SepsisEnv

# policy gradient training script for ICU sepsis environment
# runs PPO and REINFORCE with different hyperparameters

SAVE_DIR_PPO = "models/pg/ppo"
SAVE_DIR_REINFORCE = "models/pg/reinforce"
LOG_DIR = "logs/pg"

for d in [SAVE_DIR_PPO, SAVE_DIR_REINFORCE, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 100_000
N_EVAL_EPISODES = 20

# create monitored environment
def make_env():
    return Monitor(SepsisEnv())

# PPO hyperparameter combinations
PPO_PARAMS = [
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01},
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01},
    {"learning_rate": 3e-4, "gamma": 0.95, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.05},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.0},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "clip_range": 0.1, "ent_coef": 0.01},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "clip_range": 0.3, "ent_coef": 0.01},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512, "batch_size": 128, "n_epochs": 15, "clip_range": 0.2, "ent_coef": 0.01},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 512, "batch_size": 128, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.02},
]

# REINFORCE hyperparameter combinations
REINFORCE_PARAMS = [
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 100, "ent_coef": 0.01, "gae_lambda": 1.0},
    {"learning_rate": 5e-3, "gamma": 0.99, "n_steps": 100, "ent_coef": 0.01, "gae_lambda": 1.0},
    {"learning_rate": 1e-2, "gamma": 0.99, "n_steps": 100, "ent_coef": 0.01, "gae_lambda": 1.0},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 100, "ent_coef": 0.01, "gae_lambda": 1.0},
    {"learning_rate": 1e-3, "gamma": 0.95, "n_steps": 100, "ent_coef": 0.01, "gae_lambda": 1.0},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 100, "ent_coef": 0.05, "gae_lambda": 1.0},
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 100, "ent_coef": 0.0, "gae_lambda": 1.0},
    {"learning_rate": 1e-3, "gamma": 0.999, "n_steps": 100, "ent_coef": 0.01, "gae_lambda": 1.0},
    {"learning_rate": 2e-3, "gamma": 0.99, "n_steps": 50, "ent_coef": 0.01, "gae_lambda": 1.0},
    {"learning_rate": 5e-3, "gamma": 0.99, "n_steps": 100, "ent_coef": 0.02, "gae_lambda": 1.0},
]

def train_ppo(run_id, params):
    print(f"\n[PPO] Run {run_id} | LR={params['learning_rate']:.0e} | "
          f"gamma={params['gamma']} | clip={params['clip_range']} | "
          f"ent={params['ent_coef']} | steps={params['n_steps']}")

    env = make_env()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
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

    model.save(os.path.join(SAVE_DIR_PPO, f"ppo_run{run_id:02d}"))

    print(f"Mean reward: {mean_r:.2f} ± {std_r:.2f}")

    env.close()

    return {"run": run_id, **params,
            "mean_reward": round(float(mean_r), 2),
            "std_reward": round(float(std_r), 2)}

def train_reinforce(run_id, params):
    print(f"\n[REINFORCE] Run {run_id} | LR={params['learning_rate']:.0e} | "
          f"gamma={params['gamma']} | ent={params['ent_coef']} | steps={params['n_steps']}")

    env = make_env()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        n_steps=params["n_steps"],
        batch_size=params["n_steps"],
        n_epochs=1,
        clip_range=1.0,
        ent_coef=params["ent_coef"],
        gae_lambda=params["gae_lambda"],
        vf_coef=0.0,
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

    model.save(os.path.join(SAVE_DIR_REINFORCE, f"reinforce_run{run_id:02d}"))

    print(f"Mean reward: {mean_r:.2f} ± {std_r:.2f}")

    env.close()

    return {"run": run_id, **params,
            "mean_reward": round(float(mean_r), 2),
            "std_reward": round(float(std_r), 2)}

def print_summary(algo, results):
    best = max(results, key=lambda x: x["mean_reward"])

    print(f"\n{algo} Summary")
    for r in results:
        mark = " *" if r["run"] == best["run"] else ""
        print(f"Run {r['run']} | {r['mean_reward']} ± {r['std_reward']}{mark}")

    print(f"Best run: {best['run']} with reward {best['mean_reward']}")

def plot_comparison(results_dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colours = {"PPO": "steelblue", "REINFORCE": "green"}

    for ax, (algo, results) in zip(axes, results_dict.items()):
        runs = [r["run"] for r in results]
        means = [r["mean_reward"] for r in results]
        stds = [r["std_reward"] for r in results]

        best_run = max(results, key=lambda x: x["mean_reward"])["run"]
        cols = ["gold" if r == best_run else colours[algo] for r in runs]

        bars = ax.bar(runs, means, yerr=stds, capsize=4, color=cols)

        ax.set_title(f"{algo} runs")
        ax.axhline(0, linestyle="--")
        ax.bar_label(bars, fmt="%.1f")

    path = os.path.join(LOG_DIR, "pg_comparison.png")
    fig.savefig(path)
    plt.close()

    print(f"Plot saved to {path}")

def main():
    print("\nTraining PPO")
    ppo_results = [train_ppo(i + 1, p) for i, p in enumerate(PPO_PARAMS)]

    print_summary("PPO", ppo_results)

    print("\nTraining REINFORCE")
    rf_results = [train_reinforce(i + 1, p) for i, p in enumerate(REINFORCE_PARAMS)]

    print_summary("REINFORCE", rf_results)

    all_results = {"PPO": ppo_results, "REINFORCE": rf_results}

    with open(os.path.join(LOG_DIR, "pg_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    plot_comparison(all_results)

    print("\nTraining complete")

if __name__ == "__main__":
    main()