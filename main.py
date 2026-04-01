import sys, os, argparse, time, json
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import SepsisEnv, ACTION_NAMES
from environment.rendering import ICURenderer

# paths to saved models
MODEL_PATHS = {
    "dqn": "models/dqn",
    "ppo": "models/pg/ppo",
    "reinforce": "models/pg/reinforce",
}

# RL classes
ALGO_CLASS = {
    "dqn": DQN,
    "ppo": PPO,
    "reinforce": PPO,
}

# log files for results
LOG_FILES = {
    "dqn": "logs/dqn/dqn_results.json",
    "pg": "logs/pg/pg_results.json",
}


def find_best_model():
    # find model with highest reward across saved results
    best_reward = -np.inf
    best_algo = None
    best_run = None

    # check DQN results
    if os.path.exists(LOG_FILES["dqn"]):
        results = json.load(open(LOG_FILES["dqn"]))
        for r in results:
            if r["mean_reward"] > best_reward:
                best_reward = r["mean_reward"]
                best_algo = "dqn"
                best_run = r["run"]

    # check PG results (PPO/Reinforce)
    if os.path.exists(LOG_FILES["pg"]):
        results = json.load(open(LOG_FILES["pg"]))
        for algo, runs in results.items():
            for r in runs:
                if r["mean_reward"] > best_reward:
                    best_reward = r["mean_reward"]
                    best_algo = algo.lower()
                    best_run = r["run"]

    return best_algo, best_run, best_reward


def load_model(algo, run_id):
    # load saved model and attach a fresh env
    path = os.path.join(MODEL_PATHS[algo], f"{algo}_run{run_id:02d}")
    return ALGO_CLASS[algo].load(path, env=Monitor(SepsisEnv()))


def run_simulation(model, n_episodes=3, render=True):
    env = SepsisEnv()
    renderer = ICURenderer() if render else None

    rewards = []
    lengths = []
    outcomes = {"RECOVERED": 0, "DEATH": 0, "TIMEOUT": 0}

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        print(f"\nEpisode {episode}/{n_episodes}")
        print(f"Start state: HR={env.heart_rate:.1f}, BP={env.blood_pressure:.1f}, "
              f"O2={env.oxygen:.1f}, Lactate={env.lactate:.2f}, Infection={env.infection:.2f}")

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            total_reward += reward
            steps += 1

            # log step info
            print(f"Step {steps:>3} | {ACTION_NAMES[int(action)]:<20} | "
                  f"HR {info['heart_rate']:>5.1f} | "
                  f"BP {info['blood_pressure']:>5.1f} | "
                  f"O2 {info['oxygen']:>5.1f} | "
                  f"Lac {info['lactate']:>5.2f} | "
                  f"Inf {info['infection']:>5.2f} | "
                  f"Reward {reward:>+6.1f}")

            # render if enabled
            if renderer:
                if terminated:
                    # use info flag if available, otherwise fall back to reward heuristic
                    if info.get("recovered", False):
                        status = "RECOVERED"
                    elif info.get("death", False):
                        status = "DEATH"
                    else:
                        status = "RECOVERED" if reward > 0 else "DEATH"
                elif truncated:
                    status = "TIMEOUT"
                else:
                    status = "ONGOING"

                renderer.draw(
                    hr=info["heart_rate"],
                    bp=info["blood_pressure"],
                    o2=info["oxygen"],
                    lac=info["lactate"],
                    inf=info["infection"],
                    t=info["time"],
                    action=ACTION_NAMES[int(action)],
                    status=status,
                )
                time.sleep(0.08)

        # final outcome for episode — check info flags first
        if terminated:
            if info.get("recovered", False):
                outcome = "RECOVERED"
            elif info.get("death", False):
                outcome = "DEATH"
            else:
                # fallback: last reward positive = recovered
                outcome = "RECOVERED" if reward > 0 else "DEATH"
        else:
            outcome = "TIMEOUT"

        outcomes[outcome] += 1
        rewards.append(total_reward)
        lengths.append(steps)

        print(f"Outcome: {outcome} | Steps: {steps} | Total reward: {total_reward:.1f}")

    # summary
    print("\nSimulation finished")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Recovered: {outcomes['RECOVERED']}/{n_episodes}")
    print(f"Deaths: {outcomes['DEATH']}/{n_episodes}")
    print(f"Timeouts: {outcomes['TIMEOUT']}/{n_episodes}")

    env.close()
    if renderer:
        renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo", "reinforce"])
    parser.add_argument("--run", type=int)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")

    args = parser.parse_args()
    render = not args.no_render

    # pick model
    if args.algo is None or args.run is None:
        print("Searching for best model...")
        algo, run_id, reward = find_best_model()

        if algo is None:
            print("No trained models found. Run training first.")
            sys.exit(1)

        print(f"Best model: {algo.upper()} run {run_id} (reward {reward:.2f})")
    else:
        algo = args.algo
        run_id = args.run

    print(f"Loading model: {algo.upper()} run {run_id}")

    try:
        model = load_model(algo, run_id)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("Model loaded")
    print(f"Algorithm: {algo.upper()} | Episodes: {args.episodes} | Render: {'on' if render else 'off'}")

    run_simulation(model, n_episodes=args.episodes, render=render)


if __name__ == "__main__":
    main()