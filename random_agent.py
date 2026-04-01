import sys, os, time, argparse
# Add project folder to Python path so imports from environment/ work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the custom ICU Sepsis environment and rendering utilities
from environment.custom_env import SepsisEnv, ACTION_NAMES
from environment.rendering import ICURenderer


def run_random_agent(n_episodes=3, render=True):
    # Create environment instance
    env = SepsisEnv()
    # Optional GUI renderer (Pygame patient monitor)
    renderer = ICURenderer() if render else None

    for ep in range(1, n_episodes + 1):
        # Reset environment to start new episode
        obs, info = env.reset()
        total_reward = 0.0
        step = 0
        terminated = truncated = False

        # Episode header
        print(f"\n{'='*65}")
        print(f"  EPISODE {ep} — Random Agent (no model)")
        print(f"{'='*65}")
        print(f"  Start → HR:{env.heart_rate:.1f} | BP:{env.blood_pressure:.1f} | "
              f"O2:{env.oxygen:.1f} | Lac:{env.lactate:.2f} | Inf:{env.infection:.2f}")
        print(f"  {'─'*63}")
        print(f"  {'Step':>4} | {'Action':<20} | {'HR':>5} | {'BP':>5} | "
              f"{'O2':>5} | {'Lac':>5} | {'Inf':>5} | {'Rew':>7}")
        print(f"  {'─'*63}")

        # Loop until episode ends (death, recovery, or timeout)
        while not (terminated or truncated):
            # Select a random action from the action space
            action = env.action_space.sample()
            # Apply action, receive new observation and reward
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Print step info in terminal
            print(f"  {step:>4} | {ACTION_NAMES[action]:<20} | "
                  f"{info['heart_rate']:>5.1f} | "
                  f"{info['blood_pressure']:>5.1f} | "
                  f"{info['oxygen']:>5.1f} | "
                  f"{info['lactate']:>5.2f} | "
                  f"{info['infection']:>5.2f} | "
                  f"{reward:>+7.1f}")

            # Draw the GUI dashboard if enabled
            if renderer:
                # Determine status for display
                status = ("RECOVERED" if terminated and total_reward > -100
                          else "DEATH" if terminated
                          else "TIMEOUT" if truncated
                          else "ONGOING")
                renderer.draw(
                    hr=info['heart_rate'],
                    bp=info['blood_pressure'],
                    o2=info['oxygen'],
                    lac=info["lactate"],
                    inf=info["infection"],
                    t=info["time"],
                    action=ACTION_NAMES[int(action)],
                    status=status,
                )
                # Small delay for human-readable GUI updates
                time.sleep(0.06)

        # Episode summary
        outcome = ("RECOVERED" if terminated and total_reward > -100
                   else "DEATH" if terminated
                   else "TIMEOUT")
        print(f"\n  Outcome: {outcome} | Steps: {step} | Total Reward: {total_reward:.1f}")

    # Close environment and GUI
    env.close()
    if renderer:
        renderer.close()

    print("\nRandom agent demo complete.")


if __name__ == "__main__":
    # CLI arguments for number of episodes and GUI toggle
    parser = argparse.ArgumentParser(description="Run random ICU Sepsis agent demo")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable Pygame GUI")
    args = parser.parse_args()

    # Run the random agent with user-specified options
    run_random_agent(n_episodes=args.episodes, render=not args.no_render)