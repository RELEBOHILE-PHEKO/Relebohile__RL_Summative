import gymnasium as gym
from gymnasium import spaces
import numpy as np

# environment for ICU sepsis treatment
# the agent learns how to stabilise a patient by controlling vital signs

ACTION_NAMES = {
    0: "Do Nothing",
    1: "IV Fluids",
    2: "Antibiotics",
    3: "Oxygen Therapy",
    4: "Vasopressors",
}

# healthy ranges for each vital
TARGETS = {
    "heart_rate": (70, 100),
    "blood_pressure": (110, 130),
    "oxygen": (95, 100),
    "lactate": (0, 2),
    "infection": (0, 2),
}

# calculates how far a value is from a healthy range
def _dist_to_range(value, lo, hi):
    if lo <= value <= hi:
        return 0.0
    return float(min(abs(value - lo), abs(value - hi)))


class SepsisEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.renderer = None

        # 5 possible treatment actions
        self.action_space = spaces.Discrete(5)

        # observation includes vitals + time, all normalised
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

        # used for scaling values between 0 and 1
        self._raw_low = np.array([30, 50, 70, 0, 0, 0], dtype=np.float32)
        self._raw_high = np.array([180, 200, 100, 10, 10, 100], dtype=np.float32)

        self._prev_dist = None
        self.reset()

    # converts raw values to a 0–1 range
    def _normalise(self, raw):
        return (raw - self._raw_low) / (self._raw_high - self._raw_low + 1e-8)

    def _get_obs(self):
        raw = np.array([
            self.heart_rate,
            self.blood_pressure,
            self.oxygen,
            self.lactate,
            self.infection,
            float(self.time),
        ], dtype=np.float32)

        return np.clip(self._normalise(raw), 0.0, 1.0)

    # initialise a new patient in an unstable condition
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.heart_rate = float(np.random.uniform(100, 130))
        self.blood_pressure = float(np.random.uniform(75, 100))
        self.oxygen = float(np.random.uniform(88, 94))
        self.lactate = float(np.random.uniform(2, 5))
        self.infection = float(np.random.uniform(4, 7))
        self.time = 0

        self._prev_dist = self._total_distance()
        return self._get_obs(), {}

    # total deviation from healthy ranges
    def _total_distance(self):
        vitals = [
            (self.heart_rate, *TARGETS["heart_rate"]),
            (self.blood_pressure, *TARGETS["blood_pressure"]),
            (self.oxygen, *TARGETS["oxygen"]),
            (self.lactate, *TARGETS["lactate"]),
            (self.infection, *TARGETS["infection"]),
        ]
        return sum(_dist_to_range(v, lo, hi) for v, lo, hi in vitals)

    def step(self, action):
        self.time += 1

        # natural progression of illness
        self.heart_rate += np.random.normal(0.0, 0.5)
        self.blood_pressure -= np.random.normal(0.3, 0.3)
        self.oxygen -= np.random.normal(0.1, 0.1)
        self.lactate += np.random.normal(0.05, 0.05)
        self.infection -= np.random.uniform(0.0, 0.1)

        # treatment effects
        if action == 0:
            self.heart_rate -= np.random.uniform(0.0, 0.5)

        elif action == 1:
            self.blood_pressure += np.random.uniform(4, 8)
            self.lactate -= np.random.uniform(0.3, 0.7)
            self.heart_rate -= np.random.uniform(0.5, 2.0)

        elif action == 2:
            self.infection -= np.random.uniform(1.0, 2.0)
            self.lactate -= np.random.uniform(0.2, 0.5)
            self.heart_rate -= np.random.uniform(0.5, 1.5)

        elif action == 3:
            self.oxygen += np.random.uniform(3, 6)
            self.heart_rate -= np.random.uniform(0.5, 2.0)

        elif action == 4:
            self.blood_pressure += np.random.uniform(6, 12)
            self.heart_rate += np.random.uniform(1, 3)

        # keep values within realistic limits
        self.heart_rate = float(np.clip(self.heart_rate, 30, 180))
        self.blood_pressure = float(np.clip(self.blood_pressure, 50, 200))
        self.oxygen = float(np.clip(self.oxygen, 70, 100))
        self.lactate = float(np.clip(self.lactate, 0, 10))
        self.infection = float(np.clip(self.infection, 0, 10))

        # reward based on improvement towards healthy ranges
        curr_dist = self._total_distance()
        progress = self._prev_dist - curr_dist
        reward = progress * 1.5

        # small penalty to encourage faster recovery
        reward -= 0.5

        # bonus for each vital in healthy range
        in_range = [
            TARGETS["heart_rate"][0] <= self.heart_rate <= TARGETS["heart_rate"][1],
            TARGETS["blood_pressure"][0] <= self.blood_pressure <= TARGETS["blood_pressure"][1],
            TARGETS["oxygen"][0] <= self.oxygen <= TARGETS["oxygen"][1],
            TARGETS["lactate"][0] <= self.lactate <= TARGETS["lactate"][1],
            TARGETS["infection"][0] <= self.infection <= TARGETS["infection"][1],
        ]

        reward += sum(in_range) * 2.0

        # penalties for dangerous states
        if self.blood_pressure < 75:
            reward -= (75 - self.blood_pressure) * 0.3

        if self.oxygen < 88:
            reward -= (88 - self.oxygen) * 0.5

        self._prev_dist = curr_dist

        terminated = False
        truncated = False
        recovered = False
        death = False

        # failure conditions
        if self.blood_pressure < 60 or self.oxygen < 80 or self.lactate > 9:
            reward -= 100.0
            terminated = True
            death = True

        # success condition
        elif all(in_range):
            reward += 200.0
            terminated = True
            recovered = True

        # max time limit
        if self.time >= 100:
            truncated = True

        info = {
            "heart_rate": round(self.heart_rate, 1),
            "blood_pressure": round(self.blood_pressure, 1),
            "oxygen": round(self.oxygen, 1),
            "lactate": round(self.lactate, 2),
            "infection": round(self.infection, 2),
            "time": self.time,
            "action_name": ACTION_NAMES[int(action)],
            "in_range_count": sum(in_range),
            "total_distance": round(curr_dist, 2),
            "recovered": recovered,
            "death": death,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            from environment.rendering import ICURenderer

            if self.renderer is None:
                self.renderer = ICURenderer()

            self.renderer.draw(
                hr=self.heart_rate,
                bp=self.blood_pressure,
                o2=self.oxygen,
                lac=self.lactate,
                inf=self.infection,
                t=self.time,
            )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None