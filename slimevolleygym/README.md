# slimevolleygym

A **Gymnasium**-compatible port of the classic [Slime Volleyball](https://otoro.net/slimevolley/) game. Two slimes hit a ball back and forth over a fence; the agent's goal is to make the ball land on the opponent's side.

> Originally created by David Ha (2020) for OpenAI Gym.  
> This fork modernises the environment to the **Gymnasium** API (≥ 0.29).

---

## Environments

| ID | Observation | Action space |
|----|-------------|--------------|
| `SlimeVolley-v0` | `Box(12,)` state vector | `MultiBinary(3)` |
| `SlimeVolleyPixel-v0` | `Box(84, 168, 3)` RGB | `MultiBinary(3)` |
| `SlimeVolleyNoFrameskip-v0` | `Box(84, 168, 3)` RGB | `Discrete(6)` |
| `SlimeVolleySurvivalNoFrameskip-v0` | `Box(84, 168, 3)` RGB | `Discrete(6)` + survival bonus |

The 12-dimensional state vector is:

```
(x_agent, y_agent, vx_agent, vy_agent,
 x_ball,  y_ball,  vx_ball,  vy_ball,
 x_opp,   y_opp,   vx_opp,   vy_opp)
```

All values are from the **right** agent's perspective; the observation is
mirrored automatically so the same policy works on either side.

---

## Installation

### Add to a uv project

```bash
# from your project root, add the local slimevolleygym directory
uv add ./slimevolleygym
```

### Install standalone (editable)

```bash
uv pip install -e ./slimevolleygym
```

### Install standalone (regular)

```bash
uv pip install ./slimevolleygym
```

---

## Basic usage

```python
import gymnasium
import slimevolleygym  # registers the envs

env = gymnasium.make("SlimeVolley-v0")  # no visualisation
obs, info = env.reset()

done = False
total_reward = 0
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print("score:", total_reward)
```

---

## Running the visualisation

Pass `render_mode="human"` when creating the environment.
A **pygame** window (1200 × 500) opens automatically on the first `env.render()` call.

```python
import gymnasium
import slimevolleygym

env = gymnasium.make("SlimeVolley-v0", render_mode="human")
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()   # draws the current frame

env.close()
```

To get raw RGB arrays instead (useful for recording or headless servers):

```python
env = gymnasium.make("SlimeVolley-v0", render_mode="rgb_array")
...
frame = env.render()   # returns (500, 1200, 3) uint8 array
```

---

## Multi-agent / self-play

Pass a second action to `env.step()`.  
The left agent's observation is returned in `info['otherObs']`.

```python
import gymnasium
import slimevolleygym
from slimevolleygym import BaselinePolicy

env = gymnasium.make("SlimeVolley-v0", render_mode="human")

policy_right = BaselinePolicy()
policy_left  = BaselinePolicy()

obs_right, info = env.reset()
obs_left = obs_right   # both agents receive the same first observation

done = False
score = 0
while not done:
    action_right = policy_right.predict(obs_right)
    action_left  = policy_left.predict(obs_left)

    obs_right, reward, terminated, truncated, info = env.step(action_right, action_left)
    obs_left = info['otherObs']
    done = terminated or truncated
    score += reward
    env.render()

env.close()
print(f"right agent score: {score:+.0f}")
```

`info` keys:

| Key | Description |
|-----|-------------|
| `ale.lives` | Right agent remaining lives |
| `ale.otherLives` | Left agent remaining lives |
| `otherObs` | Left agent's observation (from its perspective) |
| `state` | Right agent's 12-dim state (same as obs in state mode) |
| `otherState` | Left agent's 12-dim state |

---

## Reproducibility / seeding

```python
obs, info = env.reset(seed=42)   # gymnasium-style seeding
```

---

## Wrappers

```python
from slimevolleygym import SurvivalRewardEnv, FrameStack

# +0.01 per timestep survived
env = SurvivalRewardEnv(gymnasium.make("SlimeVolley-v0"))

# Stack last 4 pixel frames (Atari-style)
env = FrameStack(gymnasium.make("SlimeVolleyPixel-v0"), n_frames=4)
```

---

## Citation

```bibtex
@misc{slimevolleygym,
  author       = {David Ha},
  title        = {Slime Volleyball Gym Environment},
  year         = {2020},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/hardmaru/slimevolleygym}},
}
```
