"""
Cross-Entropy Method training for SlimeVolley-v0.

No ML framework needed — only numpy + slimevolleygym.

Usage:
    uv run src/train.py              # 20 iterations, then visualise
    uv run src/train.py --iters 50   # more training
    uv run src/train.py --no-vis     # headless only
"""

import argparse

import numpy as np

from slimevolleygym import BaselinePolicy, SlimeVolleyEnv

# ── Policy: tiny 2-layer MLP ────────────────────────────────────────────────
LAYER_SIZES = [12, 32, 3]  # obs → hidden → actions (MultiBinary)


def n_params(sizes: list[int]) -> int:
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def forward(params: np.ndarray, obs: np.ndarray) -> list[int]:
    x, ptr = obs.astype(np.float32), 0
    for i in range(len(LAYER_SIZES) - 1):
        r, c = LAYER_SIZES[i], LAYER_SIZES[i + 1]
        W = params[ptr : ptr + r * c].reshape(r, c)
        ptr += r * c
        b = params[ptr : ptr + c]
        ptr += c
        x = np.tanh(x @ W + b)
    return (x > 0).astype(int).tolist()


# ── Episode rollout ──────────────────────────────────────────────────────────
def rollout(params: np.ndarray, seed: int = 0, render: bool = False) -> float:
    env = SlimeVolleyEnv(render_mode="human" if render else None)
    baseline = BaselinePolicy()
    obs, _ = env.reset(seed=seed)
    obs_left, done, total = obs, False, 0.0
    baseline.reset()
    while not done:
        obs, r, term, trunc, info = env.step(
            forward(params, obs), baseline.predict(obs_left)
        )
        obs_left, total, done = info["otherObs"], total + r, term or trunc
        if render:
            env.render()
    env.close()
    return total


def evaluate(params: np.ndarray, n: int = 3) -> float:
    return float(np.mean([rollout(params, seed=i) for i in range(n)]))


# ── Cross-Entropy Method ─────────────────────────────────────────────────────
def train(n_iters: int = 20, pop: int = 20, elite_frac: float = 0.25) -> np.ndarray:
    dim = n_params(LAYER_SIZES)
    n_elite = max(2, int(pop * elite_frac))
    mu = np.zeros(dim)
    sigma = np.ones(dim) * 0.5

    print(f"CEM  iters={n_iters}  pop={pop}  elite={n_elite}  params={dim}\n")

    for it in range(1, n_iters + 1):
        population = mu + sigma * np.random.randn(pop, dim)
        scores = [evaluate(p) for p in population]
        elite = population[np.argsort(scores)[-n_elite:]]
        mu, sigma = elite.mean(0), elite.std(0) + 1e-5
        print(
            f"iter {it:3d}/{n_iters}  best {max(scores):+.1f}  mean {np.mean(scores):+.2f}"
        )

    return mu


# ── Visualisation ────────────────────────────────────────────────────────────
def visualise(params: np.ndarray, n_episodes: int = 3) -> None:
    print(f"\nVisualising {n_episodes} episode(s) — close window to exit.\n")
    for ep in range(n_episodes):
        score = rollout(params, seed=200 + ep, render=True)
        winner = "agent" if score > 0 else ("baseline" if score < 0 else "draw")
        print(f"  episode {ep + 1}: score {score:+.0f}  [{winner}]")


# ── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--vis-eps", type=int, default=3)
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    best = train(args.iters, args.pop)

    if not args.no_vis:
        visualise(best, args.vis_eps)


if __name__ == "__main__":
    main()
