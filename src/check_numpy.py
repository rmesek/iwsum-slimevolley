"""
check_numpy.py — NumPy 2.0+ compatibility check for slimevolleygym.

Verifies that:
  • NumPy ≥ 2.0 is active (uses APIs added / moved in 2.0)
  • slimevolleygym produces correctly-typed arrays under NumPy 2.x
  • No legacy 1.x-only APIs are relied on

Run:
    uv run src/check_numpy.py
"""

import sys

import numpy as np

# ── 1. Version guard ──────────────────────────────────────────────────────────
major, minor = (int(x) for x in np.__version__.split(".")[:2])
assert major >= 2, f"NumPy >=2.0 required; got {np.__version__}"
print(f"Python {sys.version[:6]}   NumPy {np.__version__}")
print()

# ── 2. APIs added in NumPy 2.0 ───────────────────────────────────────────────
print("── NumPy 2.0 additions ──────────────────────────────────")

# numpy.exceptions: AxisError and friends moved out of the top-level namespace
from numpy.exceptions import AxisError  # noqa: E402

try:
    np.zeros(3)[5]
except IndexError:
    pass  # plain indexing raises IndexError, not AxisError
try:
    np.sum(np.zeros((3, 3)), axis=5)
except AxisError as e:
    print(f"  numpy.exceptions.AxisError caught correctly: {e}")

# np.linalg.vecdot — batched inner product along the last axis (new in 2.0)
a = np.array([[1.0, 0.0], [0.0, 1.0]])
b = np.array([[3.0, 4.0], [3.0, 4.0]])
dots = np.linalg.vecdot(a, b)  # [1*3+0*4, 0*3+1*4] = [3, 4]
np.testing.assert_array_equal(dots, [3.0, 4.0])
print(f"  np.linalg.vecdot([1,0],[3,4]) = {dots}   ✓")

# np.matrix_transpose — transposes the last two axes (new in 2.0)
mat = np.arange(6, dtype=np.float32).reshape(2, 3)
transposed = np.matrix_transpose(mat)  # (3, 2)
assert transposed.shape == (3, 2)
print(f"  np.matrix_transpose shape {mat.shape} → {transposed.shape}   ✓")

# np.unstack — inverse of np.stack (new in 2.0)
parts = np.unstack(np.arange(12).reshape(3, 4), axis=0)  # 3 arrays of shape (4,)
assert len(parts) == 3 and parts[0].shape == (4,)
print(f"  np.unstack: {len(parts)} arrays of shape {parts[0].shape}   ✓")

# ── 3. APIs removed in NumPy 2.0 ─────────────────────────────────────────────
print()
print("── NumPy 2.0 removals ───────────────────────────────────")

# np.find_common_type removed → use np.result_type
assert not hasattr(np, "find_common_type"), "np.find_common_type should be gone"
common = np.result_type(np.float32, np.float64)
assert common == np.float64
print(f"  np.result_type(float32, float64) = {common}   ✓")

# np.AxisError moved to numpy.exceptions
assert not hasattr(np, "AxisError"), "np.AxisError should be gone from top-level"
print("  np.AxisError removed from top-level   ✓")

# ── 4. slimevolleygym rollout + dtype checks ──────────────────────────────────
print()
print("── slimevolleygym compatibility ─────────────────────────")

from slimevolleygym import BaselinePolicy, SlimeVolleyEnv  # noqa: E402

env = SlimeVolleyEnv()
baseline = BaselinePolicy()

obs, info = env.reset(seed=0)
baseline.reset()

# Observation must be float32 — explicit in the modern gymnasium obs space
assert obs.dtype == np.float32, f"Expected float32 obs, got {obs.dtype}"
assert obs.shape == (12,)
print(f"  reset() obs  dtype={obs.dtype}  shape={obs.shape}   ✓")

# Collect one episode
obs_list: list[np.ndarray] = []
reward_list: list[float] = []
done = False
while not done:
    action = baseline.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    obs_list.append(obs)
    reward_list.append(reward)
    done = terminated or truncated

env.close()

# ── 5. NumPy 2.0 operations on rollout data ───────────────────────────────────
obs_arr = np.stack(obs_list)  # (T, 12) float32
rewards_arr = np.array(reward_list, dtype=np.float64)  # (T,)    float64

# np.unstack to split the 12-dim observation into individual fields
fields = np.unstack(obs_arr, axis=-1)  # 12 arrays of shape (T,)
assert len(fields) == 12 and fields[0].shape == (len(obs_list),)
print(f"  np.unstack obs into {len(fields)} fields of shape {fields[0].shape}   ✓")

# np.linalg.vecdot: cosine-similarity between consecutive obs pairs
if len(obs_arr) >= 2:
    pairs_a = obs_arr[:-1]  # (T-1, 12)
    pairs_b = obs_arr[1:]  # (T-1, 12)
    dots = np.linalg.vecdot(pairs_a, pairs_b)  # (T-1,)
    norms = np.linalg.norm(pairs_a, axis=1) * np.linalg.norm(pairs_b, axis=1)
    cosines = np.where(norms > 0, dots / norms, 0.0)
    print(
        f"  cosine similarity (consecutive obs): "
        f"mean={cosines.mean():.3f}  min={cosines.min():.3f}   ✓"
    )

# np.result_type to find the promoted dtype before mixing float32 obs + float64 rewards
promoted = np.result_type(obs_arr.dtype, rewards_arr.dtype)
assert promoted == np.float64
print(f"  np.result_type(obs float32, rewards float64) = {promoted}   ✓")

# Memory ownership: copy() vs view
obs_view = obs_arr[:]
obs_copy = obs_arr.copy()
assert np.shares_memory(obs_view, obs_arr), "Slice should share memory"
assert not np.shares_memory(obs_copy, obs_arr), ".copy() must own its data"
print("  copy / view memory ownership   ✓")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print()
print(f"  Episode length : {len(obs_list)} steps")
print(f"  Total reward   : {rewards_arr.sum():+.0f}")
print(f"  Obs mean       : {obs_arr.mean():.4f}")
print(f"  Obs std        : {obs_arr.std():.4f}")
print()
print("All checks passed ✓")
