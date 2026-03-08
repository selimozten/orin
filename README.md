# orin

RL gym for training LLMs on financial text reasoning.

## Quick Start

```python
import gymnasium as gym
import orin

env = gym.make("orin/FinText-Earnings-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step({"direction": 2, "confidence": 0.8})
```

## Installation

```bash
pip install orin
```

With all extras (yfinance, pufferlib, HuggingFace datasets):

```bash
pip install orin[all]
```

## Environments

| Environment | Input | Action | Reward Signal |
|---|---|---|---|
| `orin/FinText-Earnings-v0` | Earnings call transcripts | Predict stock direction | Post-earnings price move |
| `orin/FinText-News-v0` | Financial news | Predict market sentiment | Actual market reaction |
| `orin/FinText-Filing-v0` | SEC filings (10-K, 10-Q, 8-K) | Predict filing impact | Price move after filing |
| `orin/FinText-Macro-v0` | Economic data, Fed speeches | Predict market regime | Actual market behavior |

### Observations

```python
obs = {
    "text": "AAPL Q3 earnings call transcript...",
    "metadata": {"ticker": "AAPL", "date": "2024-08-01", "source": "earnings_call"},
}
```

### Actions

```python
action = {
    "direction": 2,       # 0=down, 1=flat, 2=up
    "confidence": 0.8,    # float in [0, 1]
}
```

## Reward System

Rewards combine directional accuracy and confidence calibration:

- **Directional**: +1 for correct direction, -1 for wrong, scaled by magnitude
- **Calibration**: high confidence + correct = high reward; high confidence + wrong = high penalty
- **Composite**: weighted sum (default 0.7 directional + 0.3 calibration)

```python
from orin.rewards import CompositeReward

custom_reward = CompositeReward(directional_weight=0.5, calibration_weight=0.5)
env = gym.make("orin/FinText-Earnings-v0", reward_fn=custom_reward)
```

## PufferLib

```python
from orin.wrappers.pufferlib import make_pufferlib_env

env = make_pufferlib_env("orin/FinText-Earnings-v0", max_tokens=512)
obs, info = env.reset()  # obs is now a flat int32 array
action = env.action_space.sample()  # single discrete action
```

## Custom Data

Load your own data as JSONL:

```python
from orin.data.loaders import load_jsonl

data = load_jsonl("my_earnings.jsonl")
env = gym.make("orin/FinText-Earnings-v0", data=data)
```

Expected format per line:

```json
{"text": "...", "ticker": "AAPL", "date": "2024-01-01", "outcome": {"direction": "up", "magnitude": 0.03, "timeframe": "1d"}}
```

## Philosophy

Can machines learn to read financial text and predict what happens next? This is the infrastructure to answer it, not the answer itself.

## License

MIT
