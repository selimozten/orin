# orin

RL gym for training LLMs on financial text reasoning.

## Why This Exists

There's no standardized reinforcement learning environment for training LLMs on financial text understanding. orin provides a gym-compatible API where an LLM reads financial text (earnings calls, SEC filings, news) and learns to make predictions, with rewards based on actual market outcomes.

Nobody is doing RL on financial NLP tasks in an open, reusable way.

## Core Design

### Gym API

```python
import gymnasium as gym
import orin

env = gym.make("orin/FinText-v0")

obs, info = env.reset()
# obs = {"text": "AAPL Q3 earnings call transcript...", "metadata": {...}}

action = agent.predict(obs)
# action = {"direction": "up", "confidence": 0.8, "reasoning": "..."}

obs, reward, terminated, truncated, info = env.step(action)
# reward based on actual market movement after the event
```

### PufferLib Compatibility

- Implement PufferLib environment wrapper
- Compatible with PufferLib's vectorized training
- Supports multi-agent setups (multiple LLMs competing)

## Environments

### FinText-Earnings-v0
- Input: earnings call transcripts (text)
- Action: predict stock direction (up/down/flat) + confidence
- Reward: accuracy vs actual price movement in next 1/5/20 trading days
- Data source: public earnings call transcripts + historical price data

### FinText-News-v0
- Input: financial news headlines and articles
- Action: predict market sentiment + affected sector
- Reward: accuracy vs actual market reaction
- Data source: public financial news APIs + historical data

### FinText-Filing-v0
- Input: SEC filing excerpts (10-K, 10-Q, 8-K)
- Action: extract key metrics, predict impact
- Reward: accuracy of extraction + directional prediction
- Data source: SEC EDGAR public filings

### FinText-Macro-v0
- Input: economic indicators, Fed speeches, macro reports
- Action: predict market regime (risk-on/risk-off/neutral)
- Reward: accuracy vs actual market behavior
- Data source: FRED, public economic data

## Reward Design

- Primary: directional accuracy (did the market move as predicted?)
- Secondary: confidence calibration (are high-confidence predictions more accurate?)
- Penalty: overconfidence on wrong predictions
- Bonus: reasoning quality (optional, via LLM-as-judge)

Reward function is configurable — users can define custom reward schemes.

## Tech Stack

- Python
- Gymnasium (OpenAI gym successor) for env API
- PufferLib for RL training infrastructure
- pandas / polars for data handling
- yfinance or similar for historical market data
- HuggingFace datasets for bundled data

## Project Structure

```
orin/
  README.md
  pyproject.toml
  orin/
    __init__.py
    envs/
      base.py            # Base financial text environment
      earnings.py        # FinText-Earnings-v0
      news.py            # FinText-News-v0
      filing.py          # FinText-Filing-v0
      macro.py           # FinText-Macro-v0
    rewards/
      directional.py     # Direction-based reward
      calibration.py     # Confidence calibration reward
      composite.py       # Combine multiple reward signals
    data/
      loaders.py         # Data loading and preprocessing
      sources.py         # Data source connectors
    wrappers/
      pufferlib.py       # PufferLib compatibility wrapper
    utils/
      market.py          # Market data utilities
      text.py            # Text preprocessing
  examples/
    train_basic.py       # Basic training example
    train_puffer.py      # PufferLib training example
    custom_reward.py     # Custom reward function example
  tests/
  data/                  # Sample/cached data
```

## Milestones

1. Repo setup, project structure, gym registration
2. Base environment class + reward system
3. FinText-Earnings-v0: first working environment
4. FinText-News-v0: second environment
5. PufferLib wrapper + training example
6. FinText-Filing-v0 and FinText-Macro-v0
7. Custom reward API + examples
8. Package on PyPI: `pip install orin`
9. Write documentation + training tutorial

## Research Questions

- Can an LLM learn to predict market direction from text via RL?
- Does RL fine-tuning improve financial reasoning vs supervised fine-tuning?
- What reward signal works best for financial text tasks?
- Does multi-environment training (all four envs) produce a better agent?

## Success Criteria

- pip-installable with working gym environments
- At least one trained agent that beats random baseline
- PufferLib compatible out of the box
- Clean API that others can extend with new environments
