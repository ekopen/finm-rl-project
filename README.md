## FINM RL Trading Sandbox – README

### 1. Project description & scope

This repo is a **research-style trading sandbox** that frames single-asset trading as a **reinforcement learning (RL)** problem and uses **PPO (Proximal Policy Optimization)** with an actor–critic architecture.

The goals are:

- **Model PPO-driven trading**:
  - Single asset (e.g. SPY) with discrete actions {short, flat, long}.
  - PPO agent learns a policy on historical data.
- **Compare against simple baselines**:
  - Buy-and-hold.
  - Moving-average crossover.
- **Study design choices**:
  - PPO hyperparameters: clip ε, entropy bonus, γ, GAE λ, etc.
  - State representation: simple vs (later) rich features, env variants.
  - (Later) reward shaping: transaction costs, risk penalties.
  - (Later) regime behavior: bull vs bear, high vs low volatility.

The end product is a set of **scripts and logs** you can use to generate plots and tables for a short presentation on:
- When PPO-style RL helps,
- How sensitive it is to hyperparameters and environment design,
- How its behavior compares to intuitive baselines.

---

### 2. Implementation checklist

> **Status snapshot:** Core PPO training, rich feature engineering, regime labeling/metrics, and reward-shaping knobs are all implemented. The remaining focus areas are (a) multi-asset env variants, (b) nicer regime visualizations, and (c) analysis notebooks for storytelling.

#### 2.1 Implemented

- **Data loading**
  - `features/data_loader.py`:
    - `load_price_data(ticker, start, end, interval)`:
      - Uses `yfinance` via `data/data_loader.fetch_single_asset`.
      - Ensures:
        - Lowercase columns: `open, high, low, close, volume`.
        - Flattened columns if yfinance returns MultiIndex (Price, Ticker).
        - Sorted `DatetimeIndex`.
    - `train_test_split_by_date(df, train_end_date)`:
      - Splits any OHLC/feature DataFrame into train/test by date.

- **Feature construction**
  - `features/state_builders.py`:
    - `make_simple_features(df, window=20)`:
      - Uses lowercase `close` column.
      - Adds:
        - `ret`: daily returns.
        - `ma`: rolling mean of `close` over `window`.
        - `vol`: rolling std of `ret` over `window`.
      - Returns augmented DataFrame.
    - `make_rich_features(df)`:
      - Builds on the simple features with multi-horizon returns, momentum, realized-volatility estimates, a 200-day trend factor, and a volume z-score (when volume is available).

- **Environment**
  - `envs/single_asset_env.py`:
    - `SingleAssetEnv(prices_df, features_df, initial_cash=1.0, config=None)`:
      - Uses `prices_df["close"]` and `features_df[["ret","ma","vol"]]`.
      - Discrete actions: 0=short, 1=flat, 2=long.
      - Reward starts as position × daily price return and optionally applies:
        - `transaction_cost`: penalty per unit position change,
        - `lambda_risk`: penalty proportional to squared daily return,
        - `lambda_drawdown`: penalty proportional to drawdown ratio.
      - Maintains `portfolio_value` and returns `info["portfolio_value"]`.
    - Methods:
      - `reset()` → initial state (features + position).
      - `step(action)` → `(next_state, reward, done, info)`.

- **PPO components**
  - `ppo/models.py`:
    - `ActorCritic(obs_dim, act_dim)`:
      - Wraps `Actor` and `Critic` in `ppo/actor.py` and `ppo/critic.py`.
      - `forward(obs)` → `(policy_out, values)`:
        - `policy_out`: action probabilities (for `Categorical`).
        - `values`: state values (1D tensor).
  - `ppo/ppo_agent.py`:
    - `PPOConfig` dataclass:
      - `gamma`, `gae_lambda`, `clip_eps`, `entropy_coef`, `value_coef`,
        `lr`, `max_grad_norm`, `update_epochs`, `minibatch_size`,
        `steps_per_epoch`, `epochs`, `log_interval`, `device`.
    - `RolloutBatch` dataclass: tensors for `obs, actions, log_probs, rewards, dones, values, last_value`.
    - `PPOAgent(state_dim, action_dim, config)`:
      - `act(state)` → `(action, log_prob, value)` for a single numpy state.
      - `value(state)` → scalar V(s).
      - `evaluate(states, actions)` → `(log_probs, entropy, values)`.
      - `compute_gae(...)` → advantages + returns (GAE(γ, λ)).
      - `update(rollout)`:
        - Normalizes advantages, runs multi-epoch PPO update over minibatches.
        - Uses clipped surrogate loss, value loss, entropy bonus, grad clipping.
        - Returns summary stats dict.

  - `ppo/trainer.py`:
    - `collect_rollout(env, agent, num_steps)`:
      - Collects `num_steps` transitions, resetting env on `done`.
      - Builds a `RolloutBatch` with bootstrapped `last_value`.
    - `train_ppo(env, agent, config, log_path=None)`:
      - For each epoch:
        - Calls `collect_rollout`.
        - Calls `agent.update(...)`.
        - Logs metrics: mean reward, policy/value loss, entropy, approx KL.
      - Optionally saves metrics dict as JSON if `log_path` provided.

  - `ppo/eval_utils.py`:
    - `run_policy_episode(env, agent, num_steps=None)`:
      - Runs a single test episode (or max `num_steps`) with PPOAgent.
      - Returns a numpy equity curve based on `info["portfolio_value"]`.

- **Baselines**
  - `baselines/buy_and_hold.py`:
    - `buy_and_hold(prices)` → final PnL.
    - `run_buy_and_hold(df, initial_cash=1.0)`:
      - Uses `df["close"]`/`"Close"`.
      - Computes daily returns and returns full equity curve.
  - `baselines/ma_crossover.py`:
    - `run_ma_crossover(df, fast=20, slow=50, initial_cash=1.0)`:
      - Builds a clean DataFrame with `close`, `fast`, `slow` MAs.
      - Drops warmup NaNs.
      - Position: 1 when `fast > slow`, 0 otherwise.
      - Returns `(equity, index)`.

- **Evaluation utilities**
  - `features/regimes.py`:
    - `label_bull_bear(df, ma_window=200)` → bull/bear strings based on price vs. long moving average.
    - `label_volatility(df, window=20, quantile=0.7)` → `high_vol`/`low_vol` via rolling realized volatility thresholds.

  - `eval/metrics.py`:
    - `compute_total_return(equity)`
    - `compute_annualized_return(equity, periods_per_year)`
    - `compute_volatility(equity, periods_per_year)`
    - `compute_sharpe(equity, rf=0.0, periods_per_year=252)`
    - `compute_max_drawdown(equity)`
    - `compute_hit_rate(positions, returns)`
  - `eval/plotting.py`:
    - `plot_equity_curves(curves_dict, dates=None, out_path=None)`
    - `plot_price_with_positions(price, positions, out_path=None)`
  - `eval/summarize.py`:
    - `summarize_runs(run_payloads, metric_key="test_metrics") -> DataFrame`.

- **Experiment helpers**
  - `experiments/common.py`:
    - `load_spy_splits()` → `(train_df, val_df, test_df)` for SPY.
    - `make_single_asset_env(prices_df, features_builder=..., env_config=None)` → helper to build `SingleAssetEnv` with optional rich features or shaping config.
    - `make_base_config(**overrides)` → common `PPOConfig`.
    - `make_results_dir(exp_name)` → `results/<exp_name>/`.
    - `train_env_with_config(env, config, log_path=None)` → trained `PPOAgent`.

- **Experiment scripts**
  - `experiments/exp_core_baselines.py`:
    - Trains PPO on train period, evaluates on test; compares to:
      - Buy-and-hold.
      - MA crossover.
    - Saves:
      - `results/core_baselines/core_metrics.json`
      - `results/core_baselines/core_equity.png`.
  - `experiments/exp_ppo_hyperparams.py`:
    - Sweeps several `PPOConfig` variants (clip/entropy/gamma/λ).
    - For each:
      - Trains PPO on train env.
      - Evaluates on test env (equity curve).
      - Computes test metrics.
      - Saves per-run JSON + equity plot.
    - Writes multi-run summaries (JSON + CSV) and prints a table.
  - `experiments/exp_reward_shaping.py`:
    - Runs multiple shaping configs (transaction costs, risk penalties, drawdown guards).
    - Passes the knobs into `SingleAssetEnv` so each config truly affects rewards.
    - Saves metrics + equity plot per config.
  - `experiments/exp_states_envs.py`:
    - Runs:
      - `single_simple`: SingleAssetEnv + simple features.
      - `single_rich`: SingleAssetEnv + rich features (multi-horizon stats).
    - Keeps `pairs_simple` as a placeholder for future multi-asset tests.
  - `experiments/exp_regimes.py`:
    - Trains PPO on train period, evaluates overall metrics on test.
    - Uses `features.regimes` labels to compute bull/bear and volatility-split metrics and saves them to JSON.

- **Top-level training entrypoint**
  - `train.py`:
    - Loads SPY, builds train env, sets explicit `PPOConfig`, prints it.
    - Trains PPO and writes `ppo_train_logs.json` + prints final metrics.

---

#### 2.2 Remaining gaps (next up)

- **Multi-asset envs**
  - `PairsEnv` and other multi-asset environments are not yet implemented.
  - `exp_states_envs.py` still skips the `pairs_simple` configuration until that env exists.

- **Regime visualization**
  - Regime metrics are saved to JSON, but richer plotting/reporting (e.g., regime-colored equity curves) is still TBD.

#### 2.3 Checklist of what still needs to be done

1. **Multi-asset envs & ablations**
  - Implement `PairsEnv` (or another simple multi-asset variant).
  - Extend `exp_states_envs.py` to run the `pairs_simple` configuration once the env exists.

2. **Regime visualization & reporting**
  - Build plots/tables that highlight behavior within each regime (e.g., colorized equity curves, summary tables for slides).

3. **Analysis notebooks**
  - Create notebooks under `notebooks/` to:
    - Load results from `results/`,
    - Plot training curves and equity curves,
    - Build summary tables for slides.

---

### 3. Modules and how they fit together

At a high level:

- **Data → Features → Env → PPO → Experiments → Results**

1. **Data & features**
   - `data/data_loader.py`: raw yfinance download.
   - `features/data_loader.py`: high-level wrapper:
     - Single ticker, clean schema, date handling.
   - `features/state_builders.py`: transforms price DataFrames into feature DataFrames.

2. **Environment**
   - `envs/single_asset_env.py`: converts prices + features → RL environment:
     - State: `[ret, ma, vol, position]`.
     - Action: discrete {short, flat, long}.
     - Reward: per-step PnL.

3. **PPO implementation**
   - `ppo/models.py`: actor–critic NN container.
   - `ppo/ppo_agent.py`: PPOConfig, RolloutBatch, PPOAgent implementing:
     - Action sampling,
     - GAE,
     - PPO update.
   - `ppo/trainer.py`: generic training loop that:
     - Collects rollouts,
     - Calls `update(...)`,
     - Logs metrics.
   - `ppo/eval_utils.py`: converts a trained agent + env into an equity curve.

4. **Baselines**
   - `baselines/buy_and_hold.py` and `baselines/ma_crossover.py`:
     - Provide equities for simple, interpretable strategies.

5. **Evaluation**
   - `eval/metrics.py`: canonical way to compute returns, Sharpe, drawdown, etc., from equity curves and positions.
   - `eval/plotting.py`: standard equity plots and price-with-positions visualization.
   - `eval/summarize.py`: convert multiple run payloads into a DataFrame summary.

6. **Experiments**
   - `experiments/common.py`: shared logic for all experiment scripts (data splits, config, env construction).
   - `exp_core_baselines.py`: core benchmark — PPO vs buy-and-hold vs MA crossover.
   - `exp_ppo_hyperparams.py`: PPO hyperparameter sweeps, with summary tables.
  - `exp_reward_shaping.py`: applies transaction-cost/risk/drawdown knobs and logs their impact.
   - `exp_states_envs.py`: structural harness for state/env ablations.
   - `exp_regimes.py`: structural harness for regime experiments.

7. **Training entrypoint**
   - `train.py`: a simple, single-run training script that is easy to tweak and debug.

---

### 4. How to run things

#### 4.1 Environment setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows

pip install -r requirements.txt
```

#### 4.2 Basic PPO training sanity check

```bash
cd .../finm-rl-project
python train.py
```

- Uses SPY daily data (2010–2020 train window).
- Logs PPO training metrics to `ppo_train_logs.json`.

#### 4.3 Core PPO vs baselines

```bash
python -m experiments.exp_core_baselines
```

- Output: `results/core_baselines/`:
  - `core_metrics.json` – metrics for PPO, BuyAndHold, MACrossover.
  - `core_equity.png` – equity curves for all three on the test period.

#### 4.4 Hyperparameter sweeps

```bash
python -m experiments.exp_ppo_hyperparams
```

- Output: `results/ppo_hyperparams/`:
  - Per-run:
    - `<name>_train_logs.json`
    - `<name>_equity.png`
    - `<name>.json` with config + metrics.
  - Summaries:
    - `summary_test_metrics.json`
    - `summary_test_metrics.csv` (plus a printed table).

Use a notebook or pandas to inspect:

```python
import pandas as pd
df = pd.read_csv("results/ppo_hyperparams/summary_test_metrics.csv")
df.sort_values("sharpe", ascending=False)
```

#### 4.5 State/env comparison

```bash
python -m experiments.exp_states_envs
```

- Runs `single_simple` (simple features) and `single_rich` (rich features).
- Output: `results/states_envs/` with `<name>_metrics.json` / `<name>_equity.png` per config.

#### 4.6 Reward shaping scaffold

```bash
python -m experiments.exp_reward_shaping
```

- Output: `results/reward_shaping/`:
  - `<name>_metrics.json`, `<name>_equity.png` for each shaping config.
- Each config now applies its transaction-cost / risk / drawdown knobs inside the env, so curves directly reflect the shaping choice.

#### 4.7 Regime scaffold

```bash
python -m experiments.exp_regimes
```

- Output: `results/regimes/`:
  - `overall_metrics.json`, `overall_equity.png`, `regime_metrics.json` (bull/bear + volatility splits).

---

### 5. How to use the results

- **For quick inspection**
  - Look at JSON files in `results/*` to see metrics per run.
  - Open PNG plots to get a qualitative feel for equity behavior.

- **For analysis/notebooks**
  - Create notebooks under `notebooks/` to:
    - `read_json`/`pd.read_csv` the logs from `results/`.
    - Use `eval.summarize.summarize_runs` where available.
    - Produce:
      - Tables of metrics across hyperparams,
      - Plots of equity vs baselines,
      - Regime-split summaries (now saved via `regime_metrics.json`).

- **For slides**
  - Choose:
    - 1–2 plots from `core_baselines` (PPO vs baselines),
    - 1–2 plots/tables from `ppo_hyperparams` showing sensitivity to clip/entropy/γ/λ,
    - 1 state/env table and plot once rich/pairs are implemented,
    - Reward-shaping comparison tables/plots (`results/reward_shaping`) and, once available, regime-visualization assets.

---

### 6. Suggested next steps

1. Build a simple **PairsEnv** (or similar multi-asset env) and enable the `pairs_simple` run inside `exp_states_envs.py`.
2. Polish the **regime reporting** (e.g., add regime-colored plots / nicer tables for presentations).
3. Build **analysis notebooks** that:
   - Read `results/**`,
   - Plot key comparisons,
   - Export figure/table assets for your talk.

