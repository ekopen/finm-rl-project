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

### 2. Progress snapshot

#### 2.1 Implemented pillars

- **Data & features** – `data/data_loader.py`, `features/data_loader.py`, and `features/state_builders.py` cover SPY downloads plus simple/rich feature sets.
- **Environment** – `envs/single_asset_env.py` exposes a discrete long/flat/short `SingleAssetEnv` with optional transaction-cost, risk, and drawdown shaping knobs.
- **PPO stack** – `ppo/models.py`, `ppo/ppo_agent.py`, `ppo/trainer.py`, and `ppo/eval_utils.py` implement an actor–critic PPO agent, GAE, rollouts, training loop, and evaluation helpers.
- **Baselines & evaluation** – `baselines/*`, `eval/metrics.py`, `eval/plotting.py`, and `eval/summarize.py` provide reference strategies plus consistent metric/plot utilities.
- **Experiments & entrypoint** – `experiments/*.py` plus `train.py` cover core baselines, hyperparameter sweeps, reward shaping, regime analysis, and a quick-start training script.
- **Regime visualization & reporting** – `exp_regimes.py` generates regime-colored equity plots and formatted CSV tables comparing performance across bull/bear and volatility regimes. Outputs include `bull_bear_regime_metrics.csv`, `volatility_regime_metrics.csv`, and `regime_comparison.csv` with formatted console output.
- **PPO pretraining** – Optional BC initialization (w/ vs w/o pretrain) for PPO to test improvements in sample efficiency and convergence.
- **Robustness testing** – Seed sweeps and config variations to measure performance stability via mean/std of key metrics across runs.
- **Rich feature builder added** – `make_rich_features` for extended state representations.
- **PairsEnv prototype added** – simple 2-asset environment (SPY/QQQ) with spread features and long-A/short-B vs short-A/long-B actions.
- **State/env ablation extended** – `exp_states_envs.py` now runs `single_simple`, `single_rich`, and `pairs_simple`.
- **Reward shaping fully wired** – transaction cost, risk penalty, and drawdown penalty integrated into SingleAssetEnv and used in `exp_reward_shaping.py`.
- **Analysis notebooks** – `notebooks/analysis_main.ipynb` provides comprehensive analysis of all experiments (baselines, hyperparameters, states/envs, robustness) with export functionality. `notebooks/regime_analysis.ipynb` focuses on regime-specific analysis with visualizations and heatmaps.


#### 2.2 Next up items

_No immediate next steps identified. The core functionality is complete._

---

### 3. Modules and how they fit together

At a high level:

- **Data → Features → Env → PPO → Experiments → Results**

1. **Data & features**
   - `data/data_loader.py`: raw yfinance download.
   - `features/data_loader.py`: high-level wrapper:
     - Single ticker, clean schema, date handling.
   - `features/state_builders.py`: transforms price DataFrames into feature DataFrames.
   - `make_simple_pairs_features` – aligns two tickers and constructs simple per-asset features for pairs trading.


2. **Environment**
   - `envs/single_asset_env.py`: converts prices + features → RL environment:
     - State: `[ret, ma, vol, position]`.
     - Action: discrete {short, flat, long}.
     - Reward: per-step PnL.
   - `envs/pairs_env.py` – simple 2-asset environment:
     - State includes features for both assets + spread + positions.
     - Actions = {flat, long A / short B, short A / long B}.
     - Rewards combine both legs with transaction-cost / risk / drawdown shaping.
      

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
- The `pairs_simple` configuration (SPY/QQQ) is now enabled and produces
`pairs_simple_metrics.json` and `pairs_simple_equity.png`.


#### 4.6 Reward shaping scaffold

```bash
python -m experiments.exp_reward_shaping
```

- Output: `results/reward_shaping/`:
  - `<name>_metrics.json`, `<name>_equity.png` for each shaping config.
- Each config now applies its transaction-cost / risk / drawdown knobs inside the env, so curves directly reflect the shaping choice.
  The environment now *actually applies* shaping:
  - `transaction_cost` penalizes position changes,
  - `lambda_risk` penalizes squared returns,
  - `lambda_drawdown` penalizes drawdowns relative to running peaks.


#### 4.7 Regime analysis

```bash
python -m experiments.exp_regimes
```

- Output: `results/regimes/`:
  - `overall_metrics.json`, `overall_equity.png` – overall performance metrics and equity curve.
  - `regime_metrics.json` – detailed metrics split by bull/bear and volatility regimes.
  - `bull_bear_regime_metrics.csv`, `volatility_regime_metrics.csv` – formatted CSV tables for each regime type.
  - `regime_comparison.csv` – combined comparison table across all regimes.
  - `bull_bear_equity.png`, `vol_regime_equity.png` – regime-colored equity curves.
  - Formatted tables are printed to console for quick inspection.

---

### 5. How to use the results

- **For quick inspection**
  - Look at JSON files in `results/*` to see metrics per run.
  - Open PNG plots to get a qualitative feel for equity behavior.

- **For analysis/notebooks**
  - Use existing notebooks in `notebooks/`:
    - `analysis_main.ipynb` – comprehensive analysis of all experiments:
      - Core baselines comparison (PPO vs BuyAndHold vs MACrossover)
      - Hyperparameter sensitivity analysis with scatter plots
      - State/Environment ablation comparisons
      - Robustness analysis using `eval.summarize.summarize_runs`
      - Master comparison dashboard aggregating all experiments
      - Exports tables to CSV, LaTeX, and HTML formats
    - `regime_analysis.ipynb` – regime-specific analysis:
      - Loads regime comparison CSV and JSON data
      - Creates bar charts and heatmaps for regime performance
      - Exports regime comparison tables in multiple formats
  - All exported tables and figures are saved to `notebooks/exports/` directory.

- **For slides**
  - Use exported assets from `notebooks/exports/` or choose from:
    - 1–2 plots from `core_baselines` (PPO vs baselines),
    - 1–2 plots/tables from `ppo_hyperparams` showing sensitivity to clip/entropy/γ/λ,
    - State/env comparison tables and plots (`results/states_envs`),
    - Reward-shaping comparison tables/plots (`results/reward_shaping`),
    - Regime visualization assets: regime-colored equity curves (`results/regimes/*_equity.png`) and regime comparison tables (CSV/LaTeX/HTML from notebooks).

---

### 6. Suggested next steps

_All major planned features are complete. Potential future enhancements could include:_
- Additional regime analysis visualizations (e.g., time-series breakdowns)
- Extended hyperparameter search spaces
- Additional baseline strategies for comparison
- Real-time trading simulation capabilities

