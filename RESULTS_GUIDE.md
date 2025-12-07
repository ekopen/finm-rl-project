# Complete Results Procedure for Final Project

## Overview: Setup

**Before running experiments:**
1. Activate environment:
   ```bash
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
2. Ensure data is cached (first run will download SPY data automatically)

---

## Section 1: Core Baselines Comparison
**Purpose:** Show PPO vs simple strategies

### Run:
```bash
python -m experiments.exp_core_baselines
```

### Results Location:
`results/core_baselines/`

**Key Files:**
- `core_metrics.json` – Metrics for PPO, BuyAndHold, MACrossover
- `core_equity.png` – Equity curves overlay

### In Analysis Notebook:
Open `notebooks/analysis_main.ipynb` → **Section 1: Core Baselines Comparison**

**What it does:**
- Loads `core_metrics.json` and creates comparison DataFrame
- Generates 2x2 bar charts comparing all metrics (total_return, annualized_return, sharpe, max_drawdown)
- Exports table to `notebooks/exports/baselines_comparison.csv/.tex/.html`
- Creates `baselines_comparison.png` for slides

**For Presentation:**
- Use `core_equity.png` from results (equity curves overlay)
- Use `notebooks/exports/baselines_comparison.png` (bar chart comparison)
- Use exported comparison table to highlight PPO advantages
- Key metric: Sharpe ratio comparison

---

## Section 2: Hyperparameter Sensitivity Analysis
**Purpose:** Show sensitivity to PPO hyperparameters

### Run:
```bash
python -m experiments.exp_ppo_hyperparams
```

### Results Location:
`results/ppo_hyperparams/`

**Key Files:**
- `summary_test_metrics.csv` – Quick comparison table
- `summary_test_metrics.json` – Same data in JSON
- Individual run files: `clip0.1_ent0.00.json`, `clip0.2_ent0.01.json`, etc.

### In Analysis Notebook:
Open `notebooks/analysis_main.ipynb` → **Section 2: Hyperparameter Sensitivity Analysis**

**What it does:**
- Loads summary CSV and individual run JSONs
- Extracts config details (clip_epsilon, entropy_coef, gamma, gae_lambda)
- Creates 2x2 sensitivity plots:
  - Scatter: Sharpe vs Clip Epsilon
  - Scatter: Sharpe vs Entropy Coefficient
  - Bar chart: Total Return by configuration
  - Bar chart: Sharpe by configuration
- Highlights best/worst configurations
- Exports tables to `notebooks/exports/hyperparams_summary.csv/.tex` and `hyperparams_detailed.csv`

**For Presentation:**
- Use `notebooks/exports/hyperparams_sensitivity.png` (4-panel sensitivity plots)
- Use exported summary table showing best configuration
- Key insight: Which hyperparameters matter most?

---

## Section 3: State/Environment Ablation Analysis
**Purpose:** Compare different state representations and environments

### Run:
```bash
python -m experiments.exp_states_envs
```

### Results Location:
`results/states_envs/`

**Key Files:**
- `single_simple_metrics.json` + `single_simple_equity.png` + `single_simple_equity.json`
- `single_rich_metrics.json` + `single_rich_equity.png` + `single_rich_equity.json`
- `pairs_simple_metrics.json` + `pairs_simple_equity.png` + `pairs_simple_equity.json`

### In Analysis Notebook:
Open `notebooks/analysis_main.ipynb` → **Section 3: State/Environment Ablation Analysis**

**What it does:**
- Loads all `*_metrics.json` files and creates comparison DataFrame
- Generates 2x2 bar charts comparing all metrics
- **Plots all 3 equity curves together** (loads from `*_equity.json` files)
- Exports comparison table to `notebooks/exports/states_envs_comparison.csv/.tex`
- Creates `states_envs_comparison.png` (bar charts)
- Creates `states_envs_equity_curves.png` (combined equity curves)

**For Presentation:**
- Use `notebooks/exports/states_envs_equity_curves.png` (all 3 curves on one plot)
- Use `notebooks/exports/states_envs_comparison.png` (bar chart comparison)
- Use exported comparison table
- Key insight: Does richer state representation help?

---

## Section 4: Regime Analysis
**Purpose:** Show performance across market regimes

### Run:
```bash
python -m experiments.exp_regimes
```

### Results Location:
`results/regimes/`

**Key Files:**
- `regime_comparison.csv` – Combined table (all regimes)
- `bull_bear_regime_metrics.csv` – Bull vs Bear breakdown
- `volatility_regime_metrics.csv` – High vs Low vol breakdown
- `bull_bear_equity.png` – Equity curve with bull/bear shading
- `vol_regime_equity.png` – Equity curve with volatility shading
- `overall_equity.png` – Overall performance
- `regime_metrics.json` – Detailed JSON data

**Console Output:** Formatted tables printed during run

### In Analysis Notebook:
Open `notebooks/analysis_main.ipynb` → **Section 4: Regime Analysis**

**What it does:**
- Loads `regime_comparison.csv` and `regime_metrics.json`
- Creates 2x2 bar charts comparing metrics across regimes
- Creates heatmap visualization of regime performance
- Exports tables (CSV/LaTeX/HTML) to `notebooks/exports/`
- Provides summary statistics highlighting best/worst regimes
- Creates `regime_performance.png` (bar charts)
- Creates `regime_heatmap.png` (heatmap)

**For Presentation:**
- Use `bull_bear_equity.png` and `vol_regime_equity.png` (regime-colored equity curves)
- Use `notebooks/exports/regime_heatmap.png` (heatmap)
- Use `notebooks/exports/regime_performance.png` (bar charts)
- Use exported comparison tables
- Key insight: Does PPO perform differently in bull vs bear markets?

---

## Section 5: Reward Shaping Analysis
**Purpose:** Show impact of transaction costs and risk penalties

### Run:
```bash
python -m experiments.exp_reward_shaping
```

### Results Location:
`results/reward_shaping/`

**Key Files:**
- `no_shaping_metrics.json` + `no_shaping_equity.png` (baseline)
- `high_cost_metrics.json` + `high_cost_equity.png` (transaction cost)
- `risk_penalty_metrics.json` + `risk_penalty_equity.png` (risk penalty)
- `drawdown_guard_metrics.json` + `drawdown_guard_equity.png` (drawdown penalty)

### In Analysis Notebook:
Open `notebooks/analysis_main.ipynb` → **Section 5: Reward Shaping Analysis**

**What it does:**
- Loads all `*_metrics.json` files and extracts shaping config details
- Creates 2x2 bar charts comparing all metrics across shaping configs
- Creates horizontal bar chart showing Sharpe ratio comparison
- Exports comparison table to `notebooks/exports/reward_shaping_comparison.csv/.tex/.html`
- Provides impact summary comparing each config against baseline (no_shaping)
- Creates `reward_shaping_comparison.png` (bar charts)
- Creates `reward_shaping_sharpe_comparison.png` (Sharpe comparison)

**For Presentation:**
- Use `notebooks/exports/reward_shaping_comparison.png` (bar chart comparison)
- Use `notebooks/exports/reward_shaping_sharpe_comparison.png` (Sharpe comparison)
- Use exported comparison table
- Use individual equity plots from `results/reward_shaping/` if needed
- Key insight: How do different reward shaping strategies affect performance?

---

## Section 6: Robustness Analysis
**Purpose:** Show stability across seeds and pretraining

### Run:
```bash
python -m experiments.exp_robustness
```
**Note:** This may take longer (multiple seeds × pretrained/non-pretrained)

### Results Location:
`results/ppo_seed_pretrain_compare/`

**Key Files:**
- `summary_test_metrics.csv` – Per-run results
- `summary_group_stats.csv` – Mean/std by group (pretrained vs non-pretrained)
- Individual seed files: `seed_0_pre.json`, `seed_0_nopre.json`, etc.

### In Analysis Notebook:
Open `notebooks/analysis_main.ipynb` → **Section 6: Robustness Analysis**

**What it does:**
- Uses `eval.summarize.summarize_runs()` to process run payloads
- Loads group statistics CSV
- Creates 2x2 bar charts with error bars (mean ± std) for each metric
- Exports tables to `notebooks/exports/robustness_group_stats.csv/.tex`
- Creates `robustness_comparison.png` (error bar plots)

**For Presentation:**
- Use `notebooks/exports/robustness_comparison.png` (error bar plots)
- Use exported group statistics table
- Key insight: How stable is performance? Does pretraining help?

---

## Quick Reference: Run Order

```bash
# 1. Baselines (fastest, ~5-10 min)
python -m experiments.exp_core_baselines

# 2. Hyperparameters (~20-30 min)
python -m experiments.exp_ppo_hyperparams

# 3. State/Envs (~15-20 min)
python -m experiments.exp_states_envs

# 4. Regimes (~10-15 min)
python -m experiments.exp_regimes

# 5. Reward Shaping (~20-30 min)
python -m experiments.exp_reward_shaping

# 6. Robustness (~1-2 hours)
python -m experiments.exp_robustness

# Then run analysis notebook to generate all summaries and exports
```

**Total time (core experiments):** ~1.5-2 hours  
**Total time (with all experiments):** ~2.5-3.5 hours

---

## Recommended Presentation Flow

1. **Introduction** – Problem statement and approach overview

2. **Baselines Comparison (Section 1)**
   - Show PPO vs BuyAndHold vs MACrossover
   - Use: `core_equity.png` + exported comparison table/plot

3. **Hyperparameter Sensitivity (Section 2)**
   - Show which hyperparameters matter
   - Use: Sensitivity scatter plots + best configuration highlight

4. **State/Environment Design (Section 3)**
   - Show impact of state representation
   - Use: Combined equity curves plot + comparison table

5. **Regime Analysis (Section 4)**
   - Show performance across market conditions
   - Use: Regime-colored equity curves + heatmap + comparison tables

6. **Reward Shaping (Section 5)**
   - Show impact of transaction costs and risk penalties
   - Use: Comparison plots + impact summary

7. **Robustness (Section 6)**
   - Show stability and pretraining impact
   - Use: Error bar plots + group statistics

8. **Conclusions**
   - Summarize key findings from each section
   - Highlight best overall configuration
   - Key takeaways

---

## Tips

1. Run experiments in order; later ones may depend on earlier results
2. Check console output for quick metrics during runs
3. Use the analysis notebook to generate presentation-ready assets
4. All exported files go to `notebooks/exports/` for easy access
5. JSON files contain detailed data; CSV files are easier to read
6. PNG plots are ready for slides (150 DPI)
7. Regime analysis now includes CSV tables for easy comparison
8. State/env analysis includes combined equity curves plot

---

## To Do Next

#### 1. Improving the Visuals
#### 2. Improving the Models
#### 3. Including Background and Interpretation of Results
#### 4. Cleaning Up the Slideshow

