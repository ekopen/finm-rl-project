## Project Timeline & Outline

### **Nov 25 – Dec 1 (Thanksgiving Week)**
- Set up repo, folders, and shared coding conventions.
- Agree on env API, PPO config format, and logging format.
- Implement single-asset trading environment + simple state.
- Implement PPO skeleton (actor–critic, GAE, clipped loss).
- Implement baselines (buy-and-hold, MA crossover).
- Add basic metrics and equity-curve plotting.
- Add reward shaping options (risk penalty, transaction costs).
- Expose PPO hyperparams (clip ε, entropy bonus, γ, λ).
- Run first small experiments: reward variants + a couple PPO settings.
- Save initial results and quick plots.

### **Dec 2 – Dec 6**
- Add richer state variants (more indicators, market context).
- Add regime labeling (bull/bear, low/high vol) and train/test splits.
- Run experiments:
  - Simple vs rich state,
  - Train on period A, test on period B.
- Generate plots/tables for these comparisons.
- *(Optional)* Implement pairs-trading env (spread, z-score state).
- *(Optional)* Run PPO on pairs env; compare to single-asset setup.
- Finalize key experiments:
  - Reward shaping,
  - PPO hyperparams,
  - State/env variants,
  - PPO vs baselines.
- Choose “best” runs to feature.

### **Dec 7 – Dec 8**
- Clean up code/notebooks; organize results by experiment.
- Create final figures (equity curves, ablation plots, summary tables).
- Draft and refine slide deck + main narrative.

### **Dec 9**
- Practice presentation.
- Tweak slides, clarify talking points.

### **Dec 10**
- Final presentation.
