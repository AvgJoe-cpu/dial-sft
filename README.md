# dial-sft

## Contents

```
dial-sft/
├── src/                  # Source code
│   ├── ar/               # Autoregressive baseline
│   ├── mdlm/             # MDLM model, trainers, samplers
│   └── dataset_load.py   # Dataset loading utilities
├── datasets/             # Raw and processed datasets
└── weights/              # Model checkpoints and weights
```


## Experiments 

### Pilot

**Goal:** Quickly find reasonable values for the **alpha scheduler** (which defines the masking-rate function `α(t)`) and the learning rate that produce stable, non-collapsed training on the masked objective of the base Diffusion LLM. This pilot asks: *What are generally the best parameters for our downstream dataset?*

Hyperparameter search uses `N` progressive stages with nested sub-datasets of increasing size (`D₀ ⊂ D₁ ⊂ … ⊂ D_{N-1}`), where sample sizes increase in fixed steps (e.g., `10k → 100k → full`).

**Tunable parameters:**
- **Alpha scheduler** (`α(t)` family): `linear` vs. `cosine` — controls how the per-step masking probability `1 − α(t)` evolves with diffusion time `t`.
- **Loss weighting:** `uniform` (constant) vs. `scheduler` (time-dependent weight `w(t) = −α′(t) / (1 − α(t))`).
- **`time_epsilon`:** lower bound on sampled `t ∈ (time_epsilon, 1]`; bounds the effective masking range per batch.
- **Learning rate (LR):** standard reference values for MDLM (e.g., `5e-5` → `5e-4`).
- **`max_length`:** `{512, 768, 1024}`.

**Stage `k` (`D_k`):** Coarse-to-fine grid / random search. Early stages use broad ranges (e.g., 5–7 values per parameter); later stages narrow around the top 2–3 configurations from the prior stage. Ranking uses masked-LM loss / perplexity on the withheld test set.

**Training protocol per trial:**
- Restart from base MDLM weights for every trial.
- Fixed initial `batch_size`.
- Train on `D_k` under the masked diffusion objective.
- Evaluate on the withheld test set after fixed steps / epochs.

### Implementation: Constructing the Nested Sub-datasets (`D₀ ⊆ D₁ ⊆ … ⊆ D`)

The key property is guaranteed by one-time shuffling with a fixed random seed and then taking prefixes of the shuffled list. This ensures every smaller subset is literally contained inside every larger one (no distribution shift between stages), while the full dataset `D` is only shuffled once for perfect reproducibility.

We follow established practices: Yu et al. (2024) and Shleifer & Prokop (2019) demonstrate that hyperparameter configurations identified on small representative subsets transfer reliably to larger datasets, enabling substantial compute savings while preserving model quality.

**Algorithm — Nested Sub-dataset Construction**

- **Require:** Full human-authored dataset `D`; list of target sizes `sizes = [s₀, s₁, …, s_{N-1}]` (sorted ascending, `s₀ < s₁ < … < s_{N-1}`).
- **Ensure:** Nested subsets `D₀ ⊂ D₁ ⊂ … ⊂ D_{N-1}`.

1. Set random seed to `42`.
2. `shuffled_D ← shuffle(copy of D)`
3. `subsets ← ∅`
4. **for** `i ← 0` **to** `|sizes| − 1` **do**
5. &nbsp;&nbsp;&nbsp;&nbsp;`target_size ← sizes[i]`
6. &nbsp;&nbsp;&nbsp;&nbsp;`subsets["D" + str(i)] ← shuffled_D[0 : target_size]`
7. **end for**
8. **return** `subsets`

This algorithm performs a single reproducible shuffle of the full dataset and constructs strictly nested prefixes, ensuring `D₀ ⊆ D₁ ⊆ … ⊆ D_{N-1}`. The resulting sub-datasets are used for the progressive multi-stage hyperparameter search.


**References**

- Yu, S., Pritchard, M., Ma, P.-L., Singh, B., & Silva, S. (2024). *Two-step hyperparameter optimization method: Accelerating hyperparameter search by using a fraction of a training dataset.* Artificial Intelligence for the Earth Systems, 3(1). https://doi.org/10.1175/AIES-D-23-0013.1 (arXiv:2302.03845)
- Shleifer, S., & Prokop, E. (2019). *Using small proxy datasets to accelerate hyperparameter search.* arXiv preprint arXiv:1906.04887.

------------------------------------------------------------------------------------------------------------------------------------------------------


### Experiment 1 — Subset Sampling Study (AUG)

#### Objective

This experiment investigates the impact of training on subsets with controlled **length** and **distribution** properties on model performance in recursive synthetic training. It specifically examines whether training on short examples affects generalization when evaluated on unaltered (potentially longer) test data — and the reverse.

#### Variants

- **(V1) Stratified Subset:** Sampled according to the original dataset's distribution (length distribution emerges implicitly via tokenization).
- **(V2) Short Subset:** Threshold-based sampling by total token count per example (sum of prompt + completion tokens).
- **(V Control) Random Subset:** Unbiased seed-based random sampling (baseline).

For each variant, both `train` and `test` subsets are derived consistently. An additional **unchanged original test set** is retained for cross-evaluation.

#### Setup

1. **Dataset selection:** start with the full human-authored dataset `D`.
2. **Sampling:**
   - Stratified → `(ds_train_strat, ds_test_strat)`
   - Length-threshold → `(ds_train_len, ds_test_len)`
   - Random → `(ds_train_rand, ds_test_rand)`
3. **Train–Generate loop:** for each train/test pair, perform iterative fine-tuning and synthetic data generation for up to `4` turns (empirically determined).

#### Subset combinations and runs

| Category | Run | Train subset | Test / evaluation subsets |
|---|---|---|---|
| Baseline runs | (A) | `ds_train_strat` | `ds_test_strat` (matched distribution) |
| Baseline runs | (B) | `ds_train_rand` | `ds_test_rand` (random control) |
| Generalization runs | (C) | `ds_train_len` | `ds_test_len` (matched short), `ds_test_rand` (random), `ds_test_strat` (stratified / full distribution) |
| Generalization runs | (D) | `ds_train_strat` | `ds_test_strat` (matched), `ds_test_len` (short), `ds_test_rand` (random) |

#### Research questions

- How does training exclusively on short data affect performance when the test set contains longer examples (unaltered or truncated)?
- Does training on a stratified (distribution-matched) subset improve generalization to shorter examples?
- What is the interaction between length-distribution mismatch and downstream performance in the recursive synthetic setting?

#### Compute scale (approximate)

`4 runs × 4 turns × 8 epochs = 128 training passes` → ~`1.28M` items processed at `batch_size = 64` (≈ `20k` global steps, under the no-filter assumption).

This design isolates the effect of length distribution while remaining compatible with the main recursive synthetic training chain (Algorithm 1).

**Require:** Human-authored seed dataset \(D_0\), base Diffusion LLM \(M_{\text{base}}\), number of iterations \(n\), optional filter function \(f_{\text{filter}}\)  
**Ensure:** Sequence of models \(M_1, M_2, \dots, M_n\) trained under increasing synthetic data regimes

#### Procedure

```text
for i ← 1 to n do
    Mi ← fresh instance of M_base

    if i = 1 then
        D_train ← D_0                     ▷ human-authored data
    else
        D_train ← GenerateSyntheticData(M_{i-1}, D_0)
                                          ▷ synthetic data from previous model
        if filtering is enabled then
            D_train ← f_filter(D_train)   ▷ apply quality/diversity filter
        end if
    end if

    Mi ← FineTune(Mi, D_train)
    Evaluate(Mi)                          ▷ diversity, learnability, and human test-set metrics
end for
```

#### Detailed Procedure

- **\(M_1\)** is fine-tuned solely on the original human-authored dataset \(D_0\).
- For **\(i \geq 2\)**, each model \(M_i\) is fine-tuned on synthetic data \(D_{i-1}\) generated by its predecessor \(M_{i-1}\).
- After generation, an optional filter \(f_{\text{filter}}\) may be applied to \(D_{\text{train}}\), for example based on:
  - quality
  - length
  - diversity
- Synthetic data generation, with or without filtering, preserves the target dataset size.
- A fresh model instance is used at every iteration to isolate the effect of the training data distribution.