# Pilot

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

## Constructing the Nested Sub-datasets (`D₀ ⊆ D₁ ⊆ … ⊆ D`)

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

## References

- Yu, S., Pritchard, M., Ma, P.-L., Singh, B., & Silva, S. (2024). *Two-step hyperparameter optimization method: Accelerating hyperparameter search by using a fraction of a training dataset.* Artificial Intelligence for the Earth Systems, 3(1). https://doi.org/10.1175/AIES-D-23-0013.1 (arXiv:2302.03845)
- Shleifer, S., & Prokop, E. (2019). *Using small proxy datasets to accelerate hyperparameter search.* arXiv preprint arXiv:1906.04887.
