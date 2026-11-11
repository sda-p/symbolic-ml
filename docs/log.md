# Experiment log

Each entry states the hypothesis and prediction before the run, then the result. Newest stage last.

## Stage 0: harness and baselines

### Predictions (written 2026-11-11, before any run)

- **E0.1** CTW (depth 4, α=1, ρ=0.5) on S1, 200k tokens, 5 seeds. Cumulative excess loss over the true source stays below the §7 bound on every seed. This is a theorem, so a violation means a bug.
- **E0.2** CTW and n-gram sweeps on TinyShakespeare, prequential over the whole file. I expect the best CTW to land between 2.2 and 2.6 bits per character, at depth 5–6 with α well below 1, and to beat the best n-gram by a small margin.
- **E0.3** CTW and n-gram on S2–S5, 200k tokens, depths 2–16. I expect S3 (brackets), S4 (counter) and S5 (XOR) to keep a tail excess above 0.05 bits per token at every depth. S2 (agreement) is the one I am unsure about: sentences are short, so a depth-16 suffix model may get close.

### Results (2026-11-11)

**E0.1: bound holds.** `experiments/e0_1_ctw_bound.py`. On all 5 seeds the cumulative excess is about half the bound (for example 538 against 1100 bits with 15 leaves, 3997 against 8841 bits with 372 leaves). Per token, CTW is within 0.003–0.02 bits of the source.

**E0.2: prediction half wrong.** `experiments/e0_2_text.py`. Prequential bits per character on TinyShakespeare (1.1M characters, V=65):

| Model | Best setting | Whole stream | Final 10% |
|---|---|---|---|
| CTW | depth 8, α=0.01 | 2.253 | 2.164 |
| Interpolated n-gram | depth 8, λ=32 | 2.178 | 2.094 |

- CTW landed in the predicted range, but the n-gram beat it by about 0.08 bits, where I had predicted CTW ahead.
- Both best settings sit at the edge of the sweep (smallest α, largest λ, both extended once already), so neither number is a tuned optimum. Treat them as reference points, not ceilings.
- CTW gains almost nothing past depth 5 (2.256 at depth 5, 2.253 at depth 8).
- Likely reason for the n-gram's lead, not yet tested: its estimate at each context uses the shorter context's prediction as the base measure, while CTW's Dirichlet leaves smooth toward uniform. With α=1 CTW is 0.44 bits worse than with α=0.01, which fits that reading.
- Consequence for Stage 1: leaf estimators in the predicate tree should smooth toward the parent's prediction, not toward uniform. This is the data-fragmentation concern from the plan showing up already in the baseline.

**E0.3: sources are hard as intended.** `experiments/e0_3_hardness.py`. Tail excess over source entropy (bits per token, mean of 2 seeds, 200k tokens), best CTW setting at any depth up to 16:

| Source | Best CTW excess | Depth where it stops improving |
|---|---|---|
| S2 agreement | 0.055 | 4 |
| S3 brackets | 0.184 | 12 |
| S4 counter | 0.170 | 8 |
| S5 XOR | 0.096 | 4 |

- All four stay above the 0.05 threshold, so none needs redesign. S2 is marginal (0.055), as suspected; if Stage 1 results on it are hard to read, raise the distractor probability.
- Going deeper does not help CTW: at depth 16 it holds 1.0–1.5M nodes with the same excess as at depth 8.
- The fixed-λ n-gram degrades badly at depth ≥ 8 on these sources (excess above 0.4), so CTW is the baseline to use on synthetic data.

## Stage 1: one-level learner (predicate tree)

Implementation notes that differ from the plan:

- Leaves predict by backing off along the tree path (the interpolated estimator that won E0.2), not by CTW-style mixing.
- A candidate is a (leaf, finite-valued feature) pair with a value-by-token count table. The binary split is chosen at commit time by a greedy search over value sets, so token classes are learned as a by-product and added to the DSL's class library.
- Predictions for a microbatch all use the pre-batch model, so the learner is slightly handicapped against the per-token baselines.

### Predictions (written 2026-11-11, before any run)

- **E1.1** S2 (agreement) and S4 (counter): the planted predicate is found and tail excess drops below 0.05 bits within 200k tokens, beating CTW. S3 (brackets): not solved, because DSL v0 cannot express "nearest unmatched opener"; excess no better than CTW's 0.18.
- **E1.2** S5 (XOR): fails. The greedy class search will merge `b0` and `b1` into one leaf (they predict the same answer distribution), after which neither A nor B alone has any gain. Excess stays near CTW's 0.10.
- **E1.3** TinyShakespeare: worse than the n-gram reference (2.178), somewhere in 2.3–2.6 bits per character. Non-suffix features add less than 0.02 bits over the suffix-only ablation.

### Results, first pass (2026-11-11)

Default config: batch 256, λ=8, pool of 256 candidates, budget 400k ops per update, review every 4 updates, commit threshold 8 bits.

**E1.1 synthetic recovery.** `experiments/e1_1_recovery.py`, 200k tokens, 5 seeds. Tail excess over source entropy, bits per token:

| Source | Predicate tree (range over seeds) | Best CTW (E0.3) | Prediction |
|---|---|---|---|
| S2 agreement | 0.005–0.030 | 0.055 | held |
| S3 brackets | 0.238–0.264 | 0.184 | held (not solved; in fact worse than CTW) |
| S4 counter | 0.002–0.009 on 2 seeds, 0.106–0.120 on 3 seeds | 0.170 | partly held |
| S5 XOR | 0.020–0.053 | 0.096 | wrong |

- S2: beats CTW on every seed with 6–26 leaves against CTW's 10⁵ nodes. The trees are not the clean planted rule; they reach the subject through several overlapping "token next to the last delimiter-like class" features, and excess is still falling at 200k tokens.
- S4: when `count('M' since last('.')) mod 3` is tried, it is committed immediately with a gain near 190 bits over 300 examples, and excess drops from 0.12 to under 0.01. On 3 of 5 seeds the enumerator had not reached it at the relevant leaf by 200k tokens. The failure is in search order, not in credit.
- S5: the prediction of failure was wrong. Composite features such as `tok[last('b1')-1]` encode B and (when the gap is zero) A together, so they have non-zero gain on their own and separate the leaves enough for the A feature to show gain afterwards. In this DSL the XOR is not a pure interaction. S5 therefore does not test coordinated edits; Stage 3 needs a probe where no single feature in the vocabulary carries the conjunction.
- S3: the learner spends 42–61 leaves on distance and neighbour features that approximate the stack poorly. This is the expected vocabulary limit (§9).

**E1.3 natural text: prediction wrong, in the bad direction.** `experiments/e1_3_text.py`, TinyShakespeare, prequential bits per character:

| λ | Features | Whole stream | Final 10% | Leaves |
|---|---|---|---|---|
| 8 | full DSL | 2.932 | 2.795 | 713 |
| 8 | suffix only | 2.891 | 2.745 | 894 |
| 32 | full DSL | 2.921 | 2.836 | 593 |
| 32 | suffix only | 2.864 | 2.717 | 862 |

- All four are about 0.7 bits worse than the n-gram reference (2.178). I predicted 2.3–2.6.
- The tree is starved of splits. The learner commits at most one split per review, so 4,300 updates allow at most 1,075 splits; the runs made 593–894. The n-gram reference holds over 200k contexts. This is a throttle in my implementation, not evidence about the approach.
- The full DSL is 0.04–0.06 bits worse than suffix-only at the same budget. Non-suffix features were committed often (about 40–50% of splits), but each one costs more to evaluate, so fewer candidates fit in the budget. Whether they help at equal split count is not yet known.
- Learned classes look linguistically sensible from the first split (vowels plus whitespace, then consonant groups).

**Defect: the op budget is not enforced.** On text, 142–592 of 4,300 updates exceeded the 400k budget. The refill estimate ignores the cost of the review step (the greedy class search). On the synthetic sources there were no violations. This must be fixed before E1.4, since the budget sweep is meaningless otherwise.

Next: enforce the budget including review cost, allow several commits per review, rerun E1.3, then E1.4 (budget sweep) and E1.5 (commit rules).

### Results, second pass (2026-11-11)

Changes to the learner between passes, in order:

1. Hard op budget: every unit of candidate work checks a deadline before starting; review rotates through leaves and resumes where the budget ran out. Several commits per update are allowed.
2. Class search changed from greedy row-by-row growth to "best single value, then two rounds of sort-and-take-best-prefix", which costs O(values × V).
3. Pool cap raised from 256 to 8192 candidates, so the op budget is the binding limit.
4. Eviction changed from "negative score at first judgement" to "weakest quarter by score per example, only when slots are scarce". A leaf that exhausts its features retries them with a doubled evidence requirement.

**E1.4 budget sweep on text (after all four changes).** `experiments/e1_4_budget.py`, TinyShakespeare, λ=32, threshold 8 bits. Prequential bits per character:

| Budget (ops/update) | Batch | Min examples | Suffix only | Full DSL |
|---|---|---|---|---|
| 25k | 256 | 32 | 4.780 (no splits) | 4.780 (no splits) |
| 100k | 256 | 32 | 3.078 | 3.045 |
| 400k | 256 | 32 | 2.611 | 2.858 |
| 400k | 256 | 128 | 2.602 | 2.984 |
| 400k | 256 | 512 | 2.798 | 2.958 |
| 400k | 64 | 32 | 2.546 | 2.691 |
| 1.6M | 256 | 32 | 2.547 | 2.694 |

- Budget violations: 0 in every run.
- Loss now falls with budget, as it should. Before change 3 it did not (100k and 1.6M gave the same loss), because the 256-candidate cap was the real limit.
- Best result is 2.546 (2.398 over the final 10%), with about 3,200 leaves. That is inside the 2.3–2.6 range I first predicted, and still 0.37 bits worse than the n-gram reference (2.178), which holds over 200k contexts.
- Cost is far out of proportion: about 6,000 counted ops per token at the best settings, against roughly 9 node visits per token for the n-gram.
- 25k is below the cost of judging a single 66-value candidate, so nothing is ever reviewed. That is a floor of this implementation.
- The full DSL is worse than suffix-only at every budget except 100k (by 0.03). The E1.3 prediction that non-suffix features add less than 0.02 bits holds in the strong sense: at these budgets they cost more than they return.
- The full-DSL runs at 1.6M budget and at batch 64 took about 25 minutes each, well over the two-minute run envelope in the plan.

**E1.1 rerun: the text fixes regressed the synthetic sources.** Tail excess, 5 seeds:

| Source | After changes 1–2 | After changes 1–4 (current code) | Best CTW |
|---|---|---|---|
| S2 agreement | 0.005–0.009, 5–6 leaves | 0.032–0.055, 17–21 leaves | 0.055 |
| S3 brackets | 0.259–0.305 | 0.253–0.271 | 0.184 |
| S4 counter | under 0.01 on 3 of 5 seeds | 0.119–0.135 on all 5 | 0.170 |
| S5 XOR | 0.044–0.115 | 0.096–0.114 | 0.096 |

- After changes 1–2 the S2 tree was the planted rule: `tok[last('.')+1] in {keys,dogs,cats}` under a "previous token is a noun" split.
- With the current code, S4 is no longer solved on any seed and S2 is back near CTW's level. I have not diagnosed this. The likely cause is that candidates are no longer released, so each leaf's 32 slots fill with cheap features and the enumerator never reaches `count(...) mod 3`.
- S5 is unsolved under the current code and was solved under the first-pass code, so its outcome depends on pool policy and it cannot be cited either way.

**Where Stage 1 stands.**

- The credit signal works: when the right feature is in the pool, marginal-likelihood gain identifies it within a few hundred examples.
- The search policy does not: which features get a slot, and for how long, decides the outcome, and no single policy tried so far is good on both the synthetic sources and text.
- E1.3 (the λ and ablation table above) was run before changes 3–4 and is superseded by the E1.4 table.
- E1.5 (commit-rule comparison) has not been run.
- Open before Stage 2: diagnose the S4/S2 regression; decide whether per-leaf slot turnover should depend on feature cost; reduce run time on text.
