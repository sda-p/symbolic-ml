# Experimental programme: first stages

Status: plan, written 2026-11-11. Nothing here has been run yet.
Source of ideas: [initial-research.md](initial-research.md) (cited below as §n). It is a starting point, not doctrine.

## What we are trying to find out

Can a purely symbolic next-token predictor be trained by an SGD-like loop: a stream of small batches, a bounded amount of work per update, and a credit signal that says which discrete program edits would reduce loss?

That splits into three questions, in the order we can answer them:

1. **Does a bounded-update symbolic learner work at all**, beyond fixed-suffix count models? (Stage 1)
2. **Which credit mechanism gives the most accurate edit ranking per unit of work** once programs are deeper than one level? (Stage 2)
3. **Can credit pass through learned intermediate interfaces**, so that features feed features? (Stage 3)

## Where I expect the report to bend

These are predictions made before any experiment. Each one is tested below.

- **Shared loss circuits (§3) may not pay for themselves.** At the all-zero boundary their gradient equals the singleton loss differences, which `d` cached re-evaluations also give. The circuit only wins if it is smaller than those `d` runs, or if interior-point (interaction-averaged) credit turns out to rank edits better. The baseline to beat is incremental re-evaluation, not naive re-execution.
- **Per-batch Hoeffding certification (§8) is vacuous at our scale.** With V=64 and ε=1e-3 the loss bound is M ≈ 11 nats; for b=32, k=100 candidates and δ=0.1 the radius is about 7.6 nats, far above any real edit gain. Evidence has to accumulate across updates, and variance-adaptive bounds will be needed.
- **Raw information gain (§7) will overfit a synthesised predicate pool.** Selecting the best of thousands of predicates is a multiple-comparisons problem. I plan to score splits by the change in Dirichlet marginal likelihood (the `K_s` of §7) instead: it is exact, incrementally updatable, and is a prequential code length, so it charges for the extra parameters automatically.
- **Count decoders fragment data.** Each split halves the evidence per state, and nothing shares statistical strength across states. I expect this to be the ceiling on natural text, and CTW-style mixing along the learned tree to be the first mitigation.
- **Greedy single edits will fail on planted interactions** (the `1 − z₁z₂` case of §2). This is expected; the interesting measurement is how cheaply coordinated edits fix it.

## Tech stack

| Choice | Decision | Reason |
|---|---|---|
| Language | Python 3.13, managed with `uv` | Fastest iteration for a programme whose design will change weekly. |
| Numerics | NumPy only | See "columnar evaluation" below. No autodiff framework: nothing here is differentiated through. |
| Tests | `pytest` + `hypothesis` | The report's identities are exact, so they are checked as properties against brute force. |
| Exact arithmetic | integer counts; float64 log-loss via `lgamma`; `fractions` in tests only | Rationals everywhere (§1) would dominate run time without changing any conclusion at this horizon. |
| Results | one JSONL file per run under `results/`, plots with `matplotlib` | Diffable, no database. |
| Parallelism | `multiprocessing` over seeds and configs | 10 cores; individual runs stay single-threaded and deterministic. |

**Columnar evaluation** is the central implementation decision. A DSL term is evaluated over a whole batch of contexts at once, so its value is an array of length N over a small finite type. Consequences:

- Python overhead is paid per program node, not per example.
- Two candidates with the same value array are observationally equivalent on the batch, so bottom-up enumeration deduplicates by hashing arrays. This is our form of the computation sharing that §3 attributes to SUP nodes.
- A loss pullback table (§4) is an `[N, K]` array, and scoring a candidate against it is a single gather.
- Split statistics are `bincount` calls.

**Work is measured in counted primitive operations**, not wall time. Every evaluator and learner increments an op counter, and each update has an explicit op budget. The claim under test is about bounded work, so the bound must be machine-independent and enforced, not observed.

Deferred, with the trigger for revisiting:

- *Rust kernel via PyO3*: when profiling shows a hot loop (most likely CTW's per-token path update) limiting experiment turnaround.
- *HVM/Bend for SUP-node sharing*: only if Stage 2 shows that sharing beyond hash-consing and observational equivalence is the bottleneck.
- *A small neural reference model*: when the work moves to a larger machine. Until then the reference points are n-grams and CTW.

Laptop envelope (M4, 16 GB): vocabulary ≤ 64, streams of 10⁵–10⁶ tokens, context horizon H ≤ 16, each run under about two minutes so that sweeps stay cheap.

## Repository layout

```
src/symbolic_ml/
  data/       synthetic sources, text loaders, stream/microbatch iterator
  dsl/        types, terms, columnar evaluator, enumerator, op counter
  models/     n-gram, CTW, predicate-tree model
  credit/     pullback tables, edit-loss circuits, stochastic estimators, brute-force oracle
  learn/      bounded-update loop, candidate pool, commit rules
experiments/  one script per experiment ID (e.g. e1_3_text.py)
tests/
docs/         plan.md, log.md (dated results, including negative ones)
results/      run outputs (git-ignored)
```

Each experiment records its hypothesis and predicted outcome in `docs/log.md` before it runs, and the result afterwards.

## Data

Synthetic sources come first because the generating structure is known: we can compute the source entropy and check whether the learner recovered the planted rule.

| ID | Source | What it probes |
|---|---|---|
| S1 | Random variable-order Markov source, depth ≤ 4, V=8 | CTW's home ground; harness sanity check. |
| S2 | Agreement: a verb form depends on the first noun's number, across a variable-length distractor span (the example of §4) | One selector predicate replaces exponentially many suffix contexts. |
| S3 | Bounded-depth brackets, two bracket types | The closer depends on a matched opener at variable distance. |
| S4 | Counter: next token depends on a marker count mod m since the last delimiter | Features that suffix tests cannot express compactly. |
| S5 | XOR: next token depends on the XOR of two individually uninformative context features | Planted pairwise interaction; singleton credit is zero. |
| R1 | TinyShakespeare, character level (≈1.1 MB, V=65) | First natural text. |
| R2 | First 10 MB of text8 (V=27) | Second natural text, added once R1 runs are routine. |

Metrics throughout: prequential (predict-then-update) bits per token, held-out bits per token, model size in bits, and counted ops per update.

## Stage 0: harness and baselines

Build the stream iterator, prequential evaluator, op counter, S1–S5 generators, and two baselines: smoothed n-grams and the context-tree mixture of §7.

- **E0.1** CTW on S1: cumulative excess loss over the true source stays within the §7 bound.
- **E0.2** CTW and n-gram depth sweep on R1, to fix the reference numbers that later stages are compared against.
- **E0.3** CTW on S2–S5 as a function of depth, to confirm that these sources are hard for suffix models in the intended way. A source that CTW solves cheaply gets redesigned.

Exit: baselines reproduce the bound on S1 and give stable numbers on R1.

## Stage 1: a one-level learner (predicate-tree language model)

This is the shortest path to an end-to-end learning curve, following §7.

**Model.** A decision tree whose internal nodes are DSL predicates over the context and whose leaves hold token counts with Dirichlet smoothing. Prediction mixes along the root-to-leaf path, as CTW does along a suffix path.

**DSL v0.** First-order, total, finitely typed: token at offset, membership in a token class, equality, "last position satisfying a predicate before a delimiter", bounded count mod m, Boolean connectives. Token classes are themselves learnable sets. No recursion or user-defined functions yet.

**Update loop.** Per microbatch of b tokens and an op budget W:

1. Predict, record loss, update leaf counts.
2. Spend the remaining budget advancing a persistent candidate pool: enumerate a few new predicates, accumulate split statistics for existing candidates on this batch, evict the weakest.
3. Commit at most one split when its accumulated marginal-likelihood gain clears a threshold.

The candidate pool carries search across updates (§1), and the commit threshold plays the role of a learning rate.

Experiments:

- **E1.1 Recovery.** On S2–S4, does the learner find the planted predicate, and how many tokens does it need to get within 0.05 bits of source entropy, compared with CTW?
- **E1.2 Interaction failure.** On S5, confirm and quantify the expected failure of greedy single splits. This is the motivating case for Stage 3.
- **E1.3 Natural text.** On R1, bits per character against CTW at matched data. Ablation: restrict the DSL to suffix tests, which should reduce the learner to an online context tree and so checks the implementation. Inspect which predicates were chosen.
- **E1.4 Budget.** Loss against op budget W, batch size b, and replay buffer on or off (§7 notes that old counts cannot score new features).
- **E1.5 Commit rule.** Raw information gain, marginal-likelihood gain, and a confidence-bound rule, compared on false-commit rate and final loss.

Decision after Stage 1: if synthesised predicates give no gain over the suffix-only ablation on R1, the bottleneck is the edit vocabulary (§9), and abstraction learning moves ahead of Stage 3.

## Stage 2: credit-assignment benchmark

Before building a deeper learner, measure the candidate credit mechanisms against ground truth. For small programs and edit sets, a brute-force oracle computes the full loss-difference matrix `J` (§2) by re-execution. Each mechanism is scored on how well it ranks edits (top-k recall, rank correlation) per counted op. Edit neighbourhoods are taken from Stage 1 runs where possible, so they are realistic, not random.

- **E2.1 Pullback tables (§4).** Property-test exactness on expression trees. Measure the saving from scoring candidate subtrees by table lookup instead of re-execution, as a function of interface size K.
- **E2.2 Reuse.** When a function is called at m sites, how wrong is summed per-call credit, and how fast does the exact joint table grow?
- **E2.3 Edit-loss circuits (§3).** Compile per-example decision diagrams over d edit choices. Record diagram size and compile cost against d, and compare total cost with d cached re-evaluations.
- **E2.4 Interaction spectrum.** For real neighbourhoods, how much of the loss variation sits in order-1, order-2 and higher terms of the multilinear expansion? This decides whether first-order credit is a usable approximation.
- **E2.5 Stochastic estimators (§6).** REINFORCE and ARM: check the `(d−1)/R` variance law empirically, then place them on the same accuracy-per-op axis as the exact methods.

Exit: a table of mechanisms by accuracy and cost that says which ones Stage 3 uses.

## Stage 3: learning through intermediate interfaces

Programs of the form `emit(g(h(x)))`, where `h` maps contexts to a learned finite type of size K ≤ 16 and `g` consumes it. Train by alternating local synthesis: pull the loss back to the interface as a table, synthesise `h` against it, refit `g`, repeat.

- **E3.1** Cold start: a new feature is useless until something consumes it. Compare singleton credit with coordinated feature-plus-decoder edits on S5.
- **E3.2** A compositional synthetic source that needs two stacked features, to test whether alternating synthesis converges and at what cost.
- **E3.3** R1 with two-level programs, against the Stage 1 learner at equal op budget.

## Later, in outline

- Abstraction learning: consolidate recurring predicate fragments into library functions, scored on predictive use, compression, and counterfactual evaluation cost (§9).
- Incremental change propagation for edits to shared definitions (§5).
- Larger vocabularies and corpora, a neural reference model, and a compiled kernel, on a bigger machine.

## Order of work

Stage 0, then Stage 1, then Stage 2, then Stage 3. Stage 2's oracle and pullback tables have no dependency on Stage 1 and can be built alongside it if Stage 1 experiments are waiting on runs.
