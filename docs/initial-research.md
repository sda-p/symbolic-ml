There are several mathematically credible avenues, and a restricted version of your thought experiment is already achievable. The central object should be **a calculus of counterfactual program edits**: how replacing a function, changing a branch, or introducing an abstraction changes predictive loss.

The strongest positive result is this:

> **If the effects of many discrete program edits can be represented by a compact shared decision graph, then all their credit signals can be computed in a forward–backward pass through that graph. At an appropriate boundary point, those signals are exactly the individual edits’ loss differences.**

Every candidate program can remain fully symbolic. The substantial assumption is that the shared representation remains compact and can be constructed cheaply.

A second promising construction propagates **functions of downstream loss** backward through symbolic computation. At a small finite intermediate type, these functions become small tables. Those tables turn end-to-end learning into local, cost-sensitive program synthesis.

I’ll develop these constructions, explain where Taelin’s work fits, and give a concrete symbolic sequence learner with bounded microbatch updates and a learning guarantee.

## 1. Fixing the computational problem

We need to distinguish three questions:

| Question | What would establish it? |
|---|---|
| Can a lambda program represent a useful language predictor? | An appropriate program exists. |
| Can it learn from bounded updates? | An explicit update algorithm improves prediction within a specified model class. |
| Can structural credit assignment scale? | The cost of evaluating or estimating useful edits grows acceptably with program size. |

Universality addresses the first question. The interesting work concerns the other two.

### A bounded symbolic language model

Let the vocabulary be a finite set \(\mathcal V\), with \(V=|\mathcal V|\). A training example is

\[
(x,y)\in \mathcal V^{\le H}\times\mathcal V,
\]

where \(x\) is a bounded preceding context and \(y\) the next token.

Use a typed lambda calculus with:

- finite token and Boolean types;
- products, sums, bounded lists, and finite maps;
- conditionals and bounded folds;
- explicit arithmetic primitives;
- a deterministic evaluation strategy.

Fix limits on program size, evaluator steps, heap size, and arithmetic precision. Resource exhaustion produces a declared fallback prediction. This makes evaluation total and removes termination ambiguity.

A program computes

\[
P:\operatorname{Context}_H\to\operatorname{Dist}(\mathcal V).
\]

The distribution can be represented symbolically—for example, by integer counts with rational normalization. No neural representation is required. Require positive support, such as

\[
q_P(v\mid x)\ge \frac{\epsilon}{V}.
\]

Then next-token log loss is bounded:

\[
\ell(P;x,y)=-\log q_P(y\mid x),
\qquad
0\le \ell\le M:=\log(V/\epsilon).
\]

Positive support handles the uncertainty of language prediction and gives useful statistical bounds. Token identifiers themselves remain opaque symbols; numerical distance between token IDs has no role.

For a microbatch \(B\), define

\[
L_B(P)=\frac1{|B|}\sum_{(x,y)\in B}\ell(P;x,y),
\]

with an optional program-complexity penalty \(\lambda\,\operatorname{bits}(P)\).

The learner maintains a program, counts, caches, and edit statistics. Each update receives at most \(b\) examples and has an explicit work budget. It may carry unfinished search across updates.

**“Bounded microbatch” must bound the search work as well as the data batch.** Otherwise an update could quietly perform exponential synthesis on ten examples.

For an indefinitely growing stream, exact counters and rational arithmetic also grow in bit length. We must either fix an experimental horizon, charge those bit costs, or analyze finite-precision approximation.

## 2. What a symbolic Jacobian should mean

### Directions must specify actual interventions

A function name alone does not define a direction. Replacing a function with an identity, a constant, or a conditional produces different changes.

A direction should therefore be something like

\[
e_j=(\text{typed location},\text{replacement implementation}).
\]

For a fixed incumbent \(P\), let \(P_{e_j}\) denote the edited program. An exact loss-difference matrix is

\[
\boxed{
J_{ij}
=
\ell(P_{e_j};x_i,y_i)-\ell(P;x_i,y_i).
}
\]

Its column mean is the empirical credit of edit \(j\):

\[
d_j=\frac1b\sum_i J_{ij}.
\]

This is a reasonable symbolic analogue of a Jacobian. It is relative to an edit language and a current program; there is no canonical Jacobian of an arbitrary lambda term.

One can instead record differences in output probabilities. But converting those differences to log-loss changes requires the exact nonlinear expression

\[
\Delta\ell
=
-\log\left(1+\frac{\Delta q(y\mid x)}{q_P(y\mid x)}\right).
\]

The approximation \(-\Delta q/q\) needs a genuinely small relative probability change. A short syntactic edit provides no such guarantee.

### Independent labels do not produce independent effects

Suppose \(d\) compatible binary edits define a family

\[
P_z,\qquad z\in\{0,1\}^d,\qquad P_0=P.
\]

Compatibility can be guaranteed by using distinct typed holes in a fixed template. If a hole represents a function definition, its choice must be shared across **every invocation of that function**.

Write \(f(z)=L_B(P_z)\). Every such function has an exact multilinear expansion:

\[
f(z)=c_\varnothing+\sum_i c_i z_i
+\sum_{i<j}c_{ij}z_i z_j+\cdots.
\]

Here

\[
c_i=f(e_i)-f(0),
\]

and

\[
c_{ij}
=
f(e_i+e_j)-f(e_i)-f(e_j)+f(0).
\]

The higher-order coefficients quantify interactions between edits. This is standard Boolean-function algebra, applied here to the loss landscape over program modifications. [cs.cmu.edu](https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf?utm_source=chatgpt.com)

Consider

\[
f(z_1,z_2)=1-z_1z_2.
\]

Both individual edit credits are zero at the incumbent:

\[
f(1,0)-f(0,0)=f(0,1)-f(0,0)=0.
\]

The joint edit improves loss by one.

This captures a common structural-learning problem: introducing a feature has no effect until another function uses it.

**A unique direction label gives provenance. It does not, by itself, supply the magnitude or sign of an intervention’s effect.** Those require counterfactual evaluation or exploitable semantic structure.

## 3. The strongest bridge to SUP nodes: shared loss circuits

### What Taelin’s work contributes

Taelin’s SUP-node examples represent alternatives within a shared symbolic evaluation. The ADD-CARRY demonstration fills Boolean holes in a supplied recursive template, evaluates constraints against the shared alternatives, and extracts a surviving assignment. The stated acceleration comes from sharing computation across related candidates. Taelin explicitly acknowledges that the general search problem remains exponential. [GitHub](https://gist.github.com/VictorTaelin/7fe49a99ebca42e5721aa1a3bb32e278?utm_source=chatgpt.com)

The Interaction Calculus specification makes choice labels operationally significant: matching and distinct SUP/DUP labels interact differently. For learning, the essential requirement is to preserve correlations between repeated uses of the same structural decision. [GitHub](https://github.com/VictorTaelin/Interaction-Calculus?utm_source=chatgpt.com)

These sources provide an execution-sharing mechanism. A loss-credit mechanism can be built on top of a suitable representation of its results.

### A differentiable distribution over discrete edits

Choose independent edit probabilities

\[
Z_i\sim\operatorname{Bernoulli}(p_i)
\]

and define

\[
F(p)=\mathbb E[f(Z)].
\]

This finite expectation is a multilinear polynomial. Differentiating the finite sum gives

\[
\boxed{
\frac{\partial F}{\partial p_i}
=
\mathbb E_{Z_{-i}}
\left[
f(1,Z_{-i})-f(0,Z_{-i})
\right].
}
\]

The derivative measures the average counterfactual effect of choosing edit \(i\), given the current distribution of surrounding edits.

At the all-zero boundary,

\[
\boxed{
\left.\frac{\partial F}{\partial p_i}\right|_{p=0}
=
f(e_i)-f(0).
}
\]

Thus the exact local symbolic edit differences appear as ordinary derivatives of an aggregation over discrete programs. No derivative of program execution is involved.

### Conditional theorem: all edit credits in one shared pass

Suppose \(f\) has been compiled into an ordered algebraic decision diagram \(G\):

- internal nodes test edit choices;
- leaves contain scalar losses;
- identical subproblems are shared;
- no choice variable is retested along a path.

At a node \(v\) testing choice \(i\), with children \(v_0,v_1\), compute

\[
V_v=(1-p_i)V_{v_0}+p_iV_{v_1}.
\]

At a leaf, \(V_v\) is its loss. By induction, the root value is \(F(p)\).

Now initialize \(a_{\mathrm{root}}=1\) and propagate backward:

\[
a_{v_0}\mathrel{+}=a_v(1-p_i),
\qquad
a_{v_1}\mathrel{+}=a_vp_i.
\]

Then

\[
\boxed{
\frac{\partial F}{\partial p_i}
=
\sum_{v:\operatorname{label}(v)=i}
a_v\left(V_{v_1}-V_{v_0}\right).
}
\]

The proof is the chain rule applied to the weighted decision recurrence.

**After compilation, evaluation and all \(d\) credits cost \(O(|G|+d)\) arithmetic operations.** At \(p=0\), these are exactly all singleton-edit loss differences.

For a microbatch, one can compile each example’s loss separately and average the resulting gradients. This can be substantially smaller than constructing a single diagram whose leaves contain every possible sum of example losses.

The underlying strategy has close precedents in Darwiche’s differentiation of compiled probabilistic inference circuits and Li and Eisner’s expectation semirings over shared forests of discrete derivations. The specialization to an edit-to-loss function is the construction above. [arxiv.org](https://arxiv.org/pdf/1301.3847?utm_source=chatgpt.com)

### Where the computational difficulty remains

The total cost is

\[
C_{\mathrm{compile}}+O(|G|+d).
\]

Neither term is generally bounded by the cost of executing one candidate.

A program can have a short execution trace while its counterfactual loss diagram is enormous. Different edits can change control flow, produce distinct intermediate values, or destroy opportunities for sharing. A SUP evaluation graph also does not automatically satisfy the ordering or factorization conditions needed for the recurrence.

There is a further objective distinction:

\[
\mathbb E_P[-\log q_P(y\mid x)]
\ne
-\log\mathbb E_P[q_P(y\mid x)].
\]

The left side evaluates the expected performance of individual candidate programs. The right side evaluates an ensemble predictor. Both are legitimate objectives, but substituting one for the other changes the problem.

**The research question is therefore concrete:** do useful local neighborhoods of language programs admit small, cheaply maintainable loss circuits?

## 4. Exact symbolic reverse propagation through small interfaces

There is another route that operates inside a program.

### The discrete adjoint is a loss function

Let \(f:A\to B\) be a deterministic function between finite sets. Define its action on one-hot vectors by

\[
M_f\mathbf e_a=\mathbf e_{f(a)}.
\]

Then composition is exactly linear at this representation level:

\[
M_{g\circ f}=M_gM_f.
\]

For a downstream loss function \(\Lambda:B\to\mathbb R\), the reverse action is

\[
\boxed{
f^*\Lambda=\Lambda\circ f.
}
\]

Consequently,

\[
(g\circ f)^*=f^*g^*.
\]

This supplies an exact reverse composition rule for arbitrary discrete functions.

The difference from ordinary numerical backpropagation is the size of the reverse object. At an intermediate type with \(K\) possible values, an unrestricted adjoint is a table of \(K\) downstream costs. At a large sequence or function type, it can be enormous.

Representing that table implicitly as a closure is always possible, but querying the closure may simply rerun the downstream computation.

### A tractable theorem for expression trees

Consider a total, acyclic expression tree. Node \(v\) computes

\[
y_v=f_v(y_{u_1},\ldots,y_{u_r}),
\]

with bounded arity and finite intermediate domains of size at most \(K\).

After evaluating the baseline, define

\[
C_v(a)
=
\text{root loss if the value at }v\text{ is replaced by }a.
\]

For child \(u_j\),

\[
\boxed{
C_{u_j}(a)=
C_v\!\left(
f_v(y_{u_1},\ldots,a,\ldots,y_{u_r})
\right).
}
\]

All siblings retain their baseline values.

The proof follows the unique path from the changed node to the root. With constant-cost primitives and \(N\) tree nodes, all tables cost \(O(NK)\) per example.

Now suppose a candidate subtree \(h\) replaces the subtree at \(v\). Its exact microbatch loss is

\[
\boxed{
L_B(P[v\leftarrow h])
=
\frac1b\sum_{i=1}^b
C_{iv}\!\left(\llbracket h\rrbracket\rho_{iv}\right),
}
\]

where \(\rho_{iv}\) is the cached environment at that location.

This is a useful acceleration. Many candidate subprograms can be scored by evaluating only their local behavior and consulting precomputed cost tables.

It also creates a learning target for intermediate functions: **synthesize a program whose outputs have low downstream cost on the observed environments.**

### A language example

Consider a toy predictor

\[
P(x)=\operatorname{emit}(\operatorname{number}(h(x))),
\]

where \(h\) selects a noun and `number` returns `singular` or `plural`. Let the emission table be

| State | \(q(\text{is})\) | \(q(\text{are})\) |
|---|---:|---:|
| Singular | 0.9 | 0.1 |
| Plural | 0.1 | 0.9 |

For the examples

- “the key beside the cabinets” \(\to\) “is”;
- “the keys beside the cabinet” \(\to\) “are”,

the reverse loss tables are

| Intermediate state | First example | Second example |
|---|---:|---:|
| Singular | \(-\log 0.9\) | \(-\log 0.1\) |
| Plural | \(-\log 0.1\) | \(-\log 0.9\) |

Replacing a final-noun selector with a first-noun selector improves the average loss by exactly

\[
-\log 0.9+\log 0.1=-\log 9.
\]

No derivative of noun selection is needed. The example establishes the credit mechanism; the particular selector is only a toy hypothesis.

### The crucial restrictions

The theorem applies to a fixed total expression tree, not automatically to an arbitrary recorded lambda-reduction trace.

If a function is used at \(m\) locations within one prediction, a global replacement changes all \(m\) outputs together:

\[
C(f'(a_1),\ldots,f'(a_m)).
\]

Adding separate callwise credits is generally wrong. An unrestricted joint cost table can require \(K^m\) entries. Recursion can also change later call arguments, making the old argument cache insufficient.

Condition changes require evaluating newly selected branches. Runtime penalties or timeouts require resource information in the interface; a table indexed only by return value cannot account for them.

This suggests a useful design principle:

> **Expose small symbolic interfaces at which useful counterfactual loss functions remain cheap to represent.**

Finite categories, bounded parser states, small action alphabets, and explicit memory summaries are plausible interfaces. Arbitrary closures are much harder.

## 5. Incremental lambda calculus supplies exact change semantics

There is established work close to the underlying idea. Cai and colleagues’ Incremental Lambda Calculus defines change structures and a transformation that propagates changes through higher-order programs, including first-class functions. Its correctness is proved for a family of simply typed calculi. [inc-lc.github.io](https://inc-lc.github.io/resources/pldi14-ilc-author-final.pdf?utm_source=chatgpt.com)

For an update operation \(\oplus\), a valid finite-change derivative satisfies

\[
f(x\oplus\delta x)
=
f(x)\oplus Df(x,\delta x).
\]

A compositional derivative can be defined by

\[
D(g\circ f)(x,\delta x)
:=
Dg\!\left(f(x),Df(x,\delta x)\right).
\]

No infinitesimal limit or linearity in the change is required.

For structural learning, write

\[
P=F(f_1,\ldots,f_m).
\]

Replacing one \(f_i\) becomes a higher-order input change to the fixed surrounding program \(F\). This supplies a principled basis for an incremental edit evaluator.

Useful implementations would exploit:

- unchanged subcomputations;
- sparse changes to maps or counts;
- cached aggregates;
- stable dependency structure;
- specialized update rules for primitives.

An arbitrary function replacement can still force complete recomputation. The contribution is a sound way to exploit locality when locality exists.

Combined with the previous section, this gives two complementary operations: propagate candidate changes forward cheaply, and propagate downstream costs backward through manageable interfaces.

## 6. Statistical estimation when exact sharing becomes too large

The distributional objective \(F(p)=\mathbb E[f(Z)]\) remains usable even when its exact decision graph is intractable.

### Unbiased gradients from symbolic executions

For \(0<p_i<1\),

\[
\frac{\partial F}{\partial p_i}
=
\mathbb E\left[
(f(Z)-b_i)
\frac{Z_i-p_i}{p_i(1-p_i)}
\right],
\]

where the baseline \(b_i\) must not depend on \(Z_i\), conditional on the other choices.

With \(p_i=\sigma(\theta_i)\),

\[
\boxed{
\frac{\partial F}{\partial\theta_i}
=
\mathbb E[(f(Z)-b_i)(Z_i-p_i)].
}
\]

This is the score-function/REINFORCE construction applied to program choices. It works with discrete branching and arbitrary bounded symbolic execution. [doi.org](https://doi.org/10.1007/bf00992696?utm_source=chatgpt.com)

There are also paired estimators. For independent \(U_i\sim\operatorname{Uniform}(0,1)\), define

\[
Z_i^+=\mathbf1\{U_i>1-p_i\},
\qquad
Z_i^-=\mathbf1\{U_i<p_i\}.
\]

The ARM identity gives

\[
\boxed{
\nabla_\theta F
=
\mathbb E\left[
(f(Z^+)-f(Z^-))(U-\tfrac12)
\right].
}
\]

Thus two complete discrete executions produce one unbiased gradient estimate for every choice coordinate. This estimator is established prior art; the proposed application is to correlated function choices evaluated with symbolic sharing. [arxiv.org](https://arxiv.org/html/1807.11143v2?utm_source=chatgpt.com)

### Two executions do not imply dimension-independent learning

A simple calculation exposes the limitation.

Suppose, in sign coordinates,

\[
h(s)=c+\sum_{j=1}^d a_js_j,
\qquad s_j\in\{-1,1\}.
\]

For independent random signs,

\[
\widehat a_i
=
\frac{h(s)-h(-s)}2s_i
\]

is unbiased, but

\[
\operatorname{Var}(\widehat a_i)
=
\sum_{j\ne i}a_j^2.
\]

Averaging \(R\) probes gives

\[
\boxed{
\mathbb E\|\widehat a-a\|_2^2
=
\frac{d-1}{R}\|a\|_2^2.
}
\]

Even an additive landscape can require many probes to estimate a dense credit vector accurately.

Rare perturbations create another problem. If each of \(d\) edits is sampled with probability \(O(1/d)\) to keep the expected edit radius constant, evidence about any particular coordinate arrives infrequently.

### When random directions become promising

They become more attractive under **sparsity and controlled interactions**.

In the additive example, paired probes give linear measurements

\[
y_r=\sum_i s_i^{(r)}a_i.
\]

If \(a\) is sparse, this is a sparse-recovery problem. Simultaneous perturbation combined with compressive sensing has been studied for numerical gradients; the discrete specialization here requires approximately additive edit effects and appropriate recovery assumptions. [arxiv.org](https://arxiv.org/abs/1511.08768?utm_source=chatgpt.com)

The interaction expansion makes the approximation precise. If all edit probabilities equal \(\rho\),

\[
\boxed{
\partial_{p_i}F(\rho\mathbf1)
=
d_i+
\sum_{\substack{S\ni i\\|S|\ge2}}
c_S\rho^{|S|-1}.
}
\]

Hence the difference from incumbent single-edit credit is bounded by the corresponding weighted sum of interaction magnitudes.

Small \(\rho\) helps only when that interaction mass is controlled. A large number of interactions can overwhelm the small probability factor.

My assessment is that stochastic estimation is a useful fallback and exploration mechanism. Exact sharing, informative intermediate interfaces, and a good edit vocabulary must provide much of the efficiency.

## 7. A constructive symbolic language learner

There is a clean positive answer to the existence part of the thought experiment.

### Context-tree programs

Consider programs that inspect a suffix of the context and return a count-based token distribution. Each possible context node either:

1. stops and uses its own token counts; or
2. examines one more preceding token and delegates to child contexts.

This is the setting of context-tree weighting and Bayesian context trees: uncertainty over many discrete predictive structures can be aggregated recursively. [Information Theory, IEEE Transactions on](https://www.cs.cmu.edu/~aarti/Class/10704_Fall16/CTW.pdf?utm_source=chatgpt.com)

Here is a simple version with an elementary guarantee.

Fix maximum depth \(H\), and condition on an initial \(H\)-token history. At suffix \(s\), let \(n_{s,v}\) be token counts and \(n_s=\sum_v n_{s,v}\). Use a Dirichlet-\(1\) leaf model, whose integrated sequence likelihood is

\[
K_s
=
\frac{(V-1)!\prod_v n_{s,v}!}{(n_s+V-1)!}.
\]

Its predictive update is simply

\[
q_s(y)=\frac{n_{s,y}+1}{n_s+V}.
\]

For a fixed stop probability \(\rho\in(0,1)\), define

\[
W_s=
\begin{cases}
K_s,& |s|=H,\\[1mm]
\rho K_s+(1-\rho)\displaystyle\prod_{v\in\mathcal V}W_{vs},
& |s|<H.
\end{cases}
\]

Expanding this recurrence gives a mixture over all allowed context trees:

\[
W_\varnothing(D)=\sum_T\pi(T)K_T(D).
\]

Each \(T\) is a discrete symbolic predictor. The mixture itself is also implementable as one deterministic symbolic program returning probabilities.

For any fixed tree \(T\) with categorical parameters \(\theta\),

\[
\boxed{
-\log W_\varnothing(D)
\le
-\log p_{T,\theta}(D)
+\log\frac1{\pi(T)}
+
\sum_{s\in\operatorname{leaves}(T)}
\log\binom{n_s+V-1}{V-1}.
}
\]

The argument is elementary: the mixture contains the \(T\) term, and the Dirichlet-\(1\) likelihood is at least the maximum-likelihood sequence probability divided by the number of possible count vectors.

For fixed \(T,V\), excess cumulative loss is \(O(|T|V\log N)\). Average excess loss therefore tends to zero relative to every fixed predictor in this class.

Each new token changes only its \(H+1\) suffix nodes. Cached child products give \(O(bH)\) arithmetic operations for a microbatch’s observed-token likelihood updates. Exact bit costs still depend on the horizon, and emitting a full vocabulary vector has its own cost.

This is genuine bounded-update symbolic learning within a restricted class. Selecting a single MAP tree is also possible, but the mixture’s predictive guarantee does not automatically transfer to that changing selection.

### Generalizing the contexts with learned lambda programs

The more interesting extension replaces literal suffix tests with learned symbolic features:

\[
\phi:\operatorname{Context}_H\to\{1,\ldots,K\}.
\]

The output distribution is estimated from token counts within each feature state.

A useful strategy is to **evaluate a structural feature edit together with a refitted count decoder**. This gives a new feature credit immediately, even if the old decoder did not distinguish its outputs.

Ignoring smoothing for the following identity, the best empirical categorical decoder has total loss

\[
\mathcal L(\phi)
=
\sum_u n_u H(\widehat p_u).
\]

If a symbolic predicate \(b(x)\) splits one state \(u\), the change after refitting is exactly

\[
\boxed{
\Delta\mathcal L
=
-n_u\,I_D\!\left(Y;b(X)\mid\phi(X)=u\right).
}
\]

This is the familiar information-gain principle behind decision-tree induction. Here, the proposed extension is to synthesize the predicate as a typed lambda program and share evaluation across candidate predicates. [web.engr.oregonstate.edu](https://web.engr.oregonstate.edu/~xfern/classes/cs434/decisiontree-quinlan.pdf?utm_source=chatgpt.com)

This provides a concrete path from count models toward richer symbolic language models:

- learn predicates and context selectors;
- combine them into finite semantic states;
- learn reusable subprograms;
- score changes through count statistics or downstream loss tables;
- use smoothing and complexity penalties to control overfitting.

A limitation matters for streaming learning: counts for an old feature do not generally contain enough information to evaluate an arbitrary new feature retrospectively. New candidates need replay data or prospectively maintained statistics.

## 8. Making bounded updates reliable

A practical theoretical design can combine the preceding mechanisms.

Fix limits on active edits \(d_{\max}\), candidate-generation work, shared-graph size, stochastic probes, replay memory, and edits committed per update.

A bounded update then proceeds as follows:

1. **Predict and collect a microbatch.** Preserve pre-update predictions for honest sequential evaluation.
2. **Generate a small typed edit neighborhood.** Include replacements, splits, and selected coordinated edits.
3. **Reuse computation.** Apply incremental changes, cached environments, and finite loss tables.
4. **Aggregate credit exactly where feasible.** Build shared loss circuits under a strict construction budget.
5. **Probe remaining candidates statistically.** Use paired executions on the same examples.
6. **Evaluate the proposed concrete update.** Validate its actual combined effect, then commit it and invalidate affected caches.

Exceeding a graph-construction budget should trigger a smaller edit block or a sampling fallback. It must not silently exceed the promised update bound.

### A formal condition for combining credits

Suppose we want to commit a set \(S\) of compatible edits. Define a uniform pairwise interaction bound

\[
H_{ij}
=
\sup_z
\left|
f(z_{i=1,j=1})-f(z_{i=1,j=0})
-f(z_{i=0,j=1})+f(z_{i=0,j=0})
\right|,
\]

with the other coordinates fixed by \(z\).

Telescoping one edit at a time gives

\[
\boxed{
\left|
f(\mathbf1_S)-f(0)-\sum_{i\in S}d_i
\right|
\le
\sum_{\{i,j\}\subseteq S}H_{ij}.
}
\]

If estimates satisfy \(|\widehat d_i-d_i|\le\eta\), then

\[
f(\mathbf1_S)-f(0)
\le
\sum_{i\in S}\widehat d_i
+|S|\eta
+\sum_{\{i,j\}\subseteq S}H_{ij}.
\]

A negative right-hand side certifies improvement.

This gives a precise meaning to a useful symbolic first-order approximation: **credit estimates are accurate enough, and interactions within the proposed update are small enough.**

Measuring pairwise interactions only at the incumbent does not establish the supremum bound. A certified bound would need to follow from the DSL’s structure; an empirical estimate remains a heuristic.

### What a microbatch can certify

For a candidate fixed before drawing a fresh validation batch,

\[
X=\ell(P';x,y)-\ell(P;x,y)\in[-M,M].
\]

For \(k\) candidates fixed in advance, Hoeffding’s inequality gives simultaneous error at most \(\delta\) with radius

\[
r_b=M\sqrt{\frac{2\log(2k/\delta)}b}.
\]

Thus a sufficient acceptance condition is

\[
\widehat\Delta+r_b
+\lambda\bigl(\operatorname{bits}(P')-\operatorname{bits}(P)\bigr)
<0.
\]

This requires the relevant independence assumptions. Overlapping tokens in a text stream are not automatically independent samples. Candidates generated from the validation batch also require a different analysis.

The implication is unavoidable: **a fixed tiny microbatch cannot reliably resolve arbitrarily tiny improvements.** Evidence must accumulate across updates, or the learner must tolerate noisy decisions. Streaming tree learning provides relevant precedent for accumulating sufficient statistics and using confidence bounds to decide structural changes. [alchemy.cs.washington.edu](https://alchemy.cs.washington.edu/papers/pdfs/domingos-hulten00.pdf?utm_source=chatgpt.com)

## 9. Learning the edit vocabulary is probably essential

Even perfect credit assignment can fail when the available edits are poorly chosen.

For example:

- a new feature is useless until its consumer changes;
- a new recursive helper is useless until another function calls it;
- a useful transformation may require several coordinated syntax changes.

A reusable abstraction can turn such a combination into one meaningful edit. It changes the geometry of the search space.

DreamCoder supplies relevant precedent: it jointly learns a library of functional-program abstractions and a search policy, allowing later synthesis to operate with richer building blocks. It does not establish the language-learning system proposed here, but it directly addresses how useful program coordinates can be learned. [people.csail.mit.edu](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf?utm_source=chatgpt.com)

For this project, I would evaluate a learned abstraction along three dimensions:

| Property | Why it matters |
|---|---|
| Predictive usefulness | Its use improves held-out token prediction. |
| Description compression | It captures reusable structure rather than memorized exceptions. |
| Counterfactual evaluation cost | Alternatives involving it have small loss tables or compact shared circuits. |

The third criterion is especially relevant to your proposal. Among behaviorally equivalent refactorizations, some may expose interfaces that make credit assignment much cheaper.

Compression alone does not guarantee this. A heavily reused helper can also create tightly coupled effects across many calls.

A promising learner would therefore maintain both single-function edits and a bounded set of coordinated edits—especially feature-plus-decoder changes—and periodically consolidate successful combinations into reusable abstractions.

## 10. What remains impossible without assumptions—and what is worth pursuing

The bounded framework avoids undecidability, but it retains ordinary combinatorial hardness.

A loop-free symbolic program can evaluate a Boolean formula \(\varphi(z)\). Define candidate loss by

\[
f(z)=1-\varphi(z).
\]

Finding a globally optimal edit assignment then includes SAT. Exact expectation under uniformly sampled choices includes counting satisfying assignments. Equivalent two-level predictive-loss constructions can be embedded in the language-model framework.

Consequently, no assumption-free efficient procedure for arbitrary edit optimization or exact aggregation should be expected.

There is also a simpler oracle limitation: an arbitrary loss function can assign unrelated values to all singleton edits. Group queries reveal nothing about those values unless additional structure connects them. Unique direction tags cannot manufacture that connection.

The positive avenues are nevertheless substantial:

| Avenue | What can be established | What must stay controlled |
|---|---|---|
| Shared loss circuits | Exact all-edit credit in linear time after compilation | Circuit size and construction cost |
| Finite symbolic loss pullbacks | Exact local synthesis objectives | Interface cardinality and reconvergent dependencies |
| Incremental change semantics | Correct reuse across structural changes | Size of affected computation |
| Paired stochastic estimators | Unbiased distribution-gradient estimates | Variance, exploration radius, interactions |
| Count-based symbolic partitions | Explicit structural credit and bounded updates | Candidate-feature complexity and overfitting |
| Context-tree mixtures | A concrete online prediction guarantee | The restricted finite-memory model class |

**My preferred research direction is a typed lambda DSL with learned finite semantic interfaces, shared counterfactual evaluation, and count-based predictive components.** Start with a class whose online learnability is understood, such as context trees. Replace fixed context tests with synthesized predicates and selectors. Use exact loss pullbacks for small interfaces, decision circuits for compact edit blocks, and paired sampling elsewhere. Learn abstractions that make useful coordinated changes accessible.

The decisive theoretical advance would be a result about a particular expressive DSL: that its relevant counterfactual computations have bounded interaction width, small interfaces, or another property that guarantees compact credit computation.

That would substantiate the core idea in your prompt: **programs can remain symbolic while their edit space is deliberately structured to support efficient credit assignment.** The algebra needed for this already exists in several forms. The open challenge is learning and maintaining the program structure that makes the algebra computationally useful.