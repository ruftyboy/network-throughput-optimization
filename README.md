# Network throughput optimization

Research prototype for matching-constrained capacity augmentation in undirected multicommodity networks.

## Question
How much can maximum concurrent fractional flow increase when a small subset of links is doubled, with at most one upgraded link incident to any vertex?

## Reproduce
Python 3.11:
```bash
python -m pip install -r requirements.txt
python benchmark.py --real --seeds 3 --seconds 20
```
Output: results/benchmark.json (rates, demands, source hashes, solver status, bounds, upgrade choices and failures). GitHub Actions is intended to run the same benchmark and retain its output.

## Experimental protocol
- Three synthetic graphs plus historical Abilene, ESnet and GEANT2012 topology snapshots (downloaded from linked upstream sources).
- Largest connected simple undirected component; all edges have one shared unit of capacity and all 12 sampled demands have equal weight. Traffic and capacities are synthetic, NOT operator measurements.
- Budget floor(number of links / 20), i.e. at most 5% of links. On networks with fewer than 20 links this means zero upgrades.
- Compare 20 random matchings, edge-betweenness greedy, throughput-greedy on small graphs, and mixed-integer optimization under the same budget and routing model.
- A solver timeout is not an optimality proof. Report feasible incumbent and upper bound separately. Flow residuals are checked numerically, not via exact rational arithmetic.
- Three deterministic traffic seeds are a pilot, not statistically sufficient evidence. No performance target is guaranteed.

## Mathematical bound
For uniform added capacity delta, total shortest-path demand consumption minus original total capacity, divided by delta and rounded up (clamped at zero), lower-bounds the number of upgrades. The bound need not be attainable.

## Research status
The original 50-link result on the Braverman–He network remains a claimed result pending recovery and independent checking of the exact routing certificate. The MILP and baseline methods build on standard network optimization; novelty and superiority to published methods are not established.

## Sources
- Internet Topology Zoo: https://topology-zoo.org/ (Knight et al., The Internet Topology Zoo, IEEE JSAC, 2011).
- Abilene source: https://github.com/sk2/topologyzoo/blob/master/sources/Abilene.graphml
- ESnet source: https://github.com/mroughan/InternetTopologyZoo/blob/main/graphml/Esnet.graphml
- GEANT2012 mirror: https://github.com/PacktPublishing/Network-Science-with-Python-and-NetworkX-Quick-Start-Guide/blob/master/data/UAITZ/Geant2012.graphml
- Future benchmark with published demand matrices: https://sndlib.put.poznan.pl/

The initial script is retained in network_upgrade_experiment.py. Use benchmark.py for the more structured protocol. Dataset sources remain subject to their respective terms.


## Phase 2: published weighted demand matrices and heterogeneous capacity

Run `python sndlib_benchmark.py --seconds 25`. This separate pilot downloads the official SNDlib XML archive, records its SHA-256 and uses every positive default demand with its published routing-unit multiplier. Six initial instances: Abilene, GEANT, Germany50, Nobel-US, Polska and PDH.

Two distinct capacity interpretations are reported: positive preinstalled capacities only; and a hypothetical baseline filling missing/zero installed capacities with the smallest positive published candidate capacity module. The latter is NOT a measurement of installed capacity. No assumption that the data describe present-day operations. The model uses shared undirected capacities, splittable routing and ignores SNDlib cost, hop, survivability and path restrictions. The default matrix is a published benchmark, not necessarily a measured traffic trace.

The new dual-cut solver separates the binary matching master from a weighted multicommodity routing LP. At upgrade plan x, LP duality produces the global upper cut t(y) <= t(x) + g dot (y-x), where g is the vector of nonnegative link-capacity shadow prices times the capacity increments. The master maximizes t subject to these cuts and the upgrade budget. The best evaluated feasible matching gives a lower bound; master dual bounds give upper bounds. Compare with the direct MILP under nominally equal solver time limits; report gaps, times and timeouts, not just gains. The direct solver includes a separate routing verification LP, so total wall time can exceed its nominal limit. This is a classical decomposition approach, not a claimed novel algorithm or exact rational certificate.

See the SNDlib benchmark workflow for the JSON artifact. No speedup, throughput target, or superiority is established until results are inspected. The pilot uses default matrices; dynamic historical traffic and robust upgrades across multiple matrices remain future work.

SNDlib source and attribution: https://sndlib.put.poznan.pl/networks.overview.action ; Orlowski, Wessäly, Pióro and Tomaszewski, SNDlib 1.0—Survivable Network Design Library, Networks 55(3), 2010.


## Phase 3: robust upgrades across changing traffic

One fixed matching of upgraded links is chosen before seeing held-out traffic. Routing may adapt to each matrix. For scenario s, let F_s(x) be the maximum concurrent flow with upgrade vector x and let B_s=F_s(0)>0. Maximize min_s F_s(x)/B_s over training scenarios. Report every held-out gain, its minimum and mean; the training bound does not certify unseen matrices.

Historical traces: SNDlib dynamic matrices for Germany50 (288 five-minute matrices), GEANT (11,460 fifteen-minute matrices), and Abilene (48,096 five-minute matrices). Sample 3 training matrices from the first chronological half and 6 held-out matrices from the second half. This is a pilot sample, not a full historical replay. Preserve all positive published demand weights. Capacity policy is unchanged from Phase 2: module-derived capacities remain hypothetical. Abilene has zero allowed upgrades at the strict 5% budget. These historical matrices are measurement-derived, anonymized/aggregated as described by SNDlib; they are not current telemetry.

Topology sweep: deterministically select up to 120 historical Topology Zoo files after preprocessing to the largest connected simple undirected component and filtering to 20–250 edges. Distinct files may represent versions of the same network. Use unit link capacity and 12 independently sampled weighted demands per scenario, three training and six holdout scenarios per topology. Separate these synthetic results from historical traffic. Report exclusions and errors rather than silently dropping failures. Four parallel shards of 30 files each target 1,080 traffic matrices if enough eligible files exist.

### Algorithm and proof of valid strengthening

For each scenario, LP duality gives F_s(y)/B_s <= F_s(x)/B_s + g_s dot (y-x)/B_s. Every cut is valid for all feasible upgrade vectors. Furthermore, for each vertex v, summing the demand crossing the singleton cut gives D_s(v). Then
F_s(x)/B_s <= sum_{e incident to v} c_e(1+x_e)/(B_s D_s(v))
whenever D_s(v)>0. This is a valid master inequality because each incident demand must cross that cut at least once. Add these inequalities before the first master solve. The strengthened master feasible region is contained in the plain master's region given identical dual cuts, so its upper bound cannot increase. This does not establish a runtime advantage. With exact subproblem and master solves, the finite matching space and tight cuts at queried plans yield finite convergence; implementation uses numerical tolerances and time limits.

Compare the strengthened multi-scenario method against plain multi-scenario Benders on the historical traces, plus first-snapshot optimization, edge-betweenness and no upgrades. Compare with identical training samples and budget; report time and numerical gap. These are standard methods and valid inequalities, not a demonstrated novel theorem or state-of-the-art win. Full literature comparison, independently verified certificates and broader validation remain open.

Sources: https://sndlib.put.poznan.pl/dynamicmatrices.overview.action (cite the original dynamic data references listed there); https://sndlib.put.poznan.pl/download/germany50-rm.txt ; https://github.com/mroughan/InternetTopologyZoo (Knight et al., IEEE JSAC 2011, DOI 10.1109/JSAC.2011.111002). Related work: Benders decomposition approach for the robust network design problem with flow bifurcations, DOI 10.1002/net.21486; Robust network design: Formulations, valid inequalities, and computations, DOI 10.1002/net.21497.

Run locally: python robust_benchmark.py --mode traces --output results/traces.json
Run topology subset: python robust_benchmark.py --mode zoo --limit 30 --offset 0 --seconds 10 --output results/zoo-0.json


### Robust pilot correction
Run 45539395296 failed because NumPy scalar booleans were not JSON-serializable in the topology jobs and GEANT had an empty selected matrix. The corrected implementation serializes NumPy scalars explicitly and writes artifacts atomically. Historical selection now uses disjoint chronological strata within each half, filters timestamped demand-matrix filenames, and attempts up to 12 nearest candidates per stratum when a matrix is empty; all such exclusions are recorded. This changes the sampled timestamps, so do not treat the old and new runs as identical. Both robust solvers now share a training-only first-snapshot feasible warm start; its computation and training evaluation time are reported separately. Include that overhead in end-to-end runtime comparisons. No test data is used to select the starting plan.

The partial failed-run Germany50 artifact reported 25% gains on six held-out matrices for its first-snapshot baseline, while both robust methods returned zero-gain incumbents and did not close the gap. This is an exploratory result under modeled capacities, not an operational performance claim. The revised workflow must be validated before reporting aggregate results.


## Phase 4: balancing throughput and reliability

The previous max-min pilot did not consistently beat edge betweenness on unseen synthetic traffic. This phase tests a different objective on the same topology sampling rule but a fresh, fixed synthetic seed (20261125), six training matrices and twelve held-out matrices per topology. The primary candidate is fixed before the new holdout is evaluated.

For a feasible fixed matching x, let r_s(x)=F_s(x)/F_s(0) for training scenario s. Define
J_lambda(x) = (1-lambda) mean_s r_s(x) + lambda min_s r_s(x).
Compare lambda=0 (mean-only), 0.5 (primary balanced objective), and 1 (min-only). All use the same matching budget, common feasible starting candidates (zero, edge betweenness and first-snapshot), scenario-specific dual cuts, singleton cut initialization and nominal solver time budget. These shared candidate evaluations use training data only. Training guarantees do not imply holdout guarantees.

### Mathematical formulation
The master has binary upgrade vector x and continuous throughput variables y_s and z, with z <= y_s for every s. For each evaluated x0 and each scenario, duality supplies the valid bound y_s <= r_s(x0)+g_s dot (x-x0). Maximize (1-lambda) mean(y_s)+lambda z. Singleton inequalities strengthen each scenario's envelope. Because the objective is nondecreasing in every y_s, every master upper bound is valid for the true training objective in exact arithmetic. Each fully evaluated matching provides a lower bound. With exact solves, the finite set of matchings and tight cuts at previously queried matchings imply finite convergence. The implementation reports numerical gaps and time limits, not exact rational proofs. Small examples independently enumerate all feasible matchings and compare objective values for each lambda.

The balanced solver is guaranteed to return a training objective at least as good as its shared evaluated starting candidates (up to numerical tolerance), but not necessarily to beat them on held-out traffic. This is a standard risk-sensitive scalarization and multi-cut decomposition, not an established novel algorithm. Report all paired wins, ties and losses, average gain, average per-topology minimum gain, balanced gain, gaps and timing. No post-hoc selection of lambda on the fresh synthetic holdout. Historical matrices are exploratory because earlier snapshots have already been inspected; assumed capacities remain explicitly labeled. Topology files may represent overlapping networks, so descriptive averages are not independent statistical estimates.

Run: python balanced_benchmark.py --mode self-test
Synthetic shard: python balanced_benchmark.py --mode zoo --train 6 --test 12 --seed 20261125 --limit 30 --offset 0 --seconds 15 --output results/zoo-0.json
Historical pilot: python balanced_benchmark.py --mode traces --train 3 --test 6 --seconds 45 --output results/historical.json

The separate Balanced network research workflow runs four synthetic shards and a historical pilot, then aggregates their JSON results. It targets 120 topology instances and 2,160 new synthetic traffic matrices; a successful workflow and inspected artifacts are required before claiming any new gains.


## Phase 5: persistent bottlenecks and a full-day Germany50 replay

The Phase 4 balanced objective achieved 9.238% mean gain on 1,440 synthetic holdout matrices, versus 9.255% for mean-only and 8.836% for edge betweenness; the worst-case gains were negligible. All seven workflow jobs succeeded (run 45543005831). The Germany50 historical pilot achieved 25% on six holdout snapshots under modeled capacity. It is not an operational performance claim.

### Mathematical screening guarantee
Let F_s(x) denote maximum concurrent splittable flow for scenario s and B_s=F_s(0)>0. At zero upgrades, an optimal capacity dual gives a nonnegative normalized vector g_s with sum(g_s)=1 (in exact arithmetic). This follows from strong duality: the baseline capacity vector is the only nonzero right-hand side in the routing LP, so its inner product with the capacity prices equals B_s. Doubling selected edges adds capacity c_e x_e, hence
0 <= F_s(x)/B_s - 1 <= g_s dot x.
If the matching misses the support of g_s, the gain is exactly zero. Conversely, intersecting the support does NOT imply a positive gain. Degenerate duals may provide different supports, and a single support per scenario is not a uniquely identified physical bottleneck.

For a target gain tau>0 introduce binary z_s and impose tau z_s <= g_s dot x, in addition to matching and upgrade-budget constraints. Any actual threshold success implies eligibility in this master, so maximizing sum(z_s) provides an upper bound on the number of scenarios that ANY allowed plan can improve by tau. The implementation uses a tie-breaking bonus in [0,0.5] on mean envelope gain; a valid objective upper bound rounded down bounds the integer coverage count. The evaluated candidate's actual coverage is a lower bound on achievable coverage. The numerical bounds are not exact rational certificates and need independent verification for publication. A retrospective holdout bound describes the limits of the model, and its optimizing placement is discarded.

### Faster routing formulation
Aggregate all commodities sharing the same source into a single flow with one supply and multiple specified sink balances. Summing ordinary commodity flows proves one direction; decomposing nonnegative aggregated flows into source-to-sink paths and discarding cycles proves the converse. This preserves the exact concurrent-flow optimum in the shared-capacity splittable routing model and reduces the number of commodities from the number of positive ordered demands to the number of active source nodes. Small-graph enumeration checks compare the aggregated LP with the original LP across feasible matchings. This is an established modeling technique, not a new theoretical result.

### Predeclared experiments
- Germany50: scan all 288 expected five-minute matrix files, first 144 for plan selection, last 144 for holdout; record any empty matrices and exact filenames and source hash. Previously inspected pilot snapshots overlap, so this is exploratory historical validation, not a wholly untouched holdout. Capacity policy remains the published-module hypothetical baseline where installed capacity is unavailable.
- Synthetic: 30 deterministically selected eligible Topology Zoo files, fresh fixed random seed 20261126, 24 training and 24 held-out weighted traffic matrices per topology. Historical and synthetic results are kept separate.
- Compare no upgrades, edge betweenness, first-snapshot, dual-frequency matching, dual-mean matching, dual-coverage matching, and balanced optimization on six deterministic training snapshots. Select one persistent variant based ONLY on training threshold coverage, then mean gain; evaluate all variants on the holdout without tuning. Report actual threshold coverage, mean, median, minimum, positive fraction, dual envelope and numerical coverage bound. Upgrade budget remains floor(m/20), and threshold is predeclared at 5%.
- The 30-topology benchmark is an initial fresh-traffic pilot, not a test on entirely unseen topology families. The baseline methods have different selection costs; runtime comparisons must include full preprocessing and routing solves. No guarantee of superiority or novelty is claimed.

Run locally: python persistent_benchmark.py --mode self-test
Historical: python persistent_benchmark.py --mode historical --seconds 60 --output results/germany50-full.json
Synthetic: python persistent_benchmark.py --mode synthetic --limit 30 --seconds 15 --output results/persistent-synthetic.json


## Phase 6: capacity augmentation beyond a matching

The full-day Germany50 replay (run 45544465195) evaluated all 288 matrices. Under modeled capacities, the Frankfurt–Koblenz upgrade achieved a mean 24.9887% and minimum 23.3773% gain over the 144 held-out snapshots. The baseline training dual ranking assigns 0.25 average normalized shadow weight to each of Frankfurt's four incident links. Thus a per-router upgrade limit d implies an average training gain upper bound of min(d/4,1) for this particular numerical dual vector. It does not imply attainability. The existing data are exploratory and do not establish current physical network performance.

### New optimization and cost model

Let x_e be binary link doubling decisions, z_v binary router activation, d the maximum number of upgrades incident to a router, and K=floor(m/20). Impose sum(x)<=K, z_v <= sum_{e incident to v} x_e <= d z_v. Optionally impose sum(c_e^cost x_e)+f sum(z_v)<=B, and sum_{e incident to v} c_e x_e <= h sum_{e incident to v} c_e. The latter is a sensitivity scenario for added incident capacity, not a verified router switching-fabric or port constraint.

Compare d=1,2,3,4 under the same four-link count budget for Germany50. Additionally compare d=1 and d=4 under the same normalized capacity-cost budget; test a 50% incident-capacity headroom constraint. If every link has a positive published SNDlib candidate module price covering the modeled extra capacity, also compare a module-price proxy. Normalize prices by their median; router activation cost is 0.25 normalized units, and the illustrative total budget is K+2K*0.25. Module prices are historical benchmark input costs, not vendor quotations. The cheapest module large enough to cover the modeled doubling may be larger than the incremental capacity credited in the routing model. If prices are missing, omit that entire cost family and report coverage. The capacity-weighted proxy and setup/headroom assumptions are not measured deployment economics.

### Bound and evaluation

For the training mean normalized gain A(x), at an evaluated plan x0, the average normalized capacity-price vector g gives A(x)<=A(x0)+g dot (x-x0). The master maximizes an upper envelope over the chosen policy constraints; each actual training replay gives a feasible lower bound. The initial cut comes from all baseline training LPs. Numerical gaps and evaluated plans are retained; the iteration cap means a nonclosed gap is not a proof of optimality. The objective is mean training throughput, not minimum cost, so selected plans need not be cheapest among tied solutions.

All selections use the first 144 chronological snapshots only. Evaluate the fixed placements on the final 144, report mean, minimum, median and fraction above 5%, as well as the number of affected routers and modeled capacity/cost. Compare each policy at the same budget where applicable. This is a sensitivity experiment on a previously inspected historical dataset; no independent operational validation is claimed. Independent small-star enumeration checks the degree, activation, cost and headroom constraints.

Run: python cost_relaxation.py --mode self-test
Historical: python cost_relaxation.py --mode historical --seconds 60 --iterations 6 --output results/cost-relaxation.json


## Phase 7: generalizing concentrated upgrades

Germany50 cost-relaxation run 45546688019 reported 85.2913% mean held-out gain from doubling four Frankfurt links (4/88), with minimum 24.2149% and mean training optimum 90.2890% under the count policy. The 50% modeled incident-capacity headroom case yielded 49.4459% mean held-out gain. All these results are conditional on hypothetical capacities and the historical benchmark.

### Structural theorem (tree case) and necessary condition (general case)

For a tree, let L_e be the total demand crossing the partition created by deleting edge e. There is exactly one simple route for each demand. Hence the maximum concurrent throughput multiplier F(x)=min_{e:L_e>0} c_e(1+x_e)/L_e for binary doubling x. Write B=F(0)>0 and let a target gain tau>0 be fixed. Define R_tau={e:c_e < (1+tau)B L_e}. A target gain of at least tau is possible under a specified upgrade policy if and only if every edge in R_tau can be upgraded within the policy and 2c_e >= (1+tau)B L_e for all e. Indeed, each edge has to carry its unique-path load; these inequalities are necessary and sufficient. Under monotone count/cost/degree/headroom restrictions, checking the mandatory set alone suffices. This characterization follows directly from unique-path routing; novelty is not claimed.

For any connected simple undirected graph the same inequalities on bridges are necessary, but generally not sufficient. In particular, two adjacent mandatory bridges make a target impossible under a matching policy. Combine these constraints with baseline capacity-price envelope g_s dot x >= tau z_s to bound the total number of scenarios attaining a threshold: require x_e>=z_s for each mandatory bridge, and force z_s=0 if even doubling a bridge is insufficient. A numerical upper bound on this integer maximization bounds actual threshold coverage. A serial two-edge equal-capacity path proves that unrestricted augmentation can yield positive throughput gain when matching-constrained augmentation yields zero, so there is no universal finite multiplicative bound on the ratio of gains (with a zero denominator, interpret this as failure of any finite factor inequality). This is about gains, not absolute throughput ratios, which are bounded by two in the doubling model.

For nested policies, if L_r,U_r are feasible lower and valid upper bounds on training optimum for relaxed policy r and L_m,U_m for matching m, the true advantage is in [max(0,L_r-U_m), U_r-L_m]. The implementation checks cross-policy feasible placements against all upper bounds; strict positive lower bounds certify a training advantage numerically. None guarantees superiority on unseen traffic.

### Controlled benchmark

Run on the same deterministic sampling of 120 eligible topology files, using fresh synthetic traffic seed 20261127, 12 training and 24 held-out matrices per topology. Two distinct capacity regimes (unit and synthetic heterogeneous 1/2/4 with fixed seed 1701) give 240 topology-capacity instances; the traffic for each pair is identical. Report separately. The topology files can be overlapping snapshots, and the capacities/costs do not represent actual operator measurements.

Compare equal count budgets floor(m/20): matching vs unrestricted; and equal cost and count budgets: matching, degree-two, unrestricted, and unrestricted with hypothetical 50% incident-capacity headroom. Normalized capacity cost c_e/median(c), router setup 0.25, total cost budget 1.5K. For unit capacities this cost budget does not independently bind any plan satisfying K, so it serves as a control; the heterogeneous regime makes the cost constraint potentially binding. All decisions use training data only, and prior feasible plans may warm-start subsequent policies. The iteration cap is six and the time limit is per master/LP call (not an equal total wall-clock budget). Holdout upper-bound optimization is retrospective and discards the placement. Report all training gaps, paired held-out differences, threshold success rates, number of links/routers and cost/capacity accounting. No significance testing treating topology files as independent.

Verification: enumerate subsets on a heterogeneous tree to check the exact threshold characterization; check a two-edge path where relaxing the matching strictly improves throughput. Code: concentration_benchmark.py; paired summary: summarize_concentration.py. Run workflow Concentrated upgrade generalization; aggregate requires all eight 30-file shards and matching topology sets. Targets: 5,760 held-out evaluations at the topology-capacity-scenario level, not 5,760 independent network measurements.


## Phase 8: certificate-guided adaptive concentration

Previous controlled run 45548942168 found 19.03% average held-out gain in the heterogeneous modeled capacity regime, compared with 18.26% for matching-constrained cost-aware upgrades. The unrestricted method won 37, tied 191 and lost 12 paired topology-capacity instances across both regimes. This motivates selective concentration rather than treating all networks alike.

### Partition and stopping certificate

For the unrestricted feasible binary upgrade family P under a fixed count and cost budget, define M={x in P: every vertex has at most one upgraded incident edge} and C=P minus M. These are disjoint and exhaustive. Encode C by binary witness w_v, sum(w)>=1, and sum_{e incident to v} x_e>=2w_v. This avoids assuming that a candidate with positive shadow prices at one router necessarily benefits from concentration.

Each evaluated placement gives a valid scenario-specific supporting upper bound from the routing LP dual. Maximizing the mean upper envelope over M and C separately yields upper bounds U_M and U_C. Evaluated feasible placements yield lower bounds L_M and L_C. Thus max(L_M,L_C) <= OPT(P) <= max(U_M,U_C), treating an empty C as upper bound zero for this nonnegative objective. If max(U_M,U_C)-L_M <= epsilon, a matching placement is epsilon-optimal over P. If L_C > U_M+epsilon, the optimal concentration advantage exceeds epsilon. More generally, any class whose upper bound is no greater than the best evaluated global gain plus epsilon need not be searched further for epsilon-optimality. This is a direct partition/upper-bound argument, not an established novel theorem. With exact solves, repeated evaluation yields tight cuts at each queried plan, so positive-gap master iterations cannot repeatedly return an already evaluated plan. Finite feasible space gives convergence in the absence of iteration/time caps. In numerical runs we retain open gaps and use epsilon=0.005 (0.5 percentage point).

### Compared methods and data

Compare aggregate-cut unrestricted Benders, scenario-specific multicut unrestricted Benders, and adaptive partition multicut with shared cuts across classes. All use the same feasible unrestricted cost policy, baseline LPs, and independent caches. The adaptive version first queries a matching candidate, then considers only classes whose upper bounds exceed the incumbent plus epsilon. Each method gets at most 12 distinct nonzero plan evaluations; time limits are per LP/master call, not equal wall-clock limits. Include baseline scans in total selection times; report paired speedups, LP calls, master calls, objective gaps and held-out quality together. Do not interpret speed alone as superiority when gaps differ.

Use the same 120 topology-file sampling rule, fresh traffic seed 20261128, 12 training and 24 held-out scenarios per instance, unit and heterogeneous modeled capacities. No plan selection on holdout. Also screen the official SNDlib static XML archive for networks of 20–120 edges and at most 1,000 positive demands. Distinguish entirely positive published preinstalled capacities from configurations requiring hypothetical candidate modules. Static published-demand instances have no independent held-out matrix; do not combine their training results with synthetic holdout results. No claim that published capacities are verified current operational capacities. All regimes use the same modeled capacity-cost proxy and router activation assumptions.

Relevant prior work: Benders decomposition approach for the robust network design problem with flow bifurcations (https://doi.org/10.1002/net.21486); Capacity and flow assignment of data networks by generalized Benders decomposition (https://doi.org/10.1023/A%3A1011280023547); Decomposition Methods for Network Design (https://www.sciencedirect.com/science/article/pii/S1877042811013863). The baseline implementations are standard formulations adapted to this objective, not reproductions of these papers' full algorithms. A direct state-of-the-art comparison and novelty review remain necessary.

Run `python adaptive_benchmark.py --mode self-test` for independent small-case enumeration checks. Workflow: Adaptive bottleneck classification; aggregate report: adaptive-summary.json. Targets: 240 modeled synthetic instances, 5,760 held-out scenario evaluations, plus eligible published static instances.


## Phase 9: dual-guided edge screening

Corrected adaptive run 45550572453 completed 258 instances. Adaptive partition classified 196 as matching-sufficient within 0.5 percentage points, 33 as having a certified concentration advantage and 29 as unresolved. Standard multicut was faster overall on the 240 synthetic instances (185.15 s versus 247.71 s including baseline scan), with the same 236/240 training gap closures. Eighteen published-demand instances all used hypothetical module capacities. These findings motivate removing upgrade variables rather than solving two integer masters.

### Screening certificate

Let L be the best feasible evaluated mean training gain and epsilon=0.005. For each active upgrade edge e, compute an upper bound U_e on the objective of all currently active-region feasible placements with x_e=1. If U_e <= L+epsilon, those placements cannot improve the incumbent by more than epsilon. Discard e from the upgrade variables only; routing still uses every physical edge. For simultaneous or sequential deletions, retain D, the largest conditional upper bound of any deleted region. Every original placement either survives all deletions or belongs to a first-deletion region. If U_A is a valid upper bound for the surviving region, OPT <= max(U_A,D). Keep the incumbent even if it contains discarded variables. Stop only if the global bound is within tolerance or a stated computation cap is reached; report open gaps. Exact arithmetic yields the stated guarantee; this implementation uses numerical routing duals, conservative padding, consistency checks, and withholding of contradicted certificates.

Two conditional bounds are implemented:
1. For each scenario and dual cut y_s <= b_s + g_s dot x with nonnegative slopes, force edge e and add at most K-1 largest remaining coefficients (relax cost/router restrictions). Minimize the resulting upper bound over available cuts for that scenario, then average over scenarios. For numerical safety the implementation raises negative slopes/intercepts to zero and adds an outward margin.
2. For the current multicut LP relaxation, write max c dot z subject to A z <= b and 0 <= z <= 1. For any nonnegative multiplier lambda, let r=c-A^T lambda. Then c dot z <= lambda dot b + sum_j max(r_j,0). Forcing an upgrade coordinate z_e=1 tightens the upper bound by min(r_e,0). This follows by maximizing each residual term over its box. Use LP-derived multipliers, explicitly recompute the bound rather than relying on an LP objective difference, and pad numerically. Combine both bounds by taking their minimum. This is related to classical reduced-cost fixing, not a claim of a new general optimization theorem.

### Controlled comparison

Use identical multicut drivers, private evaluation caches and 12 nonzero-plan evaluation caps: (a) no screening, (b) cardinality screening, (c) hybrid cardinality and LP-dual screening. Physically remove discarded upgrade columns and router activation columns with no remaining incident upgrade candidate. The routing LP is unchanged. Time limits are per solve, and screening overhead is included in total selection time. Report actual integer-master sizes (first and average) separately from final eliminated fraction; deleting all remaining candidates after finding an optimal solution does not by itself demonstrate speedup. Report paired time ratios, training gaps, withheld certificates, and holdout performance. No algorithmic superiority is assumed. Small graphs enumerate all feasible upgrade subsets and check forced-edge bounds and final global certificates, including a case with irrelevant upgrade candidates.

Same 120 topology-file sampling rule, fresh synthetic traffic seed 20261129, 12 training and 24 holdout matrices, unit and heterogeneous modeled capacities. Also use the same prespecified SNDlib static size filter; report published preinstalled and hypothetical module capacity regimes separately. No static holdout traffic is invented. Cost assumptions are unchanged. The target is 240 synthetic instances and 5,760 topology-capacity-scenario holdout evaluations, plus eligible static benchmarks. A full literature review, comparisons against published complete implementations, exact certificates and actual operating-network validation remain open.

Relevant literature: reduced-cost fixing and strong fixing, The Dual-Path Fixing Strategy and Its Application to the Set-Covering Problem (https://drops.dagstuhl.de/storage/00lipics/lipics-vol371-sea2026/html/LIPIcs.SEA.2026.28/LIPIcs.SEA.2026.28.html); Decomposition Methods for Network Design (https://www.sciencedirect.com/science/article/pii/S1877042811013863).

Run python screening_benchmark.py --mode self-test. Workflow: Dual-guided edge screening; aggregate artifact: screening-summary. Targets of 80–95% candidate elimination and 10x speedup are hypotheses, not results.


## Phase 10: certificate-guided routing calculation elimination

Phase 9 run 45551778770 achieved 81.08% final mean candidate elimination in the heterogeneous regime, but hybrid screening was slightly slower overall and the routing LP count was almost unchanged. This phase isolates routing reuse, without edge screening.

### Mathematical bounds and skipping rule

For fixed scenario s and old feasible multicommodity flow with throughput multiplier r and total edge loads l_e, let c'_e be capacities at a proposed new placement. Scaling every flow and demand multiplier by alpha=min_{e:l_e>0} c'_e/l_e yields another feasible flow, so alpha*r is a lower bound on the new optimum. Take the maximum over stored flows (and baseline monotonicity). Store normalized loads and rates, rather than full flow vectors, for reuse within the same scenario. Feasibility is conditional on the original flow's validity. Numerical routing solutions are checked for residuals and bounds are padded; these are not exact rational certificates.

At each previous routing optimum, the capacity dual supplies a global supporting upper bound. Take the minimum of these cuts and the physical doubling ceiling. Combining these gives per-scenario lower and upper bounds. Their means bound the training objective. When the mean gap is at most epsilon/4 (0.125 percentage points), skip remaining routing solves. The lazy variant can additionally stop evaluating a candidate if its mean upper bound is at most the incumbent plus epsilon/4. Unsolved scenario rows retain existing dual cuts; no approximate primal value is misrepresented as a new supporting cut. The master retains a global upper bound, and a feasible incumbent retains its lower bound. The algorithm can stop with an open gap on repeat candidates or computation caps; it does not promise finite epsilon convergence under the caps. Full training replay checks the chosen plan and its bounds. Contradictions with global bounds result in withheld certificates, not an assertion of success.

Related work includes Benders decomposition with adaptive oracles for large scale optimization (https://doi.org/10.1007/s12532-020-00197-0) and Adaptive Subproblem Selection in Benders Decomposition for Survivable Network Design Problems (https://drops.dagstuhl.de/storage/00lipics/lipics-vol371-sea2026/html/LIPIcs.SEA.2026.17/LIPIcs.SEA.2026.17.html). Reusing primal/dual bounds and adaptive subproblem selection are established research directions. Our specialization and experimental comparisons do not establish novelty or superiority over published full implementations.

### Paired evaluation

Same unrestricted cost-aware policy and multicut master in three modes: full routing solves for every queried plan, reuse with gap-driven scenario selection, and lazy reuse with early candidate rejection. Private caches, identical baseline scan and nominal per-solve limits, at most 12 candidate queries each, 0.5 percentage point global tolerance. The full mode still computes dual bounds for consistency checks but does not compute primal reuse bounds or skip candidate solves. Include common model preparation/baseline scans, all master and certificate overhead, and full final training replay in reported verified runtime. Report candidate skips separately from actual total routing-call reduction including scans and replays, and separate held-out quality. Holdout replay is evaluation, not optimization timing. Time limits are per solve, not identical global wall-time caps.

Use the same 120 topology-file sampling rule, fresh fixed synthetic seed 20261130, 24 training and 24 held-out matrices per topology and capacity regime (unit/heterogeneous). Re-run the 288 historical Germany50 snapshots (144 training/144 holdout) as a heavier exploratory workload with hypothetical module capacities; previously inspected data are not a fresh holdout. Target 240 synthetic instances plus one historical instance. Small-graph enumeration checks reused bounds against full LP solutions and verifies all modes, including an instance where routing solves can be skipped entirely. No new operational capacity evidence is claimed.

Run python routing_reuse.py --mode self-test. Workflow: Certificate-guided routing reuse. Aggregate artifact: reuse-summary.


## Phase 11: cross-scenario routing certificate transfer

Phase 10 (run 45552766311) demonstrated 1.58x aggregate synthetic verified speedup for within-scenario reuse, with 6,842 candidate routing solves versus 24,672 for full evaluation, and identical held-out performance on 239/240 instances. Germany50 saw no acceleration. This phase compares cross-scenario transfer against the within-scenario reuse method.

### Mathematical basis

On a fixed undirected capacitated graph with splittable concurrent routing, if a D_i <= D_j <= b D_i componentwise, for 0<a<=b, then F_i(x)/b <= F_j(x) <= F_i(x)/a. Routing feasibility is monotone in the demand vector; proportional traffic gives an exact scaling law.

More generally, choose a distribution over simple paths for each ordered demand pair p, with total path weight 1. Let T_{pe} denote the weighted edge incidence of these paths. For any target traffic vector d, its per-unit throughput load is L_e=sum_p d_p T_{pe}. Thus min_{e:L_e>0} c_e(1+x_e)/L_e is a feasible lower bound on F_d(x). The path template is independent of the target demand weights and can handle different support if routes exist for every target pair. Extract paths from source-aggregated LP flows for solved traffic; assign missing pairs or undecomposed residuals to a valid shortest-hop path. Every path distribution is feasible by construction, even if it differs from the optimal LP flow. Store the resulting normalized load matrices. Take the best feasible scaling over all templates.

For any nonnegative edge lengths ell, each routed unit from u to v traverses length at least dist_ell(u,v). Total length consumed cannot exceed sum_e c_e(1+x_e) ell_e. Consequently F_d(x) <= [sum_e c_e(1+x_e) ell_e]/[sum_{uv} d_uv dist_ell(u,v)] when the denominator is positive. Derive lengths from solved capacity dual prices, compute shortest paths afresh on the common graph and transfer this upper bound to every target matrix. If a denominator vanishes, discard that certificate for the target. This is the standard metric bound for multicommodity flow. Primal path routing and metric dual bounds are established techniques; the particular cross-traffic reuse strategy is an experimental combination, not an established novel algorithm.

For an unknown baseline B_s=F_s(0), maintain positive bounds b_s^- <= B_s <= b_s^+. A transferred primal/dual pair l_s,u_s for candidate throughput gives normalized gain bounds max(0,l_s/b_s^+-1) and min(1,u_s/b_s^--1). Skip a baseline solve during selection if b_s^+/b_s^- -1 <= 0.00025; otherwise solve a scenario with the largest relative gap and update every target. Master upper cuts use b_s^- in the denominator, so they remain upper bounds even when the baseline is uncertain. Candidate selection solves scenarios with the largest normalized gap until the average gap is at most 0.00125 or all are solved. The global numerical stopping tolerance remains 0.005. Repeated plans or iteration caps may leave open gaps. LP residual checks and outward padding are numerical precautions, not rigorous rational arithmetic. Verify all selected plans and any skipped baselines with full LP replay; report contradictions and withhold certificates if needed. Skipped baseline LPs are deferred to verification rather than permanently avoided in verified timings.

### Comparison protocol

Private setup, caches and end-to-end measured selection and verified runtime for within-scenario reuse versus cross-scenario reuse. Both use the same capacity and cost policy, 12 candidate-query cap and nominal per-solve limits. Count all baseline, candidate and final replay LPs. Include template extraction, distance computation and model construction in timing. Evaluate the held-out matrices only after selection. Compare within each workload rather than combining results: 120 previously selected topology files, two modeled capacity regimes, and two synthetic traffic regimes (independent pairs, fixed-pair correlated weights varying by +/-20%). Each has 24 training and 24 held-out scenarios, with fresh seeds 20261201 and 20261202. This gives 480 synthetic topology-capacity-workload configurations and 11,520 held-out synthetic evaluations. Also replay all 288 Germany50 historical matrices with previously inspected data and hypothetical module capacities. These workloads do not establish operational network performance. No actual speedup or superiority is claimed before inspecting the new results.

Code: cross_scenario.py. Verification enumerates small graph upgrade subsets and compares transferred primal and dual bounds to full LP values, including proportional traffic and changed support. Workflow: Cross-scenario certificate transfer. Aggregate artifact: cross-summary.


## Phase 12: persistent LP models and simplex basis reuse

Cross-scenario run 45556137057 gave only a modest aggregate synthetic speedup and slowed the Germany50 replay. This phase uses native HiGHS 1.11.0 to retain an LP model and restore simplex bases rather than transferring paths between scenarios.

For the source-aggregated routing LP, the flow incidence blocks depend on the graph and number of active sources. The throughput multiplier column and capacity RHS encode the specific traffic and scaling. Within a fixed graph, persistent models are pooled by the number of source blocks; changes are applied to the multiplier column and capacity row bounds. This remains the same routing problem, even if the active source identities change. With a fixed constraint matrix and objective, changing only the RHS preserves dual feasibility of an optimal basis in exact arithmetic, though primal feasibility may be lost. Updating traffic coefficients does not have this guarantee. Prefer a saved basis for the same scenario, otherwise try the previous compatible model's basis. If a native incremental solve or its numerical checks fail, rebuild cold and, if necessary, use the existing SciPy solver. Count fallbacks and basis rejections. Validate primal residuals, signs, reduced costs and primal-dual gap against the requested model. No exact rational certificates are claimed.

Compare five methods: existing SciPy within-scenario reuse, fresh native cold models plus screening, persistent native models with solver state reset plus screening, persistent native models with basis reuse plus screening, and persistent warm models without certificate screening. Native cold/reset/warm use the same HiGHS settings; the SciPy comparator uses its existing settings and bundled HiGHS version, so native cold versus native warm isolates the implementation differences more closely. All methods use the existing multicut master and 12 candidate-query cap, 0.5 percentage point stopping tolerance, private setup/cache, and independent cold final training replay. All solve every baseline scenario, so baseline iteration counts and scan timing compare a common workload even when candidate queries differ. Include setup, updates, basis restoration, fallback costs, certificate processing and final replay in total measured time. Report total and baseline simplex iterations, model builds, updates, basis attempts, rejections, fallback counts, training gaps and held-out gains. Rotate method order deterministically by edge count; single timings per instance remain noisy. Time limits are per solve, not equal global wall-clock limits. The persistent mode still constructs SourceProblem matrices for checking and reference evaluation; it does not eliminate all model construction.

Synthetic workload: same 120 sampled topology files, unit and heterogeneous modeled capacities, independent and fixed-pair correlated traffic, 24 training and 24 holdout scenarios with fresh seeds 20261203 and 20261204. This yields 480 synthetic configurations. Reuse the full 288 Germany50 matrices in a separate exploratory historical replay with hypothetical module capacities. Compare within each regime; no holdout information enters plan selection. Independent small-graph tests compare native cold/reset/warm LP values and duals with SciPy under changing traffic support, source counts and capacity vectors, then compare all five optimizer modes with exhaustive enumeration. The workflow gates benchmarks on these checks. Results and superiority are not assumed. Warm-start simplex and persistent LP models are established techniques, not a claim of a new mathematical algorithm.

References: https://ergo-code.github.io/HiGHS/stable/interfaces/python/example-py/ ; https://pypi.org/project/highspy/1.11.0/ . Run python incremental_routing.py --mode self-test after installing requirements.txt and highspy==1.11.0. Workflow: Incremental routing optimization. Aggregate artifact: incremental-summary.
