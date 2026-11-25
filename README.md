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
