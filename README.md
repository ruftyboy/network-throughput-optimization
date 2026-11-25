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
