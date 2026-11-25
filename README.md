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
