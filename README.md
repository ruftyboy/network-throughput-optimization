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
