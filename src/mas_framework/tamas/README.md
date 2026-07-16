# TAMAS Swarm Experiment Subset

This directory contains the TAMAS source subset used by the Swarm experiment matrix.

Selection rule:

- Attack types: Byzantine, Colluding, Contradicting
- Domains: Education, Finance, Healthcare, Legal, News
- Byzantine files are clean base cases; experiments dynamically inject 0-3 Byzantine agents
- Colluding and Contradicting include 0-node and 2-node variants
- Tasks: 2 tasks per domain/variant

Total source tasks: 50.

Files are grouped by attack type so `load_tamas_dataset()` can infer the attack type from the parent directory. See `manifest.json` for paths and counts.
