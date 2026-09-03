# DP-MMFL Experiments

## Experiment Naming

Format:
`EXP_<phase>_<description>`

Examples:
- `EXP_03_CENTRALIZED_IMAGE`
- `EXP_03_CENTRALIZED_MULTIMODAL`
- `EXP_04_FEDAVG`
- `EXP_05_FEDAVG_DP`
- `EXP_06_ADAPTIVE_DP`

## Results Policy

No experimental result may be entered unless the experiment has actually been executed.

Each experiment must record:
- Date
- Git commit
- Random seed
- Dataset version
- Dataset split
- Model
- Optimizer
- Learning rate
- Batch size
- Number of epochs
- Number of clients
- Federated rounds
- DP parameters
- Aggregation method
- Metricss
