# Pilot Timing Report (C=1 evaluation, batch 1024)

Started: 2026-06-23T22:38:32.424175+00:00  
Updated: 2026-06-24T00:51:12.516421+00:00

| phase | run | algo | problem | conc | steps | wall (min) | s/step | exit | best R2 |
|---|---|---|---|---|---|---|---|---|---|
| S | solo_pure_ppo_nguyen5 | pure_ppo | nguyen_5 | 1 | 500 | 132.7 | 15.92 | 0 | 0.35321265330264884 |

## Projections
```json
{
  "efficiency": {
    "concurrency_1": {
      "mean_sec_per_step": 15.92,
      "slowdown_vs_solo": 1.0,
      "seedruns_200steps_per_hour": 1.13,
      "usd_per_seedrun": 0.195
    }
  },
  "usd_per_hour": 0.22,
  "assumptions": {
    "phase2": "100 seed-runs x 200 steps",
    "phase42_optB": "150 seed-runs x 150 steps",
    "overhead_factor": 1.15
  },
  "best_config": "concurrency_1",
  "phase2_hours": 101.8,
  "phase2_usd": 22.39,
  "phase42_optB_hours": 114.4,
  "phase42_optB_usd": 25.17
}
```
