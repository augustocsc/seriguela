# Pilot Timing Report (C=1 evaluation, batch 1024)

Started: 2026-06-10T12:47:36.332130+00:00  
Updated: 2026-06-10T13:02:56.398721+00:00

| phase | run | algo | problem | conc | steps | wall (min) | s/step | exit | best R2 |
|---|---|---|---|---|---|---|---|---|---|
| S | solo_pure_ppo_nguyen5 | pure_ppo | nguyen_5 | 1 | 500 | 2.0 | 0.24 | 0 | 0.0 |
| C2 | c2_pure_grpo_nguyen5 | pure_grpo | nguyen_5 | 2 | 500 | 1.5 | 0.18 | 0 | 0.0 |
| C2 | c2_bon_ppo_nguyen5 | bon_ppo | nguyen_5 | 2 | 500 | 2.3 | 0.28 | 0 | 0.0 |
| C3 | c3_pure_ppo_a | pure_ppo | nguyen_5 | 3 | 150 | 2.2 | 0.87 | 0 | 0.3661034110570037 |
| C3 | c3_pure_ppo_b | pure_ppo | nguyen_5 | 3 | 150 | 2.0 | 0.8 | 0 | 0.0 |
| C3 | c3_pure_ppo_c | pure_ppo | nguyen_5 | 3 | 150 | 1.8 | 0.73 | 0 | 0.0 |
| P | probe_best_of_n | best_of_n | nguyen_5 | 1 | 50 | 6.2 | 7.4 | 0 | 0.385130723675968 |
| P | probe_feynman_ppo | pure_ppo | feynman_I_12_2 | 1 | 200 | 2.0 | 0.6 | 0 | 0.06579480629269874 |

## Projections
```json
{
  "efficiency": {
    "concurrency_1": {
      "mean_sec_per_step": 0.42,
      "slowdown_vs_solo": 1.0,
      "seedruns_200steps_per_hour": 42.86,
      "usd_per_seedrun": 0.005
    },
    "concurrency_2": {
      "mean_sec_per_step": 0.23,
      "slowdown_vs_solo": 0.55,
      "seedruns_200steps_per_hour": 156.52,
      "usd_per_seedrun": 0.001
    },
    "concurrency_3": {
      "mean_sec_per_step": 0.8,
      "slowdown_vs_solo": 1.9,
      "seedruns_200steps_per_hour": 67.5,
      "usd_per_seedrun": 0.003
    }
  },
  "usd_per_hour": 0.22,
  "assumptions": {
    "phase2": "100 seed-runs x 200 steps",
    "phase42_optB": "150 seed-runs x 150 steps",
    "overhead_factor": 1.15
  },
  "best_config": "concurrency_2",
  "phase2_hours": 0.7,
  "phase2_usd": 0.16,
  "phase42_optB_hours": 2.2,
  "phase42_optB_usd": 0.47
}
```
