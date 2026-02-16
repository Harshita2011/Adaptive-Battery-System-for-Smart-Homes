# Scenario Evaluation Report

## KPI Summary

```text
               scenario  samples  peak_temp_c  soh_start  soh_end  soh_degradation_rate_per_hour  protection_events  energy_cost_index  load_curtailment_count  live_to_sim_fallback_events  peak_temp_reduction_vs_normal_pct
     heatwave_high_temp       40         60.4     0.9981   0.9957                          0.216                  0               56.0                       0                            0                           -41.7840
missing_stale_telemetry       20         44.0     0.9850   0.9970                         -2.160                  0              104.0                       0                            8                            -3.2864
            normal_load       40         42.6     0.9974   0.9982                         -0.072                  0              208.0                       0                            0                             0.0000
           sudden_spike       52         54.0     0.9850   0.9850                          0.000                  1              224.8                       0                            0                           -26.7606
```

## Interpretation Guide

- `peak_temp_reduction_vs_normal_pct`: higher is better.
- `soh_degradation_rate_per_hour`: lower is better.
- `protection_events`: fewer entries usually indicate better stability.
- `energy_cost_index`: lower is better if safety remains acceptable.
- `load_curtailment_count`: lower means less user comfort impact.
- `live_to_sim_fallback_events`: confirms stale telemetry fallback behavior.