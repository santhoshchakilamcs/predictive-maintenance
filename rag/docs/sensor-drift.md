# Sensor Drift and Bias (SAMPLE document written for this demo project. Not an OEM manual.)

## Symptoms
One sensor steps or drifts while all correlated sensors stay normal. No matching change in performance parameters (thrust, fuel flow, temperatures). The anomaly appears suddenly rather than growing over cycles.

## Likely causes
Sensor bias, wiring or connector degradation, or a data-acquisition channel fault. True engine degradation moves several sensors together; a lone-sensor move is usually instrumentation.

## Recommended actions
1. Cross-check the suspect sensor against redundant and correlated channels.
2. Inspect wiring and connectors; swap the channel if the fault follows the hardware.
3. Recalibrate or replace the sensor, then confirm readings return to the fleet baseline before clearing the alert.
