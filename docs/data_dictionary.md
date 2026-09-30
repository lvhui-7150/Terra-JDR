# Data Dictionary

## Processed Inputs

### `data/processed/transport.json`

- `nodes`: dispatch center and service areas with coordinates and elevation.
- `types`: aircraft payload, volume, speed, range, energy, reserve, and
  handling parameters.
- `boxes`: demand identifiers, service areas, mass, volume, deadlines, and
  priorities.
- `fleet`: delivery UAV identifiers and aircraft types.
- `battery_inventory`: shared-battery counts and full-charge times.

### `data/processed/dem.tif`

Digital elevation model used for terrain profiles, cruise altitude, LOS
checks, and communication audits.

### `data/processed/arcs_240.csv`

Directed task-node segments with distance, cruise altitude, climb height, and
descent height.

### `data/processed/terrain_envelope.npz`

Projected terrain envelope used by communication verification.

### `data/processed/relay_candidates.json`

Relay-position candidates and link parameters.

## Optimization Results

- `transport_time.json`: selected transport schedule.
- `transport_delivery.json`: box-level delivery records.
- `joint_nominal.json`: nominal joint delivery--relay schedule.
- `joint_fast.json`: makespan-priority guarded schedule.
- `joint_energy.json`: energy-priority guarded schedule.
- `joint_comprehensive.json`: comprehensive guarded schedule.
- `joint_two_groups.json`: two-group joint result.
- `consolidated_results.json`: consolidated numeric results used by the code.
- `lower_bound.json`: route-workload relaxation result.
- `lower_bound_columns.json`: route columns used in the bound.
- `objective_priority_experiment.json`: objective-priority comparison.
- `partition_threshold_experiment.json`: partition-threshold experiment.

## Verification Results

- `delivery_check.json`: overall verification summary.
- `joint_*/communication.csv`: interval-level direct and relay records.
- `joint_*/resource_checks.csv`: aircraft and energy-resource checks.
- `joint_*/independent_box_checks.csv`: box-level checks.
- `lower_bound.json`: lower-bound verification.

## Units

- Distance: metres.
- Time: seconds.
- Mass: kilograms.
- Volume: cubic metres.
- Energy: kilowatt-hours.
- Frequency: megahertz.
- Link margin and loss: decibels.
