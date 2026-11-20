# IEEE AIxVR NMPC Experiments

This folder isolates the model-comparison study from the CAPE overtaking code.
The experiment matrix is one nominal NMPC controller with four internal models:

1. DPM trained on the eight-lap AutoDRIVE dataset.
2. DDM trained on the same lap-level split.
3. EPM trained on the same lap-level split.
4. A fixed single-track model with identified physical parameters.

The single-track baseline is not learned. Its wheelbase, steering actuator,
and motor gain are identified using training laps only; validation selects the
actuator delay and the held-out test lap is used only for the final report.

Use complete laps for the train/validation/test split. Do not randomly split
individual samples because adjacent telemetry samples are strongly correlated.

## Current Runtime Support

The live ROS controller supports `epm`, `ddm`, `dpm`, and `single_track`. DDM
uses its trained parameter network and matching no-load-transfer dynamic bicycle
model. DPM uses its trained Pacejka/force network and a locally evaluated learned
PWM-to-force map, preserving throttle sensitivity without EPM coefficients.

## DPM/DDM Training

`train_dpm_ddm.py` trains DPM and DDM with the same whole-lap split, history,
sample time, rollout horizon, optimizer, and network capacity. Normalization is
fit on training laps only. Since `VREF` was not recorded, DPM estimates direct
longitudinal force from measured state and the recorded actuator channels
instead of inventing a training input.

```bash
python3 src/autodrive_ros2/autodrive_f1tenth/autodrive_f1tenth/aixvr/train_dpm_ddm.py \
  --models dpm ddm --epochs 300 --patience 60 --overwrite
```

## Run

```bash
ros2 launch autodrive_f1tenth run_nmpc_test.launch.py model:=epm
ros2 launch autodrive_f1tenth run_nmpc_test.launch.py model:=ddm
ros2 launch autodrive_f1tenth run_nmpc_test.launch.py model:=dpm
ros2 launch autodrive_f1tenth run_nmpc_test.launch.py model:=single_track
```

The car follows the ETHZ reference for one lap and then stops. Results are saved
under `output/aixvr_nmpc/` using a deterministic model-specific filename.

The CSV includes lap completion time, global position, longitudinal and lateral
path error, Euclidean position error, longitudinal/lateral velocity, yaw rate,
derived longitudinal acceleration, commands, actuator feedback, prediction,
objective value, solver timing, and solver status.

Generate the four-model trajectory, telemetry/error overlays, and summary table:

```bash
ros2 run autodrive_f1tenth aixvr_plot_nmpc_comparison
```

## Uncertainty Experiments

The uncertainty launch retains the same controller, model checkpoint, reference,
costs, horizon, and initial scenario. A seeded ROS bridge perturbs the measurements
seen by NMPC and/or the commands delivered to the Unity vehicle. Ground-truth and
perturbed signals are logged separately.

All pure-pursuit warmup samples are passed through clean. The bridge activates
uncertainty only after `PURE_PURSUIT_WARMUP` ends, so the learned models build
their initial history from unperturbed measurements and actuator commands.

Wall contact is not inferred or used to stop a run. If the simulator visibly
crashes, stop the launch with `Ctrl-C`; both loggers flush every sample as it is
written and close their CSVs during shutdown, preserving the partial run.

```bash
ros2 launch autodrive_f1tenth run_nmpc_uncertainty.launch.py \
  model:=epm experiment:=perception severity:=1.0 seed:=47
ros2 launch autodrive_f1tenth run_nmpc_uncertainty.launch.py \
  model:=epm experiment:=actuation severity:=1.0 seed:=47
ros2 launch autodrive_f1tenth run_nmpc_uncertainty.launch.py \
  model:=epm experiment:=dynamics severity:=1.0 seed:=47
ros2 launch autodrive_f1tenth run_nmpc_uncertainty.launch.py \
  model:=epm experiment:=combined severity:=1.0 seed:=47
```

Use the same severity and seed matrix for `epm`, `dpm`, `ddm`, and
`single_track`. Recommended initial levels are `0.0`, `0.5`, `1.0`, `1.5`, and
`2.0`, with at least five seeds per point. Severity `1.0` uses 1 cm additive
position noise (origin invariant), the CAPE base 1% velocity/yaw measurement
scale, 1% command noise, and one control-period command delay when the selected
profile enables perception or actuation uncertainty.

`experiment:=dynamics` applies no ROS-side uncertainty. Run it with the dedicated
Unity export configured with vehicle mass `4.164 kg`, front tire stiffness `0.8`,
and nominal rear tire stiffness `1.0`. The controller's internal dynamics model is
held fixed. The launch records the expected Unity configuration in the JSON
manifest, but ROS cannot verify Unity Inspector values.

`experiment:=combined` uses that same modified Unity export and additionally
injects the perception and actuation uncertainty selected by `severity`. No
command-side dynamics proxy is applied. Use the nominal Unity export for
`nominal`, `perception`, and `actuation` experiments.

Controller CSVs are written to `output/aixvr_nmpc/`. Ground-truth uncertainty
logs and complete JSON manifests are written to
`output/aixvr_nmpc/uncertainty/`. The latter include true and measured state,
nominal and applied controls, wall clearance, and boundary-violation flags.

After collecting trials, generate the time-aligned paper metrics with:

```bash
ros2 run autodrive_f1tenth aixvr_summarize_uncertainty
```

This writes `output/aixvr_nmpc/aixvr_uncertainty_summary.csv` with true-state
lateral/position/velocity errors, completion time, solver success, boundary
violation samples and events, and throttle/steering total variation.
