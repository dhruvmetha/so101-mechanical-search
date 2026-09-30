# The real arm

What the `so101` session has measured on the physical SO-101, and what the sim deliberately ignores. Moved here from
CLAUDE.md on 2026-09-30. Read it before anything that moves toward hardware. Update it when that session reports.

The physical SO-101 is on host RobotLearning under `/home/dhruv/projects_dhruv/so101`, driven by a separate
Claude session named `so101`. The sim deliberately ignores things that only matter on hardware. On the real
arm, lerobot's degree zero is the midpoint of the calibration sweep, not the model's zero. Revisit all of
this before sending a sim-trained policy to hardware.

What that session has measured, as of 2026-09-22:

- Servo EEPROM limits are all looser than the MJCF's, so the MJCF range binds on every joint. Bus reads
  11.8 to 11.9 V, so it is the 12 V STS3215 build and menagerie's `forcerange="-2.94 2.94"` is right.
- The sag is pose-dependent, not a constant offset. `sag_mm = -0.0906 * reach_mm + 18.06`, so about 9 mm
  of droop per 100 mm of horizontal reach, R² 0.59 over 303 samples, never positive in either circle
  plane, which rules out backlash.
- The droop is two effects, not one, and the split is now measured on both sides. A reach-dependent
  part of about -8.0 mm per 100 mm, which is gravity against joint compliance and which
  `scripts/sag_experiment.py` reproduces at `kp=16` against menagerie's 998.22. Plus a symmetric
  direction-dependent band of roughly ±2.7 mm, which is 0.95 degrees of backlash at 320 mm reach,
  sitting between the published STS3215 figures of 0.62 degrees unloaded and 1.30 loaded.
- The `so101` session found the backlash by splitting its residual on the sign of each joint's velocity.
  The residual flips sign on all three joints that do the work, with a gap near 6 mm, which compliance
  cannot produce. Its first pooled slope of -9.06 was contaminated because direction and reach correlate
  around a circle; direction-averaged it is -8.00.
- What still disagrees is the level, not the gradient. At `kp=16` with matching branch and window, sim
  slope -8.00 and intercept +11.42 against a direction-averaged +14.75 on hardware, about 3.3 mm. Two
  candidates are now ruled out. Calibration and link-length errors cancel, because the hardware number
  is FK(measured angles) minus FK(commanded angles) through one model with one zero, so anything
  constant in that model appears on both sides. And joint `frictionloss` does not do it: sweeping it
  from 0 to 0.2 N·m moves the intercept only from +11.10 to +11.34 while dragging the slope from -8.19
  to -7.36. Friction trades against stiffness and leaves the level alone.
- The gap closes with a per-joint stiffness and needs nothing exotic. `kp = [16, 16, 30, 16, 16]` gives
  slope -7.81 and -10.55 mm of sag at 320 mm reach against the hardware's -8.00 and -10.85. Forcing one
  stiffness onto every joint was doing the work. Elbow stiffness sets the level nearly independently of
  the gradient, which is the lever a uniform kp cannot pull. Two observations against five free
  parameters is not a unique fit and says nothing about the true per-joint values, but it does mean the
  gap is not evidence of anything outside the model.
- Anything that scales gravity torque uniformly is degenerate with a global kp, because kp gets fitted
  to the slope. Removing the 12 g `camera_mount` drops sag at 320 mm by 0.94 mm; refitting kp to hold
  the slope puts 0.55 mm of that back. Only errors that change the shape of the torque-against-reach
  curve can move the level once the slope is matched.
- The hardware numbers are floors, both of them. Sag and backlash are both inferred from the servos'
  own encoders through a rigid model, so gearbox compliance and gear deadband show up while flex or slop
  in the horn-to-link joint and bending of the printed links cannot. Real repeatability is worse than
  5 mm peak to peak at 320 mm reach, not equal to it. A dial indicator on the gripper, read at a fixed
  pose and after approaching it from each direction, settles both at once.
- Careful with the obvious stiction test. "Sag is flatter at low reach than high" is true in this sim at
  zero friction (-7.16 over 225-280 mm against -8.19 over 280-360), because gravity torque is not linear
  in reach. The pattern alone is not evidence of stiction; only its change with friction is.
- Raising servo P should cut the compliance part and will do nothing for the backlash, because a stiffer
  loop cannot close a mechanical deadband. So repeatability has a floor near 5 mm peak to peak at
  320 mm reach whatever the gains, and it depends on approach direction. Plan grasps to approach from a
  consistent direction rather than trusting absolute position.
- Coincidence worth not repeating: lerobot writes `P_Coefficient=16` to the servos and `kp=16` is what
  matches here. Different quantities, different units, a servo register against N·m/rad. The numbers
  agreeing means nothing.
- In sim the droop is static. Streaming the circle at 12 deg/s moves the slope less than 10% from the
  settled result, so nothing velocity-dependent shows up at that speed.
- The gravity torque at the shoulder is 0.18 to 0.42 N·m against 2.94 N·m available, so this is about
  stiffness, not force. Do not touch `forcerange`. This sim still runs the stiff menagerie gains on
  purpose; softening them changes every grasp, and it is a sim-to-real step rather than a sim one.
- Sag against reach is only well defined once the IK branch is fixed. Five joints against three position
  constraints leaves two redundant DOF, and across IK branches reaching the same circle the slope moves
  from -5.7 to -7.1 at fixed kp. Worth remembering before fitting anything to a commanded-versus-achieved
  curve.
- lerobot runs the follower at `max_relative_target=4.0` with P=16, I=0, D=32, written on every connect.
  `scripts/replay.py` on that host executes a `so101-joint-trajectory-v1` file at 30 Hz and writes the
  measured angles back per frame.
- The search task puts the table `Layout.depth` below the base, 10 cm as chosen on 2026-09-25. The real
  setup needs the arm on a riser that high, or a bin set that far into the table.
- The shelf task needs a shelf board 9 cm above the base's mounting plane, its front edge 0.30 m from the base
  origin, a 32 by 18 cm board under a 15 cm opening (`ShelfLayout` in `models/shelf_objects.xml`). The whole
  side-grasp design assumes the wrist camera mount; measure the real one before building a shelf.
- Still unknown and blocking any transfer: the table surface height relative to the base mounting plane,
  whether the base position repeats across remounts, whether a bin exists, and the real jaw gap in mm at
  lerobot 0, 25, 50, 75 and 100. The gripper mapping is the worst of these. This sim maps lerobot 0..100
  onto MJCF -10..+100 degrees, a 133 mm gap at 100; the real calibration spans 59.78 degrees of motor
  travel, which lands near MJCF +50 and an 82 mm gap. Same command, 51 mm apart.
