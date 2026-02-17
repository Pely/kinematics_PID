# SECAR PID + Kinematics Simulation (2-body / 3-body, depth sampling, multi-Ex)

## Overview

This Python workflow simulates reaction kinematics and downstream energy-loss transport for SECAR-style PID studies. It can run either **2-body kinematics** (beam + target → heavy recoil + light ejectile) or **3-body kinematics** (heavy recoil + 2 neutrons), includes **random reaction-depth sampling** inside the gas target, and propagates particles through the remaining target and downstream materials up to the detector stack.
\it{This code is an adaptation of original scripts from Ruchi Garg}

## What it calculates

### 1) Beam energy at reaction point (target depth sampling)

For each Monte-Carlo event, the reaction location is sampled uniformly along the target thickness. The code computes:

* reaction depth fraction ( f \in [0,1] )
* distance traveled in target up to reaction point
* **beam energy at the reaction point** after energy loss in the target up to that depth

### 2) Reaction kinematics at updated beam energy

At the sampled depth, the code runs kinematics using the **beam energy after pre-reaction energy loss**:

* **2-body mode**: produces heavy-recoil solutions (one or two branches depending on kinematic regime) and the corresponding **light ejectile** lab angle and energy.
* **3-body mode**: generates heavy recoil + two neutrons (event-by-event), returning energies and emission angles at production.

### 3) Heavy recoil transport through materials (energy-loss chain)

For each kinematic solution/event, the code transports the **heavy recoil** through:

* the **remaining** target thickness (from reaction point to target exit)
* optional **stripper foil**
* then computes detector response via the PID routine (IC + DSSD), producing:

  * **IC dE**
  * **DSSD residual E**

### 4) PID construction (IC dE vs DSSD E)

The script builds the simulated PID scatter:

* heavy recoils after target + stripper
* optional “beam after target + stripper” point for reference (depending on settings)

### 5) Light ejectile distributions at production

The code stores the **light ejectile** kinematics at the production point:

* emission angle (lab)
* kinetic energy (lab)
  For 3-body, it can store both neutrons separately.

### 6) Multiple excitation energies (multi-Ex runs)

The workflow can run over a list of excitation energies (E_x). For each (E_x):

* a new reaction configuration is created
* kinematics + transport are repeated
* outputs are accumulated into **cumulative** distributions and CSV outputs
  Each saved row is tagged with the excitation energy used.

## Outputs

### Plots

Typical plots generated include:

* **PID**: IC dE vs DSSD E for heavy recoils (and optionally beam reference)
* **Kinematics (production)**:

  * heavy recoil energy vs recoil angle
  * light ejectile energy vs ejectile angle
* **After-strip heavy recoil distribution**: recoil energy vs recoil angle after stripper foil
* Optional **diagnostic kinematics curves** evaluated at fixed target positions (front/middle/back) to visualize the effect of beam energy loss with depth.

### CSV files

The script can write CSV tables for later analysis/plotting:

* **Heavy recoil after stripper foil** (per event / per solution): energy, emission angle, azimuth, and optionally reaction depth and (E_x).
* **Light ejectile at production**: energy and emission angle at the reaction point (tagged by (E_x), depth, etc.).

### NPZ files (optional)

PID arrays can be saved to `.npz` for quick re-plotting without rerunning simulation.

## Physics sequence per event (summary)

1. Sample reaction depth in target
2. Compute beam energy at reaction point (energy loss before reaction)
3. Run kinematics at updated beam energy (2-body or 3-body)
4. Transport heavy recoil through remaining target + stripper foil
5. Compute detector energy deposits and build PID
6. Save recoil after-strip distribution + light ejectile production kinematics

## Notes / assumptions

* Energy-loss calculations depend on the selected stopping-power library (e.g., SRIM/VICAR tables).
* In 2-body mode, there may be **one or two** kinematic branches depending on lab angle relative to (\theta_{\max}).
* Angles are tracked in radians internally; plotting may convert to mrad/deg as needed.

---
