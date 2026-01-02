# 🤖 HIL-SERL Training Guide: SO-101 Configuration

This workflow enables you to train a real-robot policy in just a few hours by combining human interventions with a distributed **Soft Actor-Critic (SAC)** algorithm.

## 1. Hardware Overview

* **Robot (Follower):** SO-101 (`/dev/ttyACM0`)
* **Teleoperator (Leader):** SO-101 Leader (`/dev/ttyACM1`)
* **Camera:** Hand-eye OpenCV Camera (Index 4)

---

## 2. The Master Configuration (`hil_serl_so101.yaml`)

This configuration uses the latest LeRobot structured format. Save this as `configs/hil_serl_so101.yaml`.

---

## 3. Workflow Steps

### Step 1: Calibration (Workspace Bounds)

Before training, you must define the physical operational space. Run the calibration script and move the leader arm to the task limits.

```bash
uv run lerobot-find-joint-limits \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=R12255108 \
  --robot.disable_torque_on_disconnect=true \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=R07255108 \
  --urdf_path=configs/robots/SO101/so101_new_calib.urdf
```

For example:

```
========================================
FINAL RESULTS
========================================

# End Effector Bounds (x, y, z):
max_ee = [0.3329, 0.2511, 0.4505]
min_ee = [0.0785, -0.1812, 0.0318]

# Joint Position Limits (radians):
max_pos = [59.274, 43.3124, 45.3394, 98.3857, 21.2607, 2.6752]
min_pos = [-67.0417, -18.239, -66.8778, -18.0119, -24.2573, 2.6752]
```

*Note: Copy the printed `Max/Min ee position` into the `end_effector_bounds` section of your YAML.*

### Step 2: Seed Demonstrations

Collect 15 successful episodes using your leader arm to "seed" the RL buffer.

```bash
uv run python -m lerobot.rl.gym_manipulator \
  --config_path=configs/hil_serl_so101.yaml \
  --mode=record

```

### Step 3: Visual ROI Cropping

Ensure the agent focuses only on the workspace to ignore background distractions.

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id your_username/so101_hil_dataset

```

*Action: Draw the box around the hand-eye view, press 'c' to confirm, and update the `crop_params_dict` in your YAML with the output coordinates.*

### Step 4: Training (Actor-Learner)

HIL-SERL requires two processes running simultaneously.

**Terminal 1 (The Learner):**

```bash
python -m lerobot.rl.learner --config_path configs/hil_serl_so101.yaml

```

**Terminal 2 (The Actor/Robot):**

```bash
python -m lerobot.rl.actor --config_path configs/hil_serl_so101.yaml

```

---

## 4. Operational Controls (SO-101 Leader)

During the **Actor** phase, use the following keyboard/leader controls:

| Action        | Control            | Description                                      |
| ------------- | ------------------ | ------------------------------------------------ |
| **Intervene** | `Spacebar`         | Pauses policy; Leader takes control of Follower. |
| **Resume RL** | `Spacebar` (again) | Hands control back to the AI policy.             |
| **Success**   | `S` Key            | Ends episode with Reward=1.                      |
| **Failure**   | `Esc` Key          | Ends episode with Reward=0.                      |

> [!TIP]
> **Effective Interventions:** At the start of training, let the robot "wobble" and explore. Only intervene when it moves out of the task area or is about to crash. As the policy improves, reduce interventions until it can complete the task autonomously.

---

## 5. Monitoring

If `wandb` is enabled in your environment, track the **Intervention Rate**. A successful run shows this rate decreasing steadily as the agent takes over the workload.
