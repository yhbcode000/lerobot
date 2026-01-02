#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from multiprocessing import Queue
import time
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
import os
import sys
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_so101_leader import (
    SO101LeaderConfig, 
    SO101LeaderEndEffectorConfig
)

import numpy as np

import placo

logger = logging.getLogger(__name__)


class SO101Leader(Teleoperator):
    """
    SO-101 Leader Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101LeaderConfig
    name = "so101_leader"

    def __init__(self, config: SO101LeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

class SO101LeaderEndEffector(SO101Leader):
    config_class = SO101LeaderEndEffectorConfig
    name = "so101_leader_ee"

    def __init__(self, config: SO101LeaderEndEffectorConfig):
        super().__init__(config)
        self.config = config
        
        # 1. Kinematics Setup
        self.model = placo.RobotWrapper(config.urdf_path)
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

        # 2. State Management
        self.misc_keys_queue = Queue()
        self.listener = None
        self.is_intervention_active = False
        self.motors_initialized = False 
        
        # 3. Smoothing Variables
        self.force_buffer = []
        self.buffer_size = 5  # Number of frames to average to prevent spikes
        self.cooldown_end_time = 0

    def _set_torque(self, enable: bool):
        state = 1 if enable else 0
        for motor_name in self.bus.motors.keys():
            try:
                # FIXED: Correct argument order (Register, Value, Motor_Name)
                self.bus.write("Torque_Enable", motor_name, state)
            except Exception as e:
                logger.error(f"Torque error on {motor_name}: {e}")
        # Add a small delay after bulk torque change to let bus settle
        time.sleep(0.05)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.is_connected:
            return

        # Recovery/Init: Only stiffen if we aren't currently being grabbed
        if not self.motors_initialized and not self.is_intervention_active:
            # Check cooldown to prevent rapid snap-back oscillations
            if time.time() > self.cooldown_end_time:
                self._set_torque(enable=True)
                self.motors_initialized = True

        # TELEOP MODE: Human is in control, Leader arm must stay limp
        if self.is_intervention_active:
            return
        
        target_joints = feedback.get("target_joint_pos")
        if target_joints:
            valid_targets = {k: v for k, v in target_joints.items() if k in self.bus.motors}
            self.bus.sync_write("Goal_Position", valid_targets)

    def get_teleop_events(self) -> dict[str, Any]:
        if not self.is_connected:
            return {TeleopEvents.IS_INTERVENTION: False, TeleopEvents.TERMINATE_EPISODE: False}

        # --- Part A: Improved Force Detection ---
        try:
            currents = self.bus.sync_read("Present_Current")
            arm_currents = [abs(val) for name, val in currents.items() if name != "gripper"]
            raw_max_force = max(arm_currents) if arm_currents else 0
            
            # Simple Moving Average to smooth out motor spikes during shadowing
            self.force_buffer.append(raw_max_force)
            if len(self.force_buffer) > self.buffer_size:
                self.force_buffer.pop(0)
            avg_force = sum(self.force_buffer) / len(self.force_buffer)
            
            # Detect human intervention based on smoothed average
            is_intervention = avg_force > self.config.intervention_threshold
        except ConnectionError:
            is_intervention = self.is_intervention_active

        # TRANSITION: SHADOW -> TELEOP (Grabbed)
        if is_intervention and not self.is_intervention_active:
            self.is_intervention_active = True
            self._set_torque(enable=False)
            logger.info(">>> INTERVENTION START: Leader is now LIMP")
        
        # TRANSITION: TELEOP -> SHADOW (Released)
        elif not is_intervention and self.is_intervention_active:
            self.is_intervention_active = False
            self.motors_initialized = False 
            # Set a 0.5s cooldown before the arm stiffens up again
            self.cooldown_end_time = time.time() + 0.5
            logger.info("<<< INTERVENTION END: Preparing to Resume Shadowing")

        # --- Part B: Keyboard ---
        terminate_episode = False
        success = False
        rerecord_episode = False
        while not self.misc_keys_queue.empty():
            char = self.misc_keys_queue.get_nowait()
            if char in ["s", "r", "q"]:
                terminate_episode = True
                self.motors_initialized = False
                if char == "s": success = True
                elif char == "r": rerecord_episode = True

        return {
            TeleopEvents.IS_INTERVENTION: self.is_intervention_active,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }