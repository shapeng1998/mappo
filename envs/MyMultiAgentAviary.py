import os

import numpy as np
import pybullet as p
from gym import spaces
from gym_pybullet_drones.envs.multi_agent_rl import BaseMultiagentAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)

# Constants
AGGR_PHY_STEPS = 5


class MyMultiAgentAviary(BaseMultiagentAviary):
    """My personal Multi Agent Aviary environment for RL research"""

    def __init__(self, gui=False, record=False, num_drones: int = 2):
        super().__init__(
            gui=gui,
            record=record,
            num_drones=num_drones,
            initial_xyzs=np.array([[0.5, 0.1, 1], [0.2, 0.6, 1]]),
            obs=ObservationType.KIN,
            act=ActionType.PID,
            aggregate_phy_steps=AGGR_PHY_STEPS,
        )

        # debug
        self.agent_num = num_drones
        self.obs_dim = self.observation_space[0].shape[0]
        self.action_dim = self.action_space[0].n

        self.EPISODE_LEN_SEC = 10
        self.SIGMA = 0.1
        self.THRESHOLD_DIST = 0.3

        # TODO: add random position
        self.TARGET_POS = np.array([[0.8, 0.3, 1], [0.5, 0.5, 1]])
        self.OBSTACLE_POS = np.array([[0.5, 0.5, 1]])

        self.DIRECTION = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, -1, 0],
                [1, 1, 0],
                [-1, -1, 0],
                [1, -1, 0],
                [-1, 1, 0],
            ]
        )

    # def _addObstacles(self):
    #     self.OBSTACLE_POS = np.array([[0.5, 0.5, 1]])
    #     self.OBSTACLE_IDS = [self._loadURDF(self.OBSTACLE_POS[i]) for i in range(len(self.OBSTACLE_POS))]

    # def _loadURDF(self, xyz):
    #     return p.loadURDF(
    #         os.path.dirname(os.path.abspath(__file__)) + "/../assets/cube.urdf",
    #         xyz,
    #         p.getQuaternionFromEuler([0, 0, 0]),
    #         physicsClientId=self.CLIENT,
    #     )

    def _actionSpace(self):
        return spaces.Dict({i: spaces.Discrete(16) for i in range(self.NUM_DRONES)})

    def _preprocessAction(self, action):
        action = {i: np.matmul(action[i], self.DIRECTION) for i in range(self.NUM_DRONES)}
        return super()._preprocessAction(action)

    def _computeReward(self):
        r1 = 0
        for t in self.TARGET_POS:
            min_dist = 2
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                min_dist = min(min_dist, np.linalg.norm(state[0:2] - t[0:2]))
                r1 -= min_dist

        r2 = 0
        for i in range(self.NUM_DRONES):
            for j in range(i + 1, self.NUM_DRONES):
                state_i = self._getDroneStateVector(i)
                state_j = self._getDroneStateVector(j)
                dist2 = np.linalg.norm(state_i[0:2] - state_j[0:2])
                if dist2 < self.THRESHOLD_DIST + self.SIGMA:
                    r2 -= self.THRESHOLD_DIST + self.SIGMA - dist2

        r3 = 0
        for i in range(self.NUM_DRONES):
            for o in self.OBSTACLE_POS:
                state = self._getDroneStateVector(i)
                dist3 = np.linalg.norm(state[0:2] - o[0:2])
                if dist3 < self.THRESHOLD_DIST + self.SIGMA:
                    r3 -= self.THRESHOLD_DIST + self.SIGMA - dist3

        # debug
        rewards = []
        tot_r = r1 + r2 + r3
        for _ in range(self.NUM_DRONES):
            rewards.append([tot_r])
        return rewards

    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and
            one additional boolean value for key "__all__".

        """
        bool_val = True if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        done = [bool_val for i in range(self.NUM_DRONES)]
        # done["__all__"] = bool_val  # True if True in done.values() else False
        return done

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(
                state,
                clipped_pos_xy,
                clipped_pos_z,
                clipped_rp,
                clipped_vel_xy,
                clipped_vel_z,
            )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = (
            state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        )

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_y,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20],
            ]
        ).reshape(
            20,
        )

        return norm_and_clipped

    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in MyMultiAgentAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                    state[0], state[1]
                ),
            )
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in MyMultiAgentAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]),
            )
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in MyMultiAgentAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                    state[7], state[8]
                ),
            )
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in MyMultiAgentAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                    state[10], state[11]
                ),
            )
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in MyMultiAgentAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]),
            )
