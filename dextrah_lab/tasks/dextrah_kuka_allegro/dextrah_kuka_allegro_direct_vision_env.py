# dextrah_kuka_allegro_direct_vision_env.py
#
# Direct-vision (no distillation) environment wrapper.
# - Policy obs: student proprio (159-dim)
# - Exports camera image each step ("img" for depth, optional "rgb")
# - Critic obs: privileged (same as original compute_critic_observations)

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

from .dextrah_kuka_allegro_env import DextrahKukaAllegroEnv
from .dextrah_kuka_allegro_direct_vision_env_cfg import DextrahKukaAllegroDirectVisionEnvCfg
from isaaclab.sensors import TiledCamera

class DextrahKukaAllegroDirectVisionEnv(DextrahKukaAllegroEnv):
    """Direct RL environment with vision + student proprio policy obs."""

    cfg: DextrahKukaAllegroDirectVisionEnvCfg

    def __init__(self, cfg: DextrahKukaAllegroDirectVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Force direct-vision behavior regardless of legacy distillation flags.
        self.use_camera = True
        self.simulate_stereo = False
        self.cfg.simulate_stereo = False

        # Ensure the public counters reflect the policy obs (student proprio).
        self.num_observations = int(self.cfg.num_student_observations)

    # ---------------------------------------------------------------------
    # Scene / sizing
    # ---------------------------------------------------------------------

    def _setup_scene(self):
        super()._setup_scene()

        if not hasattr(self, "_tiled_camera"):
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

    def _reset_idx(self, env_ids: torch.Tensor):
        orig_use_camera = self.use_camera
        self.use_camera = False
        super()._reset_idx(env_ids)
        self.use_camera = orig_use_camera

        if self.use_camera:
            self._reset_cameras_only(env_ids)

    def _reset_cameras_only(self, env_ids: torch.Tensor):
        num_ids = env_ids.shape[0]
        np_env_ids = env_ids.cpu().numpy()

        rand_rots = np.random.uniform(
            -self.cfg.camera_rand_rot_range,
            self.cfg.camera_rand_rot_range,
            size=(num_ids, 3)
        )
        new_rots = rand_rots + self.camera_rot_eul_orig  # 原 env 在 __init__ 里已算好
        new_rots_quat = R.from_euler('xyz', new_rots, degrees=True).as_quat()
        new_rots_quat = new_rots_quat[:, [3, 0, 1, 2]]  # wxyz
        new_rots_quat = torch.tensor(new_rots_quat, device=self.device).float()

        new_pos = self.camera_pos_orig + torch.empty(
            num_ids, 3, device=self.device
        ).uniform_(
            -self.cfg.camera_rand_pos_range,
            self.cfg.camera_rand_pos_range
        )

        # 更新 camera_pose（如果你后续有用到投影/uv 之类，这个保持一致更安全）
        self.camera_pose[np_env_ids, :3, :3] = R.from_euler('xyz', new_rots, degrees=True).as_matrix()
        self.camera_pose[np_env_ids, :3, 3] = (
                new_pos + self.scene.env_origins[env_ids]
        ).cpu().numpy()

        # 同步缓存
        self.left_pos[env_ids] = new_pos + self.scene.env_origins[env_ids]
        self.left_rot[env_ids] = new_rots_quat

        # 真正把相机挪到位姿
        self._tiled_camera.set_world_poses(
            positions=self.left_pos[env_ids],
            orientations=self.left_rot[env_ids],
            env_ids=env_ids,
            convention="ros"
        )

    def _setup_policy_params(self):
        """Size the actor/critic spaces for direct RL.

        - Actor uses student proprio observation (159)
        - Critic uses privileged observation (214 + one-hot object ID)
        """
        if self.cfg.objects_dir not in self.cfg.valid_objects_dir:
            raise ValueError(f"Need to specify valid directory of objects for training: {self.cfg.valid_objects_dir}")

        num_unique_objects = self.find_num_unique_objects(self.cfg.objects_dir)

        # Actor
        self.cfg.num_student_observations = 159

        # Teacher-like (kept for reference; not used as policy obs here)
        self.cfg.num_teacher_observations = 167 + num_unique_objects

        # IMPORTANT: policy obs is always the student proprio in this environment
        self.cfg.num_observations = self.cfg.num_student_observations

        # Critic privileged obs size (matches compute_critic_observations)
        self.cfg.num_states = 214 + num_unique_objects

        self.cfg.state_space = self.cfg.num_states
        self.cfg.observation_space = self.cfg.num_observations
        self.cfg.action_space = self.cfg.num_actions

    # ---------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Export clean observation dict: policy proprio + (optional) image + critic."""
        policy_obs = self.compute_student_policy_observations()

        # Keep asymmetric critic behavior compatible with your training stack.
        if getattr(self.cfg, "asymmetric_obs", True):
            critic_obs = self.compute_critic_observations()
        else:
            critic_obs = policy_obs

        observations = {
            "policy": policy_obs,
            "critic": critic_obs,
        }

        if getattr(self.cfg, "use_camera", True):
            # Depth as the main "video" signal. Shape from camera is (N, H, W, 1).
            if getattr(self.cfg, "export_depth", True):
                depth = self._tiled_camera.data.output["depth"].clone()
                depth[depth <= 1e-8] = 10
                depth[depth > self.cfg.d_max] = 0.0
                depth[depth < self.cfg.d_min] = 0.0
                observations["img"] = depth.permute((0, 3, 1, 2))  # (N,1,H,W)

            # RGB is optional (heavier).
            if getattr(self.cfg, "export_rgb", True):
                rgb = self._tiled_camera.data.output["rgb"].clone().permute((0, 3, 1, 2)) / 255.0
                observations["rgb"] = rgb

        return observations
