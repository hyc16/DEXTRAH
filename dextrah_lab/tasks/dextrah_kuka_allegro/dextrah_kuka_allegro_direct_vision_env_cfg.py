# dextrah_kuka_allegro_direct_vision_env_cfg.py
# Minimal wrapper cfg for a "direct RL + vision" experiment.
#
# Key behavior:
# - No distillation / no DAgger
# - Policy obs uses the student proprioceptive features (159-dim)
# - Camera is enabled to provide an image tensor each step
# - Critic can still use privileged obs through cfg.asymmetric_obs (unchanged)

from isaaclab.utils import configclass

from .dextrah_kuka_allegro_env_cfg import DextrahKukaAllegroEnvCfg


@configclass
class DextrahKukaAllegroDirectVisionEnvCfg(DextrahKukaAllegroEnvCfg):
    """Configuration for direct RL with vision.

    Inherits all task parameters from the original cfg so rewards and
    randomizations remain consistent, but disables distillation-only behavior.
    """

    # Disable distillation pipeline knobs (kept for backward compatibility)
    distillation: bool = False

    # Always train with camera enabled in this variant.
    use_camera: bool = True

    # Keep stereo off for simplicity.
    simulate_stereo: bool = False

    # By default, do not export distillation-only observation keys.
    export_expert_policy: bool = False
    export_aux_info: bool = False
    export_masks: bool = False

    # What image(s) to export. You can turn rgb off for speed.
    export_rgb: bool = True
    export_depth: bool = True

    # Policy obs source.
    policy_obs_mode: str = "student"  # {"student", "teacher"}
