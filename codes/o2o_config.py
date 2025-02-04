import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

from codes.core import flatten_dict
from codes.import_utils import is_bitsandbytes_available, is_torchvision_available


@dataclass
class O2OConfig:
    """
    Configuration class for O2OTrainer
    """

    # common parameters
    exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")]
    """the name of this experiment (by default is the file name without the extension name)"""
    run_name: Optional[str] = ""
    """Run name for wandb logging and checkpoint saving."""

    log_with: Optional[Literal["wandb", "tensorboard"]] = "wandb"
    """Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"""
    tracker_kwargs: dict = field(default_factory=dict)
    """Keyword arguments for the tracker (e.g. wandb_project)"""
    accelerator_kwargs: dict = field(default_factory=dict)
    """Keyword arguments for the accelerator"""
    project_kwargs: dict = field(default_factory=dict)
    """Keyword arguments for the accelerator project config (e.g. `logging_dir`)"""
    logdir: str = "logs"
    """Top-level logging directory for checkpoint saving."""

    num_checkpoint_limit: int = 10
    """Number of checkpoints to keep before overwriting old ones."""
    # tracker_project_name: str = "stable_diffusion_training"
    tracker_project_name: str = "Dev"
    """Name of project to use for tracking"""


    # train_learning_rate: float = 3e-4  
    train_learning_rate: float = 3e-4
    """Learning rate."""
    seed: int = 0  
    """Seed value for random generations"""
    global_step: int=0
    """Global step, using with checkpoint save folder"""
    high_reward: float=100.00
    """Reward for a sample picture from dataset """
    low_reward: float=50.00
    """Reward for a generated picture from model """
    resolution:int =256

    "Image square size"
    reward_function_usage: bool = False
    """ Using pretrained model to get reward, otherwise, use image with reward in advance"""

    valid_batch_size: int=1
    """ Validation dataset batch size"""
    valid_size: int=50
    dataset_index: int=1

    sample_batch_size: int = 2
    """Batch size (per GPU!) to use for sampling."""
    offpolicy_sample_batch_size: int = 1
    """Batch size for offpolicy from dataset - not larger than sample_batch_size"""

    train_batch_size: int = 2
    """Batch size (per GPU!) to use for training."""

    train_num_inner_epochs: int = 1
    """Number of inner epochs per outer epoch."""

    sample_num_steps: int = 50
    """Number of sampler inference steps."""
    sample_num_batches_per_epoch: int = 1
    """Number of batches to sample per epoch."""
    # resume_from: Optional[str] = "./outputs/checkpoint/checkpoints/checkpoint_20"


    # train_mode: str = "contrastive"
    # """ offpolicy or contrastive or other. Train mode"""


    """Top-level logging directory for checkpoint saving."""
    # hyperparameters


    num_epochs: int =11
    resume_from: Optional[str] = "./outputs/checkpoints/checkpoint_0"
    resume_from: Optional[str] = ""
    """== checkpoin from // Resume training from a checkpoint."""

    save_freq: int = 5
    """Number of epochs between saving model checkpoints."""

    huggingface_note: str = "test_VSC"
    """Save model note."""

    pass_images:int=0
    """Dataset passing using with checkpoint resuming"""

    artistic_log_on: bool= True
    """Show artistic score on wandb"""




# DEFAULT NO CHANGE
    mixed_precision: str = "fp16"
    """Mixed precision training."""
    allow_tf32: bool = True
    """Allow tf32 on Ampere GPUs."""
    sample_eta: float = 1.0
    """Eta parameter for the DDIM sampler."""
    sample_guidance_scale: float = 5.0
    """Classifier-free guidance weight."""
    train_use_8bit_adam: bool = False
    """Whether to use the 8bit Adam optimizer from bitsandbytes."""
    train_adam_beta1: float = 0.9
    """Adam beta1."""
    train_adam_beta2: float = 0.999
    """Adam beta2."""
    train_adam_weight_decay: float = 1e-4
    """Adam weight decay."""
    train_adam_epsilon: float = 1e-8
    """Adam epsilon."""
    train_gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps."""
    train_max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping."""

    train_cfg: bool = True
    """Whether or not to use classifier-free guidance during training."""
    train_adv_clip_max: float = 5
    """Clip advantages to the range."""
    train_clip_range: float = 1e-4
    """The PPO clip range."""
    train_timestep_fraction: float = 1.0
    """The fraction of timesteps to train on."""  
    per_prompt_stat_tracking: bool = False
    """Whether to track statistics for each prompt separately."""
    per_prompt_stat_tracking_buffer_size: int = 16
    """Number of reward values to store in the buffer for each prompt."""
    per_prompt_stat_tracking_min_count: int = 16
    """The minimum number of reward values to store in the buffer."""
    async_reward_computation: bool = False
    """Whether to compute rewards asynchronously."""
    max_workers: int = 2
    """The maximum number of workers to use for async reward computation."""
    negative_prompts: Optional[str] = ""
    """Comma-separated list of prompts to use as negative examples."""

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.log_with not in ["wandb", "tensorboard"]:
            warnings.warn(
                "Accelerator tracking only supports image logging if `log_with` is set to 'wandb' or 'tensorboard'."
            )

        if self.log_with == "wandb" and not is_torchvision_available():
            warnings.warn("Wandb image logging requires torchvision to be installed")

        if self.train_use_8bit_adam and not is_bitsandbytes_available():
            raise ImportError(
                "You need to install bitsandbytes to use 8bit Adam. "
                "You can install it with `pip install bitsandbytes`."
            )
        # if self.sample_batch_size>2*self.offpolicy_sample_batch_size and not self.reward_function_usage:
        #     raise ImportError(
        #         "offpolicy_sample_batch_size should be large than or equal half of sample_batch_size"
        #     )

