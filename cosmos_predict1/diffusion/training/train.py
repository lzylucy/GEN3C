# Updated training script with dataloader configuration
import argparse
import importlib
import os
from typing import Dict, Any

import torch.distributed as dist
from loguru import logger as logging
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('/fsx-siro/liuzeyi/GEN3C')
sys.path.append('/fsx-siro/liuzeyi/4dgen')

from cosmos_predict1.diffusion.config.config import Config
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.config_helper import get_config_module, override
from cosmos_predict1.utils.lazy_config import instantiate
from cosmos_predict1.utils.lazy_config.lazy import LazyConfig
from cosmos_predict1.utils.parallel_state_helper import is_tp_cp_pp_rank0

from dataset.spartan_video_dataset import SpartanVideoGEN3CDataset


def create_dataloader_config(config: Config) -> Dict[str, Any]:
    """Create dataloader configuration based on the YAML settings"""
    
    # Shape metadata for bimanual multi-view setup
    shape_meta = {
        "obs": {
            # Multi-view cameras (scene_1 through scene_12)
            **{f"scene_{i}": {"shape": [3, 256, 320], "type": "rgb"} for i in range(1, 13)},
            
            # Robot state information (actual poses)
            "robot__actual__poses__right::panda__xyz": {"shape": [3], "type": "low_dim"},
            "robot__actual__poses__right::panda__rot_6d": {"shape": [6], "type": "low_dim"},
            "robot__actual__poses__left::panda__xyz": {"shape": [3], "type": "low_dim"},
            "robot__actual__poses__left::panda__rot_6d": {"shape": [6], "type": "low_dim"},
            "robot__actual__grippers__right::panda_hand": {"shape": [1], "type": "low_dim"},
            "robot__actual__grippers__left::panda_hand": {"shape": [1], "type": "low_dim"},
            
            # Robot state information (desired poses)
            "robot__desired__poses__right::panda__xyz": {"shape": [3], "type": "low_dim"},
            "robot__desired__poses__right::panda__rot_6d": {"shape": [6], "type": "low_dim"},
            "robot__desired__poses__left::panda__xyz": {"shape": [3], "type": "low_dim"},
            "robot__desired__poses__left::panda__rot_6d": {"shape": [6], "type": "low_dim"},
            "robot__desired__grippers__right::panda_hand": {"shape": [1], "type": "low_dim"},
            "robot__desired__grippers__left::panda_hand": {"shape": [1], "type": "low_dim"},
        },
        "action": {"shape": [20]}  # (3+6+1)*2 for bimanual
    }
    
    # Dataset paths - UPDATE THESE PATHS AS NEEDED
    dataset_paths = [
        "/fsx-siro/liuzeyi/4dgen/data/BimanualPlaceAppleFromBowlIntoBin/2025-03-29T08-08-56+00-00/diffusion_spartan/episode_*",
        "/fsx-siro/liuzeyi/4dgen/data/BimanualPlaceAppleFromBowlIntoBin/2025-04-08T02-26-26+00-00/diffusion_spartan/episode_*",
        "/fsx-siro/liuzeyi/4dgen/data/BimanualPlaceAppleFromBowlIntoBin/2025-06-27T21-47-33+00-00/diffusion_spartan/episode_*",
    ]
    
    # Dataset configuration
    dataset_config = {
        "episode_path_globs": dataset_paths,
        "stride": 5,
        "horizon": getattr(config, 'horizon', 11),
        "pad_before": 0,
        "pad_after": 0,
        "shape_meta": shape_meta,
        "rotation_rep": "rotation_6d",
        "repeat_head": 0,
        "repeat_tail": 0,
        "has_gripper": False,
        "has_depth": True,
        "has_label": True,
        "val_ratio": 0.1,
        "imagenet_normalization": False,
        "mode": "zarr",
        "replay_buffer_path": "/fsx-siro/liuzeyi/4dgen/data/single_task/place_apple_from_bowl_into_bin/replay_buffer.zarr",
        "compressor": "blosc",
        "is_multiarm": True,
        "num_workers": 16,
        "raw_rgb": False,
        "apply_static_filter": True,
    }
    
    return dataset_config, shape_meta


def create_dataloaders(config: Config) -> tuple:
    """Create training and validation dataloaders"""
    
    dataset_config, shape_meta = create_dataloader_config(config)
    
    # Create the dataset
    full_dataset = SpartanVideoGEN3CDataset(**dataset_config)
    
    # Split dataset into train/val based on val_ratio
    val_ratio = dataset_config.get("val_ratio", 0.1)
    dataset_size = len(full_dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.trainer.seed)
    )
    
    # Training dataloader configuration
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=getattr(config, 'dataloader', {}).get('batch_size', 4),
        num_workers=getattr(config, 'dataloader', {}).get('num_workers', 16),
        shuffle=getattr(config, 'dataloader', {}).get('shuffle', True),
        pin_memory=getattr(config, 'dataloader', {}).get('pin_memory', True),
        persistent_workers=getattr(config, 'dataloader', {}).get('persistent_workers', True),
        drop_last=True,  # Important for consistent batch sizes in distributed training
    )
    
    # Validation dataloader configuration
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=getattr(config, 'val_dataloader', {}).get('batch_size', 4),
        num_workers=getattr(config, 'val_dataloader', {}).get('num_workers', 16),
        shuffle=getattr(config, 'val_dataloader', {}).get('shuffle', False),
        pin_memory=getattr(config, 'val_dataloader', {}).get('pin_memory', True),
        persistent_workers=getattr(config, 'val_dataloader', {}).get('persistent_workers', True),
        drop_last=False,
    )
    
    return train_dataloader, val_dataloader


@misc.timer("instantiate model")
def instantiate_model(config: Config, trainer) -> None:
    misc.set_random_seed(seed=config.trainer.seed, by_rank=False)
    config.model_obj.config = config.model
    if getattr(config.model, "fsdp_enabled", False):
        assert config.trainer.distributed_parallelism == "fsdp", "FSDP model is only supported with FSDP trainer"
        log.critical("FSDP enabled")
        config.model_obj.fsdp_checkpointer = trainer.checkpointer
        model = instantiate(config.model_obj)
        config.model_obj.fsdp_checkpointer = None
    else:
        model = instantiate(config.model_obj)
    config.model_obj.config = None
    misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
    return model


def destroy_distributed():
    log.info("Destroying distributed environment...")
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except ValueError as e:
            print(f"Error destroying default process group: {e}")


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    
    # Create the model
    model = instantiate_model(config, trainer)
    model.on_model_init_end()

    print(model)
    
    # Create the dataloaders using our custom function
    if args.mp0_only_dl:
        log.critical(
            "Using only tp_cp_pp_rank0 dataloader for faster dataloading! Make sure val dl is mock and mock data has same keys as real data."
        )
        raise NotImplementedError(
            "mp0_only_dl is not implemented correctly! Please revisit this code and propose a more robust impl that raise error timely! It does not do necessary check before training to confirm it can work with image / video data. Current impl is problematic for image training."
        )
    if is_tp_cp_pp_rank0() or not args.mp0_only_dl:
        dataloader_train, dataloader_val = create_dataloaders(config)
        log.info(f"Created training dataloader with {len(dataloader_train)} batches")
        log.info(f"Created validation dataloader with {len(dataloader_val)} batches")
    else:
        # Use validation dataloader for non-rank0 processes when mp0_only_dl is enabled
        _, dataloader_train = create_dataloaders(config)
        dataloader_train, dataloader_val = create_dataloaders(config)
    
    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )
    destroy_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--config",
        default="cosmos_predict1/diffusion/posttrain/config/config.py",
        help="Path to the config file",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    parser.add_argument(
        "--mp0_only_dl",
        action="store_true",
        help="Use only model parallel rank 0 dataloader for faster dataloading! Make sure mock data has same keys as real data.",
    )
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    if args.dryrun:
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(OmegaConf.to_yaml(OmegaConf.load(f"{config.job.path_local}/config.yaml")))
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
