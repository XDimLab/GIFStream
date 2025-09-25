import json
import math
import os
import time
import shutil
from contextlib import nullcontext
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, ContextManager

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.GIFStream_new import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from gsplat.compression_simulation.ops import fake_quantize_factors
from utils import CameraEmbedding, knn, set_random_seed, find_k_neighbors
import random

from gsplat.compression import GIFStreamEnd2endCompression, GIFStream2dcodecCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization, view_to_visible_anchors
from gsplat.strategy import GIFStreamStrategy

from gsplat.compression_simulation.simulation import GIFStreamCompressionSimulation

class ProfilerConfig:
    def __init__(self):
        self.enabled = False
        self.activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        self.wait = 1 
        self.warmup = 2 
        self.active = 30_000 
        
        self.schedule = self._create_schedule()
        
        self.on_trace_ready = torch.profiler.tensorboard_trace_handler('./log/profiler')
        self.record_shapes = True
        self.profile_memory = True
        self.with_stack = True
    
    def _create_schedule(self):
        return torch.profiler.schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
        )
    
    def update_schedule(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.schedule = self._create_schedule()

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["end2end", "2dcodec"]] = None
    # Quantization parameters when set to hevc
    qp: Optional[int] = None

    # Enable profiler
    profiler_enabled: bool = False

    # Enable compression simulation
    compression_sim: bool = False

    # Enable entropy model
    entropy_model_opt: bool = False
    # Define the type of entropy model
    entropy_model_type: Literal["conditional_gaussian_model"] = "conditional_gaussian_model"
    # Bit-rate distortion trade-off parameter
    rd_lambda: float = 5e-4 # default: 1e-2
    # Steps to enable entropy model into training pipeline
    # conditional gaussian model:
    entropy_steps: Dict[str, int] = field(default_factory=lambda: {"anchors": -1, 
                                                                   "quats": 10_000, 
                                                                   "scales": 10_000, 
                                                                   "opacities": 10_000, 
                                                                   "anchor_features": 10_000, 
                                                                   "offsets": 10_000,
                                                                   "factors": 10_000,
                                                                   "time_features": 10_000,
                                                                   })

    
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera camera_modelmodel
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: GIFStreamStrategy = field(
        default_factory=GIFStreamStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Scale regularization
    scale_reg: float = 0.01

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 12
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # Dimensionality of anchor features
    anchor_feature_dim: int = 24
    # Dimensionality of entropy model predicting the distribution of anchor features
    entropy_channel: int = 8
    # Number offsets
    n_offsets: int = 5
    # voxel size for Scaffold-GS
    voxel_size = 0.01
    # whether add dist for neural gaussaian decoding mlps
    add_opacity_dist = False
    add_cov_dist = False
    add_cov_dist = False
    add_color_dist = False

    # GIFStream setup
    # Dimensionality of time-dependent feature per frame
    c_perframe: int = 8
    # GOP size for training
    GOP_size: int = 50
    # number of anchors for feature aggregation
    knn: bool = False
    n_knn: int = 6
    # time positional embedding dim, must be even
    time_dim: int = 16
    # time positional embedding base
    phi: float = 2
    # whether add view to neural gaussian decoding mlps
    view_adaptive: bool = False
    # test view cameras
    test_set: Optional[List[int]] = None
    # filter out cameras
    remove_set: Optional[List[int]] = None
    # regularization lambda
    factor_reg: float = 0.005
    smooth_reg: float = 0.005
    # GOP start frame
    start_frame: int = 0
    # whether continue from an existing ckpt
    continue_training: bool = False
    # rate number
    rate: int = 0
    # quantization scalings
    compression_scaling = [
        {
            "anchors": None,
            "scales": 0.02,
            "quats": None,
            "opacities": None,
            "anchor_features": 1,
            "offsets": 0.02,
            "factors": 1/16,
            "time_features": 1,
        },
        {
            "anchors": None,
            "scales": 0.04,
            "quats": None,
            "opacities": None,
            "anchor_features": 1,
            "offsets": 0.04,
            "factors": 1/16,
            "time_features": 1,
        },
        {
            "anchors": None,
            "scales": 0.06,
            "quats": None,
            "opacities": None,
            "anchor_features": 1.5,
            "offsets": 0.06,
            "factors": 1/16,
            "time_features": 1.5,
        },
        {
            "anchors": None,
            "scales": 0.08,
            "quats": None,
            "opacities": None,
            "anchor_features": 2,
            "offsets": 0.08,
            "factors": 1/16,
            "time_features": 2,
        },
    ]

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)

        strategy = self.strategy
        if isinstance(strategy, GIFStreamStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    app_embed_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    voxel_size: int = 0.001,
    anchor_feature_dim: int = 48,
    n_offsets: int = 5,
    use_feat_bank: bool = False,
    add_opacity_dist: bool = False,
    add_cov_dist: bool = False,
    add_color_dist: bool = False,
    c_perframe: int = 4,
    GOP_size: int = 50,
    n_knn: int = 8,
    time_dim: int = 16,
    view_adaptive: bool = False
) -> Tuple[torch.nn.ParameterDict, torch.nn.ModuleDict, Dict[str, torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = parser.points
        np.random.shuffle(points)
        points = torch.from_numpy(np.unique(np.round(points/voxel_size), axis=0)*voxel_size).float()
    else:
        raise ValueError("Now only Support SFM Initialization")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 6)  # [N, 6]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.zeros((N, 4))  # [N, 4]
    quats[:,0] = 1
    opacities = torch.logit(torch.full((N,1), init_opacity))  # [N,]
    anchor_features = torch.zeros((N, anchor_feature_dim))
    offsets = torch.zeros((N, n_offsets, 3))
    time_features = torch.zeros((N, GOP_size, c_perframe))
    factors = torch.zeros((N, 4)) # [time_feature factor, motion_factor, knn_factor, pruning_factor]
    
    params = [
        # name, value, lr
        ("anchors", torch.nn.Parameter(points), 0),
        ("scales", torch.nn.Parameter(scales.requires_grad_(True)), 7e-3),
        ("quats", torch.nn.Parameter(quats), 0),
        ("opacities", torch.nn.Parameter(opacities), 0),
        ("offsets", torch.nn.Parameter(offsets.requires_grad_(True)), 1e-2),
        ("anchor_features", torch.nn.Parameter(anchor_features.requires_grad_(True)), 0.0075),
        ("time_features", torch.nn.Parameter(time_features.requires_grad_(True)), 0.0075),
        ("factors", torch.nn.Parameter(factors.requires_grad_(True)), 1e-3),
    ]

    view_dim = 3 if view_adaptive else 0
    opacity_dist_dim = 1 if add_opacity_dist else 0
    mlp_opacity = torch.nn.Sequential(
        torch.nn.Linear(anchor_feature_dim+view_dim+opacity_dist_dim+c_perframe, anchor_feature_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(anchor_feature_dim, n_offsets),
        torch.nn.Tanh()
    ).to(device)

    cov_dist_dim = 1 if add_cov_dist else 0
    mlp_cov = torch.nn.Sequential(
        torch.nn.Linear(anchor_feature_dim+view_dim+cov_dist_dim+c_perframe, anchor_feature_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(anchor_feature_dim, 7*n_offsets),
    ).to(device)


    color_dist_dim = 1 if add_color_dist else 0
    mlp_color = torch.nn.Sequential(
        torch.nn.Linear(anchor_feature_dim+view_dim+color_dist_dim+app_embed_dim+c_perframe, anchor_feature_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(anchor_feature_dim, 3*n_offsets),
        torch.nn.Sigmoid()
    ).to(device)

    # deformation net
    mlp_motion = torch.nn.Sequential(
        torch.nn.Linear(anchor_feature_dim+time_dim+c_perframe, anchor_feature_dim),
        torch.nn.ReLU(True),
        torch.nn.Linear(anchor_feature_dim, 3+4),
    ).to(device)
    torch.nn.init.constant_(mlp_motion[-1].weight,0)
    torch.nn.init.constant_(mlp_motion[-1].bias,0)

    net_params = [
        # name, value, lr
        ("mlp_opacity", mlp_opacity, 2e-3),
        ("mlp_cov", mlp_cov, 4e-3),
        ("mlp_color", mlp_color, 8e-3),
        ("mlp_motion", mlp_motion, 8e-3),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    decoders = torch.nn.ModuleDict({n: v for n, v, _ in net_params}).to(device)

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = torch.optim.Adam

    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    decoder_optimizers = {
        name: optimizer_class(
            [{"params": decoders[name].parameters(), "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in net_params
    }
    return splats, decoders, optimizers, decoder_optimizers

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)
        print("results will be saved to ", cfg.result_dir)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            first_frame=cfg.start_frame,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=False,
            test_set=cfg.test_set,
            remove_set=cfg.remove_set,
            GOP_size=cfg.GOP_size,
            start_frame=cfg.start_frame,
        )
        self.valset = Dataset(self.parser, split="val", test_set=cfg.test_set, remove_set=cfg.remove_set, GOP_size=cfg.GOP_size, start_frame=cfg.start_frame)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        app_embed_dim = cfg.app_embed_dim if cfg.app_opt else 0
        self.splats,self.decoders, self.optimizers, self.net_optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=None,
            sparse_grad=False,
            visible_adam=False,
            batch_size=cfg.batch_size,
            app_embed_dim=app_embed_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            voxel_size=cfg.voxel_size,
            anchor_feature_dim=cfg.anchor_feature_dim,
            n_offsets=cfg.n_offsets,
            use_feat_bank=False,
            add_opacity_dist=cfg.add_opacity_dist,
            add_cov_dist=cfg.add_cov_dist,
            add_color_dist=cfg.add_color_dist,
            c_perframe=cfg.c_perframe,
            GOP_size=cfg.GOP_size,
            n_knn=cfg.n_knn,
            time_dim=cfg.time_dim,
            view_adaptive=cfg.view_adaptive
        )
        print("Model initialized. Number of Anchor:", len(self.splats["anchors"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, GIFStreamStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale,
                n_offsets=cfg.n_offsets,
                voxel_size=cfg.voxel_size,
                anchor_feature_dim=cfg.anchor_feature_dim
            )
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "end2end":
                self.compression_method = GIFStreamEnd2endCompression()
            elif cfg.compression == "2dcodec":
                self.compression_method = GIFStream2dcodecCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")
        
        if cfg.compression_sim:
            cap_max = cfg.strategy.cap_max if cfg.strategy.cap_max is not None else None
            self.compression_sim_method = GIFStreamCompressionSimulation(cfg.entropy_model_opt,
                                                    cfg.entropy_model_type,
                                                    cfg.entropy_steps,
                                                    self.device,
                                                    False,
                                                    None,
                                                    None,
                                                    cap_max=cap_max,
                                                    feature_dim=cfg.anchor_feature_dim,
                                                    n_offsets=cfg.n_offsets,
                                                    c_channel=self.cfg.entropy_channel,
                                                    p_channel=self.cfg.c_perframe,
                                                    scaling=self.cfg.compression_scaling[self.cfg.rate],
                                                    max_steps=self.cfg.max_steps)

            if cfg.entropy_model_opt:
                selected_key = min((k for k, v in cfg.entropy_steps.items() if v > 0), key=lambda k: cfg.entropy_steps[k])
                self.entropy_min_step = cfg.entropy_steps[selected_key]
        
        # Profiler
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiler_config = ProfilerConfig()
        if cfg.profiler_enabled:
            self.profiler_config.enabled = True

        self.app_optimizers = []
        if cfg.app_opt:
            assert cfg.app_embed_dim > 0
            self.app_module = CameraEmbedding(
                self.trainset.cameras_length, cfg.app_embed_dim
            ).to(self.device)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=False
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )
        if self.cfg.knn:
            self.indices = None
        self.istraining = False

    def init_dynamic(self) -> None:
        grads = self.strategy_state["offset_grad2d"] / self.strategy_state["offset_demon"]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1).view((-1,self.cfg.n_offsets)).mean(dim=-1)
        mini = grads_norm.min()
        maxi = grads_norm.max()
        grads_norm = ((grads_norm - mini) / (maxi - mini + 1e-6)).clamp(0.15, 1)
        grads_norm = - torch.log(1/grads_norm - 1)
        self.splats["factors"].data[:,1] = grads_norm

    def get_profiler(self, tb_writer) -> ContextManager:
        if self.profiler_config.enabled:
            return torch.profiler.profile(
                activities=self.profiler_config.activities,
                schedule=self.profiler_config.schedule,
                # on_trace_ready=self.profiler_config.on_trace_ready, 
                on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_writer.log_dir),
                record_shapes=self.profiler_config.record_shapes,
                profile_memory=self.profiler_config.profile_memory,
                with_stack=self.profiler_config.with_stack
            )
        return nullcontext()

    def step_profiler(self):
        """step profiler"""
        if self.profiler is not None:
            self.profiler.step()

    def decoding_features(self,
        camtoworlds: Tensor,
        time: float,
        visible_anchor_mask: Tensor,
        canonical: bool = False,
        step: int = -1,
        camera_ids: Tensor = None,
    )-> Dict:
        feat_start = int(time * (self.cfg.GOP_size-1))
        # coarse to fine training for time-dependent features
        if step > 0 and self.istraining:
            gap = int((self.cfg.GOP_size // 5) * (1 - min(1, 5 * step / (self.cfg.max_steps - 1))))
            pre = max(0, feat_start - gap)
            aft = min(self.cfg.GOP_size, feat_start+gap+1)
        else:
            pre = feat_start
            aft = feat_start + 1

        # consider dynamic gaussians which may be unselected (not visible in canonical space)
        # if step == -1 or step > self.cfg.max_steps // 6:
        #     visible_anchor_mask = torch.logical_or(visible_anchor_mask, torch.sigmoid(self.splats["factors"][:,1]) > 0.2 )

        if not self.cfg.compression_sim:
            selected_features = self.splats["anchor_features"][visible_anchor_mask]  # [M, c]
            selected_anchors = self.splats["anchors"][visible_anchor_mask]  # [M, 3]
            selected_scales = torch.exp(self.splats["scales"][visible_anchor_mask])  # [M, 6]
            selected_time_features = self.splats["time_features"][visible_anchor_mask][:,pre:aft].mean(dim=1) if aft - pre >1 else self.splats["time_features"][visible_anchor_mask][:,feat_start]# [M,T,C]
            factors = fake_quantize_factors(self.splats["factors"], q_aware=False)
            selected_factors = factors[visible_anchor_mask]
            if self.cfg.knn:
                if self.indices is None or self.indices.shape[0] != self.splats["anchors"].shape[0]:
                    _, self.indices = find_k_neighbors(self.splats["anchors"], self.cfg.n_knn)
                selected_indices = self.indices[visible_anchor_mask].reshape(-1)
                knn_features = self.splats["anchor_features"][selected_indices].reshape(-1,self.cfg.n_knn,self.cfg.anchor_feature_dim).mean(dim=1)
                knn_time_features = (
                    self.splats["time_features"][:,feat_start] * 
                    (factors[:,0].unsqueeze(-1) if not canonical else 0)
                )[selected_indices].reshape(-1,self.cfg.n_knn,self.cfg.c_perframe).mean(dim=1)
        else:
            selected_features = self.comp_sim_splats["anchor_features"][visible_anchor_mask]  # [M, c]
            selected_anchors = self.comp_sim_splats["anchors"][visible_anchor_mask]  # [M, 3]
            selected_scales = torch.exp(self.comp_sim_splats["scales"][visible_anchor_mask])  # [M, 6]
            selected_factors = self.comp_sim_splats["factors"][visible_anchor_mask] # [M,4]
            selected_time_features = self.comp_sim_splats["time_features"][visible_anchor_mask][:,pre:aft].mean(dim=1) if aft - pre > 1 else self.comp_sim_splats["time_features"][visible_anchor_mask][:,feat_start] 

            if self.cfg.knn:
                if self.indices is None:
                    _, self.indices = find_k_neighbors(self.splats["anchors"], self.cfg.n_knn)
                selected_indices = self.indices[visible_anchor_mask].reshape(-1)
                knn_features = self.comp_sim_splats["anchor_features"][selected_indices].reshape(-1,self.cfg.n_knn,selected_features.shape[-1]).mean(dim=1)
                knn_time_features = (
                    self.comp_sim_splats["time_features"][:,feat_start] * (
                    self.comp_sim_splats["factors"][:,0].unsqueeze(-1) if not canonical else 0)
                )[selected_indices].reshape(-1, self.cfg.n_knn, self.cfg.c_perframe).mean(dim=1)

        cam_pos = camtoworlds[:, :3, 3]
        view_dir = selected_anchors - cam_pos  
        length = view_dir.norm(dim=1, keepdim=True)
        view_dir_normalized = view_dir / length  

        if self.cfg.view_adaptive:
            feature_view_dir = torch.cat([selected_features, view_dir_normalized], dim=1)
        else:
            feature_view_dir = selected_features
        
        if self.cfg.knn:
            knn_feature_view_dir = knn_features

        i = torch.ones((1),dtype=torch.float32)
        time_embedding = torch.cat(
            [torch.sin(self.cfg.phi**n * torch.pi * i * time) for n in range(self.cfg.time_dim // 2)] + 
            [torch.cos(self.cfg.phi**n * torch.pi * i * time) for n in range(self.cfg.time_dim // 2)]
        ).to(self.splats["anchors"].device)

        time_feature_factor = selected_factors[:,0].unsqueeze(-1)
        motion_factor = selected_factors[:,1].unsqueeze(-1)
        knn_factor = selected_factors[:,2].unsqueeze(-1)
        pruning_factor = selected_factors[:,3].unsqueeze(-1)

        selected_scales = torch.cat([selected_scales[:,:3], selected_scales[:,3:] * pruning_factor],dim=-1)
        if canonical:
            time_feature_factor = 0
            motion_factor = 0
            knn_factor = 0.5
        if self.cfg.knn:
            time_adaptive_features = torch.cat([
                feature_view_dir, 
                selected_time_features * time_feature_factor
            ],dim=-1)
            time_adaptive_features_ = knn_factor * torch.cat([
                selected_features, 
                selected_time_features * time_feature_factor
            ],dim=-1) + (1 - knn_factor) * torch.cat([knn_feature_view_dir, knn_time_features],dim=-1)
        else:
            time_adaptive_features = torch.cat([
                feature_view_dir, 
                selected_time_features * time_feature_factor
            ],dim=-1)
            time_adaptive_features_ = torch.cat([
                selected_features, 
                selected_time_features * time_feature_factor
            ],dim=-1)
        time_adaptive_features_ = torch.cat([time_adaptive_features_, time_embedding.unsqueeze(0).expand((time_adaptive_features.shape[0],-1))],dim=1)


        k = self.cfg.n_offsets  # Number of offsets per anchor

        # Apply MLPs
        neural_opacity = self.decoders["mlp_opacity"](
            time_adaptive_features
        )
        neural_opacity = neural_opacity.view(-1, 1) * pruning_factor.view(-1,1).expand((-1,k)).reshape((-1,1)) 

        # Get color
        neural_colors = self.decoders["mlp_color"](
            torch.cat([time_adaptive_features, self.app_module(camera_ids).to(self.device).view((1,-1)).expand(time_adaptive_features.shape[0],-1)],dim=-1) if self.cfg.app_opt else time_adaptive_features
        )
        neural_colors = neural_colors.view(-1, 3)  # [M*k, 3]

        # Get scale and rotation
        neural_scale_rot = self.decoders["mlp_cov"](
            time_adaptive_features
        )
        neural_scale_rot = neural_scale_rot.view(-1, 7)  # [M*k, 7]

        # Get anchor motion
        motion = self.decoders["mlp_motion"](
            time_adaptive_features_
        )  
        motion = motion * motion_factor

        return {
            "neural_opacity":neural_opacity,
            "neural_colors":neural_colors,
            "neural_scale_rot":neural_scale_rot,
            "motion":motion,
            "selected_factors":selected_factors,
            "selected_scales":selected_scales,
        }

    def get_neural_gaussians(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        packed: bool,
        rasterize_mode: str,
        time: float,
        canonical: bool = False,
        regular: bool = False,
        step: int = -1,
        camera_ids: Tensor = None,
    ) -> Dict:
        """
        Compute the neural Gaussian parameters for the current view and time.

        Args:
            camtoworlds (Tensor): Camera-to-world transformation matrices, shape [C, 4, 4].
            Ks (Tensor): Camera intrinsic matrices, shape [C, 3, 3].
            width (int): Image width.
            height (int): Image height.
            packed (bool): Whether to use packed mode for rasterization.
            rasterize_mode (str): Rasterization mode (e.g., 'classic', 'antialiased').
            time (float): Normalized time in [0, 1] for dynamic scenes.
            canonical (bool, optional): Whether to use canonical (static) mode. Defaults to False.
            regular (bool, optional): Whether to compute regularization loss. Defaults to False.
            step (int, optional): Current training step. Defaults to -1.
            camera_ids (Tensor, optional): Camera IDs for appearance embedding. Defaults to None.

        Returns:
            Dict: A dictionary containing the parameters of visible neural Gaussians, including means, colors, opacities, scales, rotations, and auxiliary losses.
        """
        # Compute which anchors (Gaussians) are visible in the current view
        visible_anchor_mask = view_to_visible_anchors(
            means=self.splats["anchors"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"][:, :3]),
            viewmats=torch.linalg.inv(camtoworlds), 
            Ks=Ks,
            width=width,
            height=height,
            packed=packed,
            rasterize_mode=rasterize_mode,
        )

        # Select anchors and offsets for visible Gaussians
        if not self.cfg.compression_sim:
            selected_anchors = self.splats["anchors"][visible_anchor_mask]  # [M, 3]
            selected_offsets = self.splats["offsets"][visible_anchor_mask]  # [M, k, 3]
        else:
            selected_anchors = self.comp_sim_splats["anchors"][visible_anchor_mask]  # [M, 3]
            selected_offsets = self.comp_sim_splats["offsets"][visible_anchor_mask]  # [M, k, 3]

        # Decode neural features (opacity, color, scale/rotation, motion, etc.)
        results = self.decoding_features(
            camtoworlds,
            time,
            visible_anchor_mask,
            canonical,
            step,
            camera_ids,
        )
        # Compute smoothness loss by comparing with a nearby time step (every x steps)
        if not canonical and step % 4 == 0 and self.istraining:
            with torch.no_grad():
                idx_dif = random.choice([-2,-1,1,2])
                results_ = self.decoding_features(
                    camtoworlds,
                    torch.tensor(time + idx_dif / (self.cfg.GOP_size - 1)).clamp(0,1).item(),
                    visible_anchor_mask,
                    canonical,
                    step,
                    camera_ids,
                )
            item_list = ["neural_opacity", "neural_colors", "neural_scale_rot", "motion"]
            smooth_loss = sum([torch.abs(results[k] - results_[k]).mean() for k in results.keys() if (k in item_list)])
        else:
            smooth_loss = 0

        # Unpack decoded features
        neural_opacity = results["neural_opacity"]
        neural_colors = results["neural_colors"]
        neural_scale_rot = results["neural_scale_rot"]
        motion = results["motion"]
        selected_scales = results["selected_scales"]
        selected_factors = results["selected_factors"]
        
        # Mask out Gaussians with non-positive opacity (they do not contribute to rendering)
        neural_selection_mask = (neural_opacity > 0.0).view(-1)  # [M*k]
        # Apply motion offset to anchor positions
        anchor_offset = motion[:,-7:-4]
        selected_anchors += anchor_offset
        # Compute anchor rotation from motion output (as quaternion)
        anchor_rot = torch.nn.functional.normalize(0.1 * motion[:,-4:] + torch.tensor([[1,0,0,0]],device="cuda"))
        anchor_rotation = quaternion_to_rotation_matrix(anchor_rot)
        # Transform offsets by scale and rotation
        selected_offsets = torch.bmm(selected_offsets.view(-1,self.cfg.n_offsets,3) * selected_scales.unsqueeze(1)[:,:,:3] ,anchor_rotation.reshape((-1,3,3)).transpose(1, 2)).reshape((-1,3))
        # Repeat scales and anchors for each offset
        scales_repeated = (selected_scales.unsqueeze(1).repeat(1, self.cfg.n_offsets, 1).view(-1, 6))  # [M*k, 6]
        anchors_repeated = (selected_anchors.unsqueeze(1).repeat(1, self.cfg.n_offsets, 1).view(-1, 3))  # [M*k, 3]
        # Combine neural and anchor rotations
        # neural_scale_rot = torch.cat([neural_scale_rot[:,:3],quaternion_multiply(anchor_rot.unsqueeze(1).expand([-1,self.cfg.n_offsets,-1]).flatten(0,1), neural_scale_rot[:, 3:7])],dim=-1)
        
        # Apply mask to select valid Gaussians
        selected_opacity = neural_opacity[neural_selection_mask].squeeze(-1)  # [M]
        selected_colors = neural_colors[neural_selection_mask]  # [M, 3]
        selected_scale_rot = neural_scale_rot[neural_selection_mask]  # [M, 7]
        selected_offsets = selected_offsets[neural_selection_mask]  # [M, 3]
        scales_repeated = scales_repeated[neural_selection_mask]  # [M, 6]
        anchors_repeated = anchors_repeated[neural_selection_mask]  # [M, 3]

        # Compute final scales and rotations
        scales = scales_repeated[:, 3:] * torch.sigmoid(selected_scale_rot[:, :3])
        rotation = torch.nn.functional.normalize(selected_scale_rot[:, 3:7])

        # Compute final means (positions) of Gaussians
        offsets = selected_offsets  # [M, 3]
        means = anchors_repeated + offsets  # [M, 3]

        info = {
            "means": means,  # Final positions of Gaussians
            "colors": selected_colors,  # RGB colors
            "opacities": selected_opacity,  # Opacity values
            "scales": scales,  # Scale parameters
            "quats": rotation,  # Rotation as quaternion
            "neural_opacity": neural_opacity,  # All predicted opacities
            "neural_selection_mask": neural_selection_mask,  # Mask for valid Gaussians
            "anchor_visible_mask": visible_anchor_mask,  # Mask for visible anchors
            "reg_loss": selected_factors[:,:-1].mean() + 0.1 * selected_factors[:,-1].mean() if regular else 0,  # Regularization loss
            "smooth_loss": smooth_loss,# Smoothness loss
            "motion": anchor_offset,  # Anchor offset
        }
        return info

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        time: float = 0.,
        canonical: bool = False,
        regular: bool = False,
        step: int = -1,
        camera_ids: Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        neural_gaussians = self.get_neural_gaussians(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            rasterize_mode="antialiased" if self.cfg.antialiased else "classic",
            time=time,
            canonical=canonical,
            regular=regular,
            step=step,
            camera_ids=camera_ids,
        )
        
        means = neural_gaussians["means"]  # [N, 3]
        quats = neural_gaussians["quats"]  # [N, 4]
        scales = neural_gaussians["scales"]  # [N, 3]
        opacities = neural_gaussians["opacities"]  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        
        colors = neural_gaussians["colors"]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, GIFStreamStrategy)
                else False
            ),
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        info["anchor_visible_mask"] = neural_gaussians["anchor_visible_mask"]
        info["neural_selection_mask"] = neural_gaussians["neural_selection_mask"]
        info["update_filter"] = info["radii"] > 0
        info["scales"] = neural_gaussians["scales"]
        info["neural_opacity"] = neural_gaussians["neural_opacity"]
        info["reg_loss"] = neural_gaussians["reg_loss"]
        info["smooth_loss"] = neural_gaussians["smooth_loss"]
        info["gop"] = self.cfg.GOP_size
        info["time"] = int(time * (self.cfg.GOP_size - 1))
        info["motion"] = neural_gaussians["motion"]
        return render_colors, render_alphas, info

    def train(self, init_step: int=0):
        self.istraining = True
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = init_step

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["offsets"], gamma=0.03 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.net_optimizers["mlp_opacity"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.net_optimizers["mlp_color"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.net_optimizers["mlp_motion"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        self.decoders["mlp_opacity"].train()
        self.decoders["mlp_cov"].train()
        self.decoders["mlp_color"].train()
        self.decoders["mlp_motion"].train()

        with self.get_profiler(self.writer) as prof:
            self.profiler = prof if self.profiler_config.enabled else None

            # Training loop.
            global_tic = time.time()
            pbar = tqdm.tqdm(range(init_step, max_steps))
            for step in pbar:
                if not cfg.disable_viewer:
                    while self.viewer.state.status == "paused":
                        time.sleep(0.01)
                    self.viewer.lock.acquire()
                    tic = time.time()

                try:
                    batch_data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    batch_data = next(trainloader_iter)
                if step == int(max_steps * self.cfg.strategy.deformation_gate):
                    self.init_dynamic()
                
                #* batch forward
                info_list = []
                for batch_ind in range(self.cfg.batch_size):
                    if batch_ind >= batch_data["camtoworld"].shape[0]:
                        info_list.append(None)
                        continue
                    else:
                        data = {}
                        for k,v in batch_data.items():
                            data[k] = v[batch_ind].unsqueeze(0)
                    camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
                    Ks = data["K"].to(device)  # [1, 3, 3]
                    pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
                    num_train_rays_per_step = (
                        pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                    )
                    image_ids = data["image_id"].to(device)
                    masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
                    camera_ids = data["camera_id"].to(device)

                    height, width = pixels.shape[1:3]


                    # sh schedule
                    sh_degree_to_use = None

                    # compression simulation
                    if cfg.compression_sim and cfg.entropy_model_opt and cfg.entropy_model_type == "gaussian_model": # if hash-based gaussian model, need to estiblish bbox
                        if step == self.entropy_min_step:
                            self.compression_sim_method._estiblish_bbox(self.splats["means"])

                    if cfg.compression_sim:
                        self.comp_sim_splats, self.esti_bits_dict = self.compression_sim_method.simulate_compression(self.splats, step, int(float(data["time"]) * (self.cfg.GOP_size - 1)), self.cfg.entropy_channel)

                    # forward
                    renders, alphas, info = self.rasterize_splats(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=image_ids,
                        render_mode="RGB",
                        masks=masks,
                        time=float(data["time"]),
                        canonical= (step <= int(max_steps * self.cfg.strategy.deformation_gate)),
                        regular= (step > int(max_steps * (self.cfg.strategy.deformation_gate + 0.1))),
                        step=step,
                        camera_ids=camera_ids,
                    )
                    if renders.shape[-1] == 4:
                        colors, depths = renders[..., 0:3], renders[..., 3:4]
                    else:
                        colors, depths = renders, None


                    if cfg.random_bkgd:
                        bkgd = torch.rand(1, 3, device=device)
                        colors = colors + bkgd * (1.0 - alphas)

                    self.cfg.strategy.step_pre_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                    )

                    # loss
                    l1loss = F.l1_loss(colors, pixels)
                    ssimloss = 1.0 - fused_ssim(
                        colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
                    )
                    loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda + info["scales"].prod(dim=1).mean() * cfg.scale_reg + info["reg_loss"] * cfg.factor_reg + info["smooth_loss"] * cfg.smooth_reg
                    scale_loss = info["scales"].prod(dim=1).mean()
                    reg_loss = info["reg_loss"]
                    smooth_loss = info["smooth_loss"]
                    # entropy constraint
                    if cfg.entropy_model_opt and step>self.entropy_min_step:
                        total_esti_bits = 0
                        for n, n_step in cfg.entropy_steps.items():
                            if step > n_step and self.esti_bits_dict[n] is not None:
                                # maybe give different params with different weights
                                total_esti_bits += torch.sum(self.esti_bits_dict[n]) / self.esti_bits_dict[n].numel() # bpp
                            else:
                                total_esti_bits += 0

                        loss = (
                            loss
                            + cfg.rd_lambda * total_esti_bits
                        )
                    
                    # tmp workaround
                    loss_show = loss.detach().cpu()
                    loss.backward()
                    info_list.append(info)
                
                desc = f"loss={loss_show.item():.3f}| " f"sh degree={sh_degree_to_use}| "
                pbar.set_description(desc)

                # tensorboard monitor
                if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    self.writer.add_scalar("train/loss", loss.item(), step)
                    self.writer.add_scalar("train/scale_loss", scale_loss.item(), step)
                    self.writer.add_scalar("train/reg_loss", reg_loss.item() if reg_loss>0 else reg_loss, step)
                    self.writer.add_scalar("train/smooth_loss", smooth_loss.item() if smooth_loss>0 else smooth_loss, step)
                    self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                    self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                    self.writer.add_scalar("train/num_anchor", len(self.splats["anchors"]), step)
                    self.writer.add_scalar("train/mem", mem, step)
                    if self.cfg.compression_sim:
                        self.writer.add_scalar("train/dynamic", (self.comp_sim_splats["factors"][:,0] > 0).to(torch.float32).mean(), step)
                        self.writer.add_scalar("train/dynamic_", torch.logical_or((self.comp_sim_splats["factors"][:,0] > 0),(self.comp_sim_splats["factors"][:,1] > 0)).to(torch.float32).mean(), step)
                        self.writer.add_scalar("train/pruning", (self.comp_sim_splats["factors"][:,-1] > 0).to(torch.float32).mean(), step)
                    if cfg.tb_save_image:
                        canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                        canvas = canvas.reshape(-1, *canvas.shape[2:])
                        self.writer.add_image("train/render", canvas, step)
                    if cfg.compression_sim:
                        if cfg.entropy_model_opt and step>self.entropy_min_step:
                            self.writer.add_histogram("train_hist/quats", self.splats["quats"], step)
                            self.writer.add_histogram("train_hist/scales", self.splats["scales"], step)
                            self.writer.add_histogram("train_hist/anchor_features", self.splats["anchor_features"], step)
                            self.writer.add_histogram("train_hist/offsets", self.splats["offsets"], step)
                            self.writer.add_histogram("train_hist/factors", self.splats["factors"], step)
                            if total_esti_bits > 0:
                                self.writer.add_scalar("train/bpp_loss", total_esti_bits.item(), step)
                        
                    self.writer.add_histogram("train_hist/means", self.splats["anchors"], step)
                    self.writer.flush()

                # save checkpoint before updating the model
                if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    stats = {
                        "mem": mem,
                        "ellipse_time": time.time() - global_tic,
                        "num_GS": len(self.splats["anchors"]),
                    }
                    print("Step: ", step, stats)
                    with open(
                        f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                        "w",
                    ) as f:
                        json.dump(stats, f)
                    
                    # prepare data to be saved
                    data = {"step": step, "splats": self.splats.state_dict(), "decoders": self.decoders.state_dict()}
                    if cfg.app_opt:
                        if world_size > 1:
                            data["app_module"] = self.app_module.module.state_dict()
                        else:
                            data["app_module"] = self.app_module.state_dict()

                    if cfg.compression_sim and cfg.entropy_model_opt:
                        self.entropy_models = self.compression_sim_method.entropy_models
                        for name, entropy_model in self.compression_sim_method.entropy_models.items():
                            if entropy_model is not None:
                                data[name+"_entropy_model"] = entropy_model.state_dict()
                    data["compression_sim"] = self.cfg.compression_sim
                    data["scaling"] = self.compression_sim_method.scaling
                    torch.save(
                        data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                    )

                # optimize
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.net_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.app_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.bil_grid_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()
                # (optional) entropy model params. optimize
                if cfg.compression_sim:
                    if cfg.entropy_model_opt:
                        for name, optimizer in self.compression_sim_method.entropy_model_optimizers.items():
                            if optimizer is not None:
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                        for name, scheduler in self.compression_sim_method.entropy_model_schedulers.items():
                            if scheduler is not None and step > cfg.entropy_steps[name]:
                                scheduler.step()

                # Run post-backward steps after backward and optimizer
                if isinstance(self.cfg.strategy, GIFStreamStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info_list,
                        packed=cfg.packed,
                        mask=(self.comp_sim_splats["factors"][:,-1] == 0) if self.cfg.compression_sim else None,
                        max_steps=self.cfg.max_steps
                    )
                else:
                    assert_never(self.cfg.strategy)

                if (
                    step > self.cfg.strategy.refine_start_iter
                    and step % self.cfg.strategy.refine_every == 0
                    and self.cfg.knn
                ):
                    _, self.indices = find_k_neighbors(self.splats["anchors"], self.cfg.n_knn)

                self.step_profiler()

                # eval the full set
                if step in [i - 1 for i in cfg.eval_steps]:
                    self.eval(step)
                    self.render_traj(step)

                # run compression
                if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                    self.run_compression(step=step)

                if not cfg.disable_viewer:
                    self.viewer.lock.release()
                    num_train_steps_per_sec = 1.0 / (time.time() - tic)
                    num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                    )
                    # Update the viewer state.
                    self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                    # Update the scene.
                    self.viewer.update(step, num_train_rays_per_step)
        self.istraining = False
        

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        training_state = self.istraining
        self.istraining = False
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            camera_ids = data["camera_id"].to(device)
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=None,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                time=float(data["time"]),
                camera_ids=camera_ids,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images 
                # canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() # side by side
                canvas = canvas_list[1].squeeze(0).cpu().numpy() # signle image
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["anchors"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()
        self.istraining = training_state

    @torch.no_grad()
    def render_traj(self, step: int, stage: str = "val"):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        training_state = self.istraining
        self.istraining = False
        cfg = self.cfg
        device = self.device

        num_imgs = len(self.parser.camtoworlds)

        camtoworlds_all = self.parser.camtoworlds[: num_imgs//2]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 6 #1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/{stage}_traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=None,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                time=(i%self.cfg.GOP_size)/(self.cfg.GOP_size - 1)
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            # canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = canvas_list[0].squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/{stage}_traj_{step}.mp4")
        self.istraining = training_state

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"

        if os.path.exists(compress_dir):
            shutil.rmtree(compress_dir)
        os.makedirs(compress_dir)

        self.run_param_distribution_vis(self.splats, save_dir=f"{cfg.result_dir}/visualization/raw")
        
        if isinstance(self.compression_method, GIFStreamEnd2endCompression):
            self.compression_method.compress(compress_dir, self.comp_sim_splats, self.entropy_models, self.cfg.entropy_channel, self.cfg.c_perframe, self.scaling, self.cfg.voxel_size)
            nets = {}
            nets["decoders"] = self.decoders.state_dict()
            for name, entropy_model in self.entropy_models.items():
                if entropy_model is not None:
                    nets[name+"_entropy_model"] = entropy_model.state_dict()
            nets["scaling"] = self.compression_sim_method.scaling
            torch.save(nets, os.path.join(compress_dir, "nets.pt"))
        else:
            raise NotImplementedError(f"The compression method is not implemented yet.")

        # evaluate compression
        if isinstance(self.compression_method, GIFStreamEnd2endCompression):
            self.load_models_from_compressed_dir(compress_dir, self.cfg.entropy_model_type)
        splats_c = self.compression_method.decompress(compress_dir, self.entropy_models, self.device)
        
        self.run_param_distribution_vis(splats_c, save_dir=f"{cfg.result_dir}/visualization/quant")
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        if self.cfg.knn:
            _, self.indices = find_k_neighbors(self.splats["anchors"], self.cfg.n_knn)
        if isinstance(self.compression_method, GIFStreamEnd2endCompression):
            self.comp_sim_splats, self.esti_bits_dict = self.compression_sim_method.simulate_compression(self.splats, self.cfg.max_steps, 0, self.cfg.entropy_channel)
        self.eval(step=step, stage="compress")
        self.render_traj(step=step, stage="compress")

    @torch.no_grad()
    def run_param_distribution_vis(self, param_dict: Dict[str, Tensor], save_dir: str):
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)
        for param_name, value in param_dict.items():
            
            tensor_np = value.flatten().detach().cpu().numpy()
            min_val, max_val = tensor_np.min(), tensor_np.max()
            plt.figure(figsize=(6, 4))
            n, bins, patches = plt.hist(tensor_np, bins=50, density=False, alpha=0.7, color='b')

            for count, bin_edge in zip(n, bins):
                plt.text(bin_edge, count, f'{int(count)}', fontsize=8, va='bottom', ha='center')

            plt.annotate(f'Min: {min_val:.2f}', xy=(min_val, 0), xytext=(min_val, max(n) * 0.1),
                        arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green')

            plt.annotate(f'Max: {max_val:.2f}', xy=(max_val, 0), xytext=(max_val, max(n) * 0.1),
                        arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red')

            plt.title(f'{param_name} Distribution')
            plt.xlabel('Value')
            plt.ylabel('Density')

            plt.savefig(os.path.join(save_dir, f'{param_name}.png'))

            plt.close()
        
        print(f"Histograms saved in '{save_dir}' directory.")
    
    def load_entropy_model_from_ckpt(self, ckpt: Dict, entropy_model_type: str):
        self.entropy_models = {}
        if entropy_model_type == "conditional_gaussian_model":
                self.compression_sim_method = GIFStreamCompressionSimulation(self.cfg.compression_sim,
                                                    self.cfg.entropy_model_type,
                                                    self.cfg.entropy_steps,
                                                    self.device,
                                                    False,
                                                    None,
                                                    None,
                                                    feature_dim=self.cfg.anchor_feature_dim,
                                                    n_offsets=self.cfg.n_offsets,
                                                    c_channel=self.cfg.entropy_channel,
                                                    p_channel=self.cfg.c_perframe)
        self.entropy_models = self.compression_sim_method.entropy_models
        for name, value in ckpt.items():
            if "_entropy_model" in name:
                attr_name = name[:(len(name) - len("_entropy_model"))]
                num_ch = ckpt["splats"][attr_name].shape[-1]
                if value is not None:
                    self.entropy_models[attr_name].load_state_dict(value)
        self.compression_sim_method.scaling = ckpt["scaling"]
        self.scaling = ckpt["scaling"]
        self.comp_sim_splats, self.esti_bits_dict = self.compression_sim_method.simulate_compression(self.splats, self.cfg.max_steps, 0, self.cfg.entropy_channel)

    def load_models_from_compressed_dir(self, compress_dir, entropy_model_type: str):
        self.entropy_models = {}
        if entropy_model_type == "conditional_gaussian_model":
            if hasattr(self, 'compression_sim_method'):
                simulation = self.compression_sim_method
            else:
                self.compression_sim_method = GIFStreamCompressionSimulation(self.cfg.compression_sim,
                                                    self.cfg.entropy_model_type,
                                                    self.cfg.entropy_steps,
                                                    self.device,
                                                    False,
                                                    None,
                                                    None,
                                                    feature_dim=self.cfg.anchor_feature_dim,
                                                    n_offsets=self.cfg.n_offsets,
                                                    c_channel=self.cfg.entropy_channel,
                                                    p_channel=self.cfg.c_perframe)
        self.entropy_models = self.compression_sim_method.entropy_models
        
        ckpt = torch.load(os.path.join(compress_dir, "nets.pt"), map_location=self.device, weights_only=False)
        for name, value in ckpt.items():
            if "_entropy_model" in name:
                attr_name = name[:(len(name) - len("_entropy_model"))]
                if value is not None:
                    self.entropy_models[attr_name].load_state_dict(value)
        self.compression_sim_method.scaling = ckpt["scaling"]
        self.scaling = ckpt["scaling"]
        self.decoders.load_state_dict(ckpt["decoders"])

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=None,
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        if not cfg.continue_training:
            # run eval only
            ckpts = [
                torch.load(file, map_location=runner.device, weights_only=False)
                for file in cfg.ckpt
            ]
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            runner.decoders.load_state_dict(ckpts[0]["decoders"])
            step = ckpts[0]["step"]
            runner.cfg.compression_sim = ckpts[0]["compression_sim"]
            if runner.cfg.compression_sim:
                runner.load_entropy_model_from_ckpt(ckpts[0], cfg.entropy_model_type)
            if cfg.knn:
                _, runner.indices = find_k_neighbors(runner.splats["anchors"], cfg.n_knn)
            runner.eval(step=step)
            runner.render_traj(step=step)
            if cfg.compression is not None:
                if cfg.compression == "end2end":
                    assert ckpts[0]["compression_sim"]
                    runner.run_compression(step=step)
                else:
                    print(f"Do not support {cfg.compression} now !")
        else:
            ckpts = [
                torch.load(file, map_location=runner.device, weights_only=False)
                for file in cfg.ckpt
            ]
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            runner.decoders.load_state_dict(ckpts[0]["decoders"])
            if runner.cfg.app_opt:
                runner.app_module.load_state_dict(ckpts[0]["app_module"])
            runner.train(init_step=7001)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

def quaternion_to_rotation_matrix(quaternion):
    if quaternion.dim() == 1:
        quaternion = quaternion.unsqueeze(0)
    
    w, x, y, z = quaternion.unbind(dim=-1)
    
    B = quaternion.size(0)
    
    rotation_matrix = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
    ], dim=-1).view(B, 3, 3)
    
    return rotation_matrix

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w_new = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_new = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_new = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_new = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q_new = torch.stack([w_new, x_new, y_new, z_new], dim=-1)
    return q_new

if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer_scaffold.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "neur3d_0": (
            "neur3d dataset",
            Config(
                strategy=GIFStreamStrategy(verbose=True,densify_grad_threshold=0.0005,deformation_gate=0.03),
                test_set=[0],
                normalize_world_space=False,
                anchor_feature_dim=24,
                c_perframe = 4,
                app_opt=True,
                app_embed_dim=6,
            ),
        ),
        "neur3d_1": (
            "neur3d dataset",
            Config(
                strategy=GIFStreamStrategy(verbose=True,densify_grad_threshold=0.0006,deformation_gate=0.03),
                test_set=[0],
                normalize_world_space=False,
                anchor_feature_dim=48,
                c_perframe = 4,
                app_opt=False,
            ),
        ),
        "neur3d_2": (
            "neur3d dataset",
            Config(
                strategy=GIFStreamStrategy(verbose=True,densify_grad_threshold=0.0006,deformation_gate=0.03),
                test_set=[0],
                remove_set=[12],
                normalize_world_space=False,
                anchor_feature_dim=48,
                c_perframe = 4,
                app_opt=False,
            ),
        ),
        "GSC": (
            "mpeg dataset",
            Config(
                strategy=GIFStreamStrategy(verbose=True,densify_grad_threshold=0.0002,deformation_gate=0.03),
                test_set=[8,10,12],
                normalize_world_space=False,
                anchor_feature_dim=24,
                c_perframe = 8,
                app_opt=False,
                batch_size=4,
            ),
        ),
        "default": (
            "GIFStream with compression.",
            Config(
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
