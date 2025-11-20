#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from pytorch_wavelets import DWTForward
    WAVELETS_AVAILABLE = True
except ImportError:
    WAVELETS_AVAILABLE = False

def compute_wavelet_loss_curriculum(
    rgb_pred,
    rgb_gt,
    dwt,
    iteration,
    max_iterations,
    ll_weight,
    hf_weight,
    transition_ratio,
):
    """Compute curriculum wavelet loss with separate low/high bands."""

    Yl_pred, Yh_pred = dwt(rgb_pred)
    Yl_gt, Yh_gt = dwt(rgb_gt)

    low_freq_loss = torch.abs(Yl_pred - Yl_gt).mean()

    high_freq_diff = Yh_pred[0] - Yh_gt[0]
    high_freq_abs = torch.abs(high_freq_diff)
    high_freq_loss = high_freq_abs.mean(dim=(0, 1, 3, 4)).sum()

    max_iterations = max(1, int(max_iterations))
    transition_ratio = float(max(1e-6, min(1.0, transition_ratio)))
    transition_steps = max(1, int(max_iterations * transition_ratio))
    iter_index = max(0, iteration - 1)
    progress = min(iter_index / transition_steps, 1.0)

    lambda_ll = float(ll_weight) * max(0.0, 1.0 - progress)
    lambda_hf = float(hf_weight) * progress

    total_loss = lambda_ll * low_freq_loss + lambda_hf * high_freq_loss

    stats = {
        "low_freq_loss": low_freq_loss.detach(),
        "high_freq_loss": high_freq_loss.detach(),
        "lambda_ll": lambda_ll,
        "lambda_hf": lambda_hf,
    }
    return total_loss, stats


def wavelet_loss(rgb_pred, rgb_gt, dwt, weight_min, weight_max, eps=1e-6):
    """Compute HF-weighted RGB L1 using GT wavelet energy as weights."""

    with torch.no_grad():
        _, Yh_gt = dwt(rgb_gt)
        hf_coeffs = Yh_gt[0]
        hf_energy = torch.sum(hf_coeffs ** 2, dim=(1, 2))
        hf_energy = torch.sqrt(hf_energy + eps).unsqueeze(1)
        weight_map = F.interpolate(
            hf_energy,
            size=rgb_gt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        weight_map_mean = weight_map.mean(dim=(2, 3), keepdim=True)
        weight_map = weight_map / (weight_map_mean + eps)
        weight_map = torch.clamp(weight_map, min=weight_min, max=weight_max)
        weight_map = weight_map.detach()

    per_pixel = torch.abs(rgb_pred - rgb_gt).mean(dim=1, keepdim=True)
    weighted_sum = (weight_map * per_pixel).sum()
    normalization = weight_map.sum().clamp_min(1e-6)
    return weighted_sum / normalization

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.hash_size, dataset.width, dataset.depth)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Initialize wavelet transform for wavelet loss if enabled
    dwt = None
    if opt.use_wavelet_loss:
        if WAVELETS_AVAILABLE:
            dwt = DWTForward(J=1, wave='haar', mode='zero').cuda()
            print(f"[Wavelet Loss] Enabled with weight={opt.lambda_wavelet}")
        else:
            print("[Warning] use_wavelet_loss=True but pytorch_wavelets not installed. Disabling wavelet loss.")
            print("[Info] Install with: pip install pytorch_wavelets")
            opt.use_wavelet_loss = False

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        hf_rgb_loss_value = None
        rgb_pred_batch = None
        rgb_gt_batch = None
        if opt.use_wavelet_loss and dwt is not None:
            rgb_pred_batch = image.unsqueeze(0)
            rgb_gt_batch = gt_image.unsqueeze(0)
            pixel_loss = wavelet_loss(
                rgb_pred_batch,
                rgb_gt_batch,
                dwt,
                weight_min=opt.hf_weight_min,
                weight_max=opt.hf_weight_max,
                eps=opt.hf_weight_epsilon,
            )
            hf_rgb_loss_value = pixel_loss.detach()
        else:
            pixel_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        mask_loss = torch.mean(gaussians.get_mask)
        sh_mask_loss = 0.0
        if iteration > opt.densify_until_iter:
            for degree in range(1, gaussians.active_sh_degree + 1):
                lambda_degree = (2 * degree + 1) / ((gaussians.max_sh_degree + 1) ** 2 - 1)
                sh_mask_loss += lambda_degree * torch.mean(gaussians.get_sh_mask[..., degree - 1])

        loss = pixel_loss + opt.lambda_mask * mask_loss + opt.lambda_sh_mask * sh_mask_loss
        
        wavelet_stats = None
        wavelet_loss_value = None
        # Add wavelet loss if enabled
        if opt.use_wavelet_loss and dwt is not None:
            if rgb_pred_batch is None or rgb_gt_batch is None:
                rgb_pred_batch = image.unsqueeze(0)
                rgb_gt_batch = gt_image.unsqueeze(0)
            wavelet_loss_raw, wavelet_stats = compute_wavelet_loss_curriculum(
                rgb_pred_batch,
                rgb_gt_batch,
                dwt,
                iteration=iteration,
                max_iterations=opt.iterations,
                ll_weight=opt.wavelet_ll_weight,
                hf_weight=opt.wavelet_hf_weight,
                transition_ratio=opt.wavelet_transition_ratio,
            )
            loss = loss + opt.lambda_wavelet * wavelet_loss_raw
            wavelet_loss_value = wavelet_loss_raw.detach()
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            # Store loss components for detailed logging
            if iteration % 1000 == 0:
                loss_pixel_val = pixel_loss.item()
                loss_mask_val = mask_loss.item()
                loss_sh_mask_val = sh_mask_loss.item() if isinstance(sh_mask_loss, torch.Tensor) else sh_mask_loss
                loss_total_val = loss.item()
                
                # Enhanced debug logging with all key metrics
                mem_allocated_gb = torch.cuda.memory_allocated() / 1e9
                mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
                n_gaussians = gaussians.get_xyz.shape[0]
                
                # Get learning rates from optimizer
                grid_lr = gaussians.optimizer.param_groups[0]['lr']
                other_lr = gaussians.optimizer_i.param_groups[0]['lr'] if hasattr(gaussians, 'optimizer_i') else 0.0
                
                # Compute average iteration time
                avg_iter_time = iter_start.elapsed_time(iter_end) / 1000.0  # Convert to seconds
                
                print(f"\n{'='*80}")
                print(f"[Iter {iteration:05d}] Training Progress Report")
                print(f"{'-'*80}")
                print(f"  Loss Breakdown:")
                print(f"    Total       : {loss_total_val:.6f}")
                pixel_label = "Pixel (RGB)"
                if opt.use_wavelet_loss and hf_rgb_loss_value is not None:
                    pixel_label = "Pixel (HF RGB)"
                print(f"    {pixel_label:<13}: {loss_pixel_val:.6f} (L1={Ll1.item():.6f})")
                print(f"    Mask        : {loss_mask_val:.6f} (weight={opt.lambda_mask:.4f})")
                print(f"    SH Mask     : {loss_sh_mask_val:.6f} (weight={opt.lambda_sh_mask:.4f})")
                if opt.use_wavelet_loss and wavelet_loss_value is not None and wavelet_stats is not None:
                    print(
                        f"    Wavelet     : {wavelet_loss_value.item():.6f} "
                        f"(lambda_LL={wavelet_stats['lambda_ll']:.3f}, lambda_HF={wavelet_stats['lambda_hf']:.3f}, weight={opt.lambda_wavelet:.4f})"
                    )
                print(f"  Gaussians:")
                print(f"    Count       : {n_gaussians:,} ({n_gaussians//1000}k points)")
                print(f"    SH Degree   : {gaussians.active_sh_degree}/{gaussians.max_sh_degree}")
                print(f"  Learning Rates:")
                print(f"    Grid        : {grid_lr:.6f}")
                print(f"    Implicit    : {other_lr:.6f}")
                print(f"  Resources:")
                print(f"    GPU Memory  : {mem_allocated_gb:.2f}GB allocated, {mem_reserved_gb:.2f}GB reserved")
                print(f"    Time/Iter   : {avg_iter_time:.3f}s")
                print(f"{'='*80}\n")
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Num": f"{gaussians.get_xyz.shape[0]:07d}",
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            else:
                if iteration % opt.prune_interval == 0:
                    gaussians.mask_prune()
            
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer_i.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer_i.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
