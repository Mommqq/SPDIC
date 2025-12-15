import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import Adam, AdamW
import matplotlib.pyplot as plt
# import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import cycle
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ms_ssim
from math import log10
import PIL.Image as Image
from PIL import Image, ImageFilter
import pandas as pd
import os
import math
import numpy as np
import lpips
from lpips import LPIPS
from collections import OrderedDict
from pytorch_fid import fid_score
from datetime import datetime
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from DISTS_pytorch import DISTS
import torch.nn.functional as F
import config
from accelerate import Accelerator
import time
# accelerator = Accelerator()
from src.helpers import prob_mask_like, find_linear_layers, update_scheduler, get_pred_original_sample
device0 = config.device0
device1 = config.device1
from torchvision.transforms import GaussianBlur
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def nt_xent_loss(anchor, positive, negatives, temperature=0.1):
    """
    anchor: (B, C, H, W)
    positive: (B, C, H, W)
    negatives: list of (B, C, H, W)
    """

    B, C, H, W = anchor.size()
    anchor_flat = F.normalize(anchor.view(B, -1), dim=1)
    positive_flat = F.normalize(positive.view(B, -1), dim=1)
    neg_flat = [F.normalize(n.view(B, -1), dim=1) for n in negatives]

    # Cosine similarities
    sim_pos = torch.exp(torch.sum(anchor_flat * positive_flat, dim=1) / temperature)
    sim_negs = [torch.exp(torch.sum(anchor_flat * n, dim=1) / temperature) for n in neg_flat]

    sim_den = sum(sim_negs) + sim_pos + 1e-8
    loss = -torch.log(sim_pos / sim_den)
    return loss.mean()

def visualize_aggregated(feature_map, method='mean', save_dir='outputs_featmap', file_prefix='heatmap'):
    os.makedirs(save_dir, exist_ok=True)

    if feature_map.dim() == 3:
        feature_map = feature_map.unsqueeze(0)  

    with torch.no_grad():
        if feature_map.shape[1] == 1:
            feat = feature_map.squeeze(1)  # (B,H,W)
        else:
            feat = feature_map.squeeze(0)  # (C,H,W)

        if method == 'mean':
            aggregated = torch.mean(feat, dim=0)  # (H,W)
        else:
            aggregated, _ = torch.max(feat, dim=0)  # (H,W)

        aggregated = aggregated.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    upsampled = F.interpolate(aggregated, size=(256, 512), mode='bilinear', align_corners=False)
    upsampled_np = upsampled.squeeze().cpu().numpy()

    height, width = upsampled_np.shape
    aspect_ratio = width / height
    fig_width = 10
    fig_height = fig_width / aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    im = ax.imshow(upsampled_np, cmap='viridis', aspect='auto')
    ax.axis('off')
    plt.title(f'{method.upper()} Heatmap', fontsize=12, pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'{file_prefix}_{method}_{timestamp}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def calculate_dists(img1, img2):
        """
        Calculate DISTS between two images.

        Parameters:
            img1: torch.Tensor, Image 1 of shape (1, 3, H, W), normalized to [0, 1].
            img2: torch.Tensor, Image 2 of shape (1, 3, H, W), normalized to [0, 1].

        Returns:
            float, DISTS value.
        """
        dists_model = DISTS().to(img1.device)
        dists_value = dists_model(img1, img2)
        return dists_value.item()

def calculate_lpips(img1, img2, model_type='vgg'):
        """
        Calculate LPIPS between two images.

        Parameters:
            img1: torch.Tensor, Image 1 of shape (1, 3, H, W), normalized to [0, 1].
            img2: torch.Tensor, Image 2 of shape (1, 3, H, W), normalized to [0, 1].
            model_type: str, Backbone network for LPIPS ('alex', 'vgg', 'squeeze').

        Returns:
            float, LPIPS value.
        """
        lpips_model = lpips.LPIPS(net=model_type)
        lpips_model = lpips_model.to(img1.device)
        lpips_value = lpips_model(img1, img2)
        return lpips_value.item()


# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        sample_steps,
        diffusion_model,
        train_dl,
        val_dl,
        test_dl,
        scheduler_function,
        num_epochs = 500000000,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        results_folder_load="./results",
        tensorboard_dir="./tensorboard_logs/diffusion/",
        model_name_load="model",
        model_name_save="model",
        val_num_of_batch=1,
        optimizer="adam",
        sample_mode="ddpm",
        log_file_train = "./kitti_result_distribute/train_log_alpha0.9_beta0.0128",
        log_file_val = "./kitti_result_distribute/val_log_alpha0.9_beta0.0128"
    ):
        super().__init__()
        self.model = diffusion_model

        # self.ema = EMA(ema_decay)
        # self.ema_model = copy.deepcopy(self.model)
        self.sample_mode = sample_mode
        # self.update_ema_every = update_ema_every
        self.val_num_of_batch = val_num_of_batch
        self.sample_steps = sample_steps

        # self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.loss_fn_vgg = lpips.LPIPS(net="vgg", eval_mode=True) 
        self.loss_fn_vgg.to(device0)
        self.train_num_steps = train_num_steps
        self.num_epochs = num_epochs
        self.weight_loss = weight_Loss()
        # self.train_dl_class = train_dl
        # self.val_dl_class = val_dl
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        # self.W = AutoBalanceWeight()
        if optimizer == "adam":
            self.opt = Adam(self.model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            module_a_params = list(self.model.unet.parameters())
            all_params = set(self.model.parameters())
            special_params = set(module_a_params) 
            base_params = list(all_params - special_params)
            # self.opt = AdamW(self.model.parameters(), lr=train_lr)
            self.opt = AdamW([{"params": base_params, "lr": train_lr},{"params": module_a_params, "lr": 1e-4}])
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)
        # self.model, self.opt, self.train_dl = accelerator.prepare(self.model, self.opt, self.train_dl)
        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder_load = Path(results_folder_load)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name_load = model_name_load
        self.model_name_save = model_name_save
 
        # if os.path.isdir(tensorboard_dir):
        #     shutil.rmtree(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)
        self.train_log_file = log_file_train
        self.val_log_file = log_file_val

        self.initialize_log_file()

    def save_image(self,x_recon, x, path, name):
        img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
        img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
        img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
        img = np.transpose(img, (1, 2, 0)).astype('uint8')
        # img_final = Image.fromarray(np.concatenate((img, img_recon), axis=1), 'RGB')
        img_final = Image.fromarray((img_recon),'RGB')        
        if not os.path.exists(path):
            os.makedirs(path)
        img_final.save(os.path.join(path, name + '.png'))
    


    def calculate_dists(self,img1, img2):
        """
        Calculate DISTS between two images.

        Parameters:
            img1: torch.Tensor, Image 1 of shape (1, 3, H, W), normalized to [0, 1].
            img2: torch.Tensor, Image 2 of shape (1, 3, H, W), normalized to [0, 1].

        Returns:
            float, DISTS value.
        """
        dists_model = DISTS().to(img1.device)
        dists_value = dists_model(img1, img2)
        return dists_value.item()

    def calculate_lpips(self,img1, img2, model_type='vgg'):
        """
        Calculate LPIPS between two images.

        Parameters:
            img1: torch.Tensor, Image 1 of shape (1, 3, H, W), normalized to [0, 1].
            img2: torch.Tensor, Image 2 of shape (1, 3, H, W), normalized to [0, 1].
            model_type: str, Backbone network for LPIPS ('alex', 'vgg', 'squeeze').

        Returns:
            float, LPIPS value.
        """
        lpips_model = lpips.LPIPS(net=model_type)
        lpips_model = lpips_model.to(img1.device)
        lpips_value = lpips_model(img1, img2)
        return lpips_value.item()

    def save(self):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            # "ema": self.ema_model.module.state_dict(),
        }
        # idx = (self.step // self.save_and_sample_every) % 2
        idx = 1
        torch.save(data, str(self.results_folder / f"{self.model_name_save}_{idx}_{config.save_name}.pt"))

    def load(self, idx=config.load_idx, load_step=True):
        data = torch.load(
            str(self.results_folder_load / f"{self.model_name_load}_{idx}_{config.load_name}.pt"),
            # str(self.results_folder_load / "big-l1-Cityscape-d64-t20000-b0.0012-vbrFalse-noise-linear-aux1.0lpips_0_pro0_0.31bpp.pt"),
            map_location=lambda storage, loc: storage,
        )
        
        if load_step:
            self.step = data["step"]
        try:
            checkpoint_state = data["model"]
            model_state = self.model.module.state_dict()

            filtered_state = {}
            for key, value in checkpoint_state.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_state[key] = value
                    else:
                        print(f"skip {key}: checkpoint shape {value.shape} not match {model_state[key].shape}")
                else:
                    print(f"p {key} not in model")
            
            model_state.update(filtered_state)
            self.model.module.load_state_dict(model_state, strict=False)
            # self.model.module.load_state_dict(data["model"], strict=False)
            for name, param in self.model.named_parameters():
                    if "blip2." in name:
                        param.requires_grad = False     
                    if "text_encoder." in name:
                        param.requires_grad = False                         
                    if "vae" in name:
                        param.requires_grad = False   
                    if "unet.conv_in.weight" in name:
                        param.requires_grad = False 
                    if "unet.conv_in.bias" in name:
                        param.requires_grad = False         
        except:
            checkpoint_state = data["model"]
            model_state = self.model.state_dict()
            filtered_state = {}
            for key, value in checkpoint_state.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_state[key] = value
                    else:
                        print(f"skip p {key}: checkpoint shape {value.shape} not match {model_state[key].shape}")
                else:
                    print(f"p {key} not in model")
            
            model_state.update(filtered_state)
            self.model.load_state_dict(model_state, strict=False)
            for name, param in self.model.named_parameters():

                    if "blip2." in name:
                        param.requires_grad = False     
                    if "text_encoder." in name:
                        param.requires_grad = False                           
                    if "vae" in name:
                        param.requires_grad = False   
                    if "unet.conv_in.weight" in name:
                        param.requires_grad = False 
                    if "unet.conv_in.bias" in name:
                        param.requires_grad = False         

    def initialize_log_file(self):
        if not os.path.exists(self.val_log_file):
            with open(self.val_log_file, 'a') as f:
                f.write("Epoch, Avg PSNR, Avg ms-ssim, Avg Loss, Avg Bpp\n") 
        # print(self.model)
        for name, param in self.model.named_parameters():


                if "blip2." in name:
                    param.requires_grad = False     
                if "text_encoder." in name:
                    param.requires_grad = False                           
                if "vae" in name:
                    param.requires_grad = False   
                if "unet.conv_in.weight" in name:
                    param.requires_grad = False 
                if "unet.conv_in.bias" in name:
                    param.requires_grad = False              

    def val_log_to_file(self, epoch, avg_psnr, avg_ms_ssim, avg_ms_ssim_db,avg_loss, avg_bpp, avg_bpp_y,avg_bpp_texty):
        with open(self.val_log_file, 'a') as f:  
            f.write(f"{epoch}, {avg_psnr:.2f}, {avg_ms_ssim:.4f}, {avg_ms_ssim_db:.4f}, {avg_loss:.4f}, {avg_bpp: .4f}, {avg_bpp_y: .4f},{avg_bpp_texty: .4f}\n")
    
    def save_image(self, x_recon, x, path, name):
        img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
        img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
        img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
        img = np.transpose(img, (1, 2, 0)).astype('uint8')
        # img_final = Image.fromarray(np.concatenate((img, img_recon), axis=1), 'RGB')
        img_final = Image.fromarray((img_recon),'RGB')        
        if not os.path.exists(path):
            os.makedirs(path)
        img_final.save(os.path.join(path, name + '.png'))
    def train(self):
        # plt.ion() 
        print("lambda:",config.beta)
        for epoch in range(self.num_epochs):
            total_aloss = []
            total_loss = []

            print(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            pbar = tqdm(total=len(self.train_dl), desc=f"Epoch {epoch + 1}", ncols=120)
            for data_x, data_y, _, _ in self.train_dl:
                
                self.opt.zero_grad()
                data_x = data_x.to(device0)
                data_y = data_y.to(device0)
                # data_x = data_x
                # data_y = data_y
                self.model.train()
              
                q_latent, q_hyper_latent, state4bpp = self.model.encode_x(data_x*2-1, None)
                latent_y, hyper_latent_y, state4bpp_y = self.model.encode_y(data_y*2-1, None) 
                w, text_y = self.model.encode_w(data_y*2-1)
                # print(text_y)
                bpp_x,bpp_y = self.model.bpp(data_x.shape, state4bpp, state4bpp_y)

                byte_stream_text = self.model.compress_text(text_y)
                bpp_texty = self.model.calculate_bpp(byte_stream_text, 128*256)
                # self.model.decode_xy.to(device1)
                q_latent = q_latent.to(device1)
                latent_y = latent_y.to(device0)
                w = w.to(device1)
                output, output_y, noise_pred, noise_org = self.model.decode_xy(q_latent, latent_y, w,None)
                
                # get_extra_loss=self.model.get_extra_loss().to(device0)

                ox, oy, bppx, bppy, aloss = output[0], output_y[0], bpp_x, bpp_y ,self.model.get_extra_loss()
                ox = ox.to(device0)
                oy = oy.to(device0)
                bppxM = bppx.mean().to(device0)
                bppyM = bppy.mean().to(device0)
                aloss = aloss.to(device0)
                pred_x = (ox.clamp(-1, 1) + 1.0) * 0.5 
                pred_y = (oy.clamp(-1, 1) + 1.0) * 0.5
                ms_ssim_value = ms_ssim(data_x, pred_x, data_range=1.0, size_average=True, win_size=7)

                dr_loss = 0.9*F.mse_loss(pred_x, data_x)+0.1*self.loss_fn_vgg(pred_x, data_x).mean() + config.beta*(bppxM)

                # dr_loss = 0.3*F.mse_loss(pred_x, data_x)+0.7*self.loss_fn_vgg(pred_x, data_x).mean()  + config.beta*(bppxM)
                # dr_loss = 0.1*F.mse_loss(pred_x, data_x)+0.9*self.loss_fn_vgg(pred_x, data_x).mean()  + config.beta*(bppxM)
                # dr_loss = self.loss_fn_vgg(pred_x, data_x).mean()  + config.beta*(bppxM)
                # dr_loss = F.mse_loss(pred_x, data_x) + config.beta*(bppxM)

                loss = dr_loss + aloss
                mse = torch.mean((data_x - pred_x) ** 2)
                psnr = 20 * log10(1.0 / torch.sqrt(mse))            
                loss.backward()
                
                msssim = 1-ms_ssim_value.detach()
                msssim_db = -10 * torch.log10(msssim + 1e-8)
                self.opt.step()
                
                total_loss.append(dr_loss.item())
                total_aloss.append(aloss.item())
                
                pbar.set_postfix({
                    'loss': f"{dr_loss.item():.4f}",
                    # 'aloss': f"{aloss.item():.4f}",
                    'PSNR': f"{psnr:.4f}",
                    'MS-SSIM': f"{ms_ssim_value:.4f}",
                    'MS-SSIM-DB': f"{msssim_db:.4f}",
                    'bpp': f"{bppx.mean().item():.4f}",
                    'lr': f"{self.opt.param_groups[0]['lr']:.1e}"
                })
                pbar.update(1)

                if self.step % self.scheduler_checkpoint_step == 0 and self.step != 0:
                    self.scheduler.step()
                self.step += 1
                # if self.step % 500 == 0 : 
                #     metrics = self.validate()
                #     print(f"Epoch {epoch + 1}: Validation Metrics: {metrics}")
                #     self.val_log_to_file(
                #         epoch + 1, 
                #         metrics['avg_psnr'], 
                #         metrics['avg_ms_ssim'],
                #         metrics['avg_ms_ssim_db'],
                #         metrics['avg_loss'], 
                #         metrics['avg_bpp'], 
                #         metrics['avg_bpp_y'],
                #         metrics['avg_bpp_texty'],
                #     )
                
                #     idx = (self.step // self.save_and_sample_every) % 2
                #     print("step", self.step, "idx:", idx)
                #     self.save() 
          
            # 计算每个 epoch 的平均指标
            avg_aloss = sum(total_aloss) / len(total_aloss)
            avg_loss = sum(total_loss) / len(total_loss)
            print("avg_loss:", avg_loss, "avg_aloss:", avg_aloss, 'lr:', self.opt.param_groups[0]['lr'])

            pbar.close()
            print("current step:", self.step)
            self.save()            
            if (epoch + 1) % 5 == 0 : 
            # if 1:  
                metrics = self.validate()
                print(f"Epoch {epoch + 1}: Validation Metrics: {metrics}")
                self.val_log_to_file(
                    epoch + 1, 
                    metrics['avg_psnr'], 
                    metrics['avg_ms_ssim'],
                    metrics['avg_ms_ssim_db'],
                    metrics['avg_loss'], 
                    metrics['avg_bpp'], 
                    metrics['avg_bpp_y'],
                    metrics['avg_bpp_texty'],
                )
            
            idx = (self.step // self.save_and_sample_every) % 2
            print("epoch", epoch + 1, "idx:", idx)
        self.save()
        print("Training completed")




    def validate(self):
        self.model.eval()
        metrics = {'psnr': [], 'ms_ssim': [], 'ms_ssim_db': [], 'loss': [], 'bpp': [], 'bpp_y': [],'bpp_texty':[]}
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, _, _ in tqdm(self.val_dl, desc="Validating", ncols=100):
                batch_x = batch_x.to(device0)
                batch_y = batch_y.to(device0)
             
                q_latent, q_hyper_latent, state4bpp = self.model.encode_x(batch_x*2-1, None)
                latent_y, hyper_latent_y, state4bpp_y = self.model.encode_y(batch_y*2-1, None) 
                w,text_y = self.model.encode_w(batch_y*2-1)
                bpp_x,bpp_y = self.model.bpp(batch_x.shape, state4bpp, state4bpp_y)
                byte_stream_text = self.model.compress_text(text_y)
                bpp_texty = self.model.calculate_bpp(byte_stream_text, 128*256)
                # self.model.decode_xy.to(device1)
                q_latent = q_latent.to(device1)
                latent_y = latent_y.to(device0)
                w = w.to(device1)
                num_steps = config.num_denoise_steps
                self.model.latent_proj.to(device1)
                self.model.latent_proj_back.to(device1)
                self.model.unet.to(device1)
                self.model.vae.to(device1)
                x_t=self.model.latent_proj(q_latent)
                noise = torch.randn(x_t.shape, device=x_t.device)
                self.model.noise_scheduler.set_timesteps(num_steps)
                timesteps = self.model.noise_scheduler.timesteps

                # assert len(timesteps) == num_steps, "Scheduler didn't pick correct number of steps!"
                # print(f"[DEBUG] timesteps (len {len(timesteps)}): {timesteps[:3]} ... {timesteps[-3:]}")

                # x_t = self.model.noise_scheduler.add_noise(x_t, noise, timesteps[0])
                # x_t = x_t*self.model.vae.config.scaling_factor
                x_t = noise
                for t in timesteps:
                    # old_xt = x_t.clone()
                    t_tensor = torch.tensor([t], device=device1)
                    model_pred = self.model.unet(x_t, t_tensor, w, q_latent)[0]
                    out = self.model.noise_scheduler.step(model_pred, t, x_t)
                    x_t = out.prev_sample


                x_t = self.model.latent_proj_back(x_t)
                x_t = x_t + q_latent
                q_latent = x_t
                output, output_y, noise_pred , noise_org = self.model.decode_xy(q_latent, latent_y, w, w)
                
                # get_extra_loss=self.model.get_extra_loss().to(device0)

                ox, oy, bppx, bppy, aloss = output[0], output_y[0], bpp_x, bpp_y ,self.model.get_extra_loss()
                # ox = ox.to(device0)
                # oy = oy.to(device0)
                bppxM = bppx.mean().to(device0)
                bppyM = bppy.mean().to(device0)
                aloss = aloss.to(device0)
                pred_x = (ox.clamp(-1, 1) + 1.0) * 0.5 
                pred_y = (oy.clamp(-1, 1) + 1.0) * 0.5
                ms_ssim_value = ms_ssim(batch_x, pred_x, data_range=1.0, size_average=True, win_size=7)
                # loss = 0.1*F.mse_loss(pred_x, batch_x)+0.9*self.loss_fn_vgg(pred_x, batch_x).mean() + F.mse_loss(noise_pred, noise_org) + config.beta*(bppx.mean())
                loss = 0.9*F.mse_loss(pred_x, batch_x)+0.1*self.loss_fn_vgg(pred_x, batch_x).mean() + config.beta*(bppx.mean())
                # loss = 0.9*(1-ms_ssim_value)+0.1*self.loss_fn_vgg(pred_x, batch_x).mean() + config.beta*(bppx.mean()) 

                total_loss += loss.item()

                mse = torch.mean((batch_x - pred_x) ** 2)
                psnr = 20 * log10(1.0 / torch.sqrt(mse))
                metrics['psnr'].append(psnr)
                metrics['bpp'].append(bppx.mean())
                metrics['bpp_y'].append(bppy.mean())
                metrics['bpp_texty'].append(bpp_texty)
                metrics['loss'].append(loss)
                
                msssim = 1-ms_ssim_value
                metrics['ms_ssim'].append(ms_ssim_value.item())
                ms_ssim_db = -10 * torch.log10(msssim + 1e-8)
                metrics['ms_ssim_db'].append(ms_ssim_db.item())
                #print("bpp:", bppx.mean().item(), "psnr:", psnr, "ms-ssim:", ms_ssim_value.item())

        avg_psnr    = sum(metrics['psnr']) / len(metrics['psnr'])
        avg_ms_ssim = sum(metrics['ms_ssim']) / len(metrics['ms_ssim'])
        avg_ms_ssim_db = sum(metrics['ms_ssim_db']) / len(metrics['ms_ssim_db'])

        avg_bpp     = sum(metrics['bpp']) / len(metrics['bpp'])
        avg_bpp_y   = sum(metrics['bpp_y']) / len(metrics['bpp_y'])
        avg_bpp_texty   = sum(metrics['bpp_texty']) / len(metrics['bpp_texty'])
        avg_loss    = sum(metrics['loss']) / len(metrics['loss'])

        print(f"avg bpp: {avg_bpp:.4f}, avg bpp_y: {avg_bpp_y:.4f},  Avg PSNR: {avg_psnr:.2f}, "
            f"Avg MS-SSIM: {avg_ms_ssim:.4f}, Avg MS-SSIM_DB: {avg_ms_ssim_db:.4f}, Avg Loss: {avg_loss:.4f} ")

        return {
            'avg_bpp': avg_bpp,
            'avg_bpp_y': avg_bpp_y,
            'avg_bpp_texty':avg_bpp_texty,
            'avg_psnr': avg_psnr,
            'avg_ms_ssim': avg_ms_ssim,
            'avg_ms_ssim_db': avg_ms_ssim_db,            
            'avg_loss': avg_loss
        }


    def test(self):
        inference_times = []
        self.model.eval()
        total_params = count_parameters(self.model) / 1e6  
        metrics = {'psnr': [], 'ms_ssim': [], 'ms_ssim_db': [], 'lpips':[], 'dists':[],'loss': [], 'bpp': [], 'bpp_y': [],'bpp_texty':[]}
        total_loss = 0
        id = 1
        cols = dict() 
        names = ["Image Number", "BPP", "PSNR", "MS-SSIM", "MS-SSIM (dB)", "LPIPS", "DISTS"]        
        with torch.no_grad():
            for batch_x, batch_y, _, _ in tqdm(self.test_dl, desc="test", ncols=100):
                batch_x = batch_x.to(device0)
                batch_y = batch_y.to(device0)
                start_time = time.time()            
                q_latent, q_hyper_latent, state4bpp = self.model.encode_x(batch_x*2-1, None)
                latent_y, hyper_latent_y, state4bpp_y = self.model.encode_y(batch_y*2-1, None) 
                w,text_y = self.model.encode_w(batch_y*2-1)
                bpp_x,bpp_y = self.model.bpp(batch_x.shape, state4bpp, state4bpp_y)
                byte_stream_text = self.model.compress_text(text_y)
                bpp_texty = self.model.calculate_bpp(byte_stream_text, 128*256)
                # self.model.decode_xy.to(device1)
                q_latent = q_latent.to(device1)
                latent_y = latent_y.to(device0)
                w = w.to(device1)
                num_steps = config.num_denoise_steps
                self.model.latent_proj.to(device1)
                self.model.latent_proj_back.to(device1)
                self.model.unet.to(device1)
                self.model.vae.to(device1)
                x_t=self.model.latent_proj(q_latent)
                noise = torch.randn(x_t.shape, device=x_t.device)
                self.model.noise_scheduler.set_timesteps(num_steps)
                timesteps = self.model.noise_scheduler.timesteps
                x_t = noise
                for t in tqdm(timesteps, desc="denoising", ncols=num_steps):
                    # old_xt = x_t.clone()
                    t_tensor = torch.tensor([t], device=device1)
                    model_pred = self.model.unet(x_t, t_tensor, w, q_latent)[0]
                    out = self.model.noise_scheduler.step(model_pred, t, x_t)
                    x_t = out.prev_sample
                x_t = self.model.latent_proj_back(x_t)
                # print(f"t={num_steps}, norm(pred1)={x_t.norm():.4f}")  
                x_t = x_t + q_latent
                q_latent = x_t
                output, output_y, noise_pred, noise_org = self.model.decode_xy(q_latent, latent_y, w, w)
                end_time = time.time()
                inference_times.append(end_time - start_time)

                # print(len(output))
                # print(output[0].shape,output[1].shape,output[2].shape,output[3].shape)
                # get_extra_loss=self.model.get_extra_loss().to(device0)
                # visualize_aggregated(output_y[4], method='max',save_dir=f'out/outputs_featmap_y{id}')
                # visualize_aggregated(output_y[3], method='max',save_dir=f'out/outputs_featmap_y{id}')
                # visualize_aggregated(output_y[2], method='max',save_dir=f'out/outputs_featmap_y{id}')
                # visualize_aggregated(output_y[1], method='max',save_dir=f'out/outputs_featmap_y{id}')

                # visualize_aggregated(output[4], method='max',save_dir=f'out/outputs_featmap_x{id}')
                # visualize_aggregated(output[3], method='max',save_dir=f'out/outputs_featmap_x{id}')
                # visualize_aggregated(output[2], method='max',save_dir=f'out/outputs_featmap_x{id}')
                # visualize_aggregated(output[1], method='max',save_dir=f'out/outputs_featmap_x{id}')

                # visualize_aggregated(torch.cat((output[4],output_y[4]),1), method='max',save_dir=f'out/outputs_featmap_xcaty{id}')
                # visualize_aggregated(torch.cat((output[3],output_y[3]),1), method='max',save_dir=f'out/outputs_featmap_xcaty{id}')
                # visualize_aggregated(torch.cat((output[2],output_y[2]),1), method='max',save_dir=f'out/outputs_featmap_xcaty{id}')
                # visualize_aggregated(torch.cat((output[1],output_y[1]),1), method='max',save_dir=f'out/outputs_featmap_xcaty{id}')
                ox, oy, bppx, bppy, aloss = output[0], output_y[0], bpp_x, bpp_y ,self.model.get_extra_loss()
                
                # ox = ox.to(device0)
                # oy = oy.to(device0)
                bppxM = bppx.mean().to(device0)
                bppyM = bppy.mean().to(device0)
                aloss = aloss.to(device0)
                pred_x = (ox.clamp(-1, 1) + 1.0) * 0.5 
                pred_y = (oy.clamp(-1, 1) + 1.0) * 0.5
                # loss = 0.1*F.mse_loss(pred_x, batch_x) + 0.9*self.loss_fn_vgg(pred_x, batch_x).mean() + F.mse_loss(noise_pred,noise_org) + config.beta*(bppx.mean())
                loss = 0.9*F.mse_loss(pred_x, batch_x) + 0.1*self.loss_fn_vgg(pred_x, batch_x).mean() +config.beta*(bppx.mean())
                total_loss += loss.item()

                mse = torch.mean((batch_x - pred_x) ** 2)
                psnr = 20 * log10(1.0 / torch.sqrt(mse))
                lpips_score = self.calculate_lpips(batch_x, pred_x)
                dists_score = self.calculate_dists(batch_x, pred_x)
                # lpips_score = dists_score
                metrics['psnr'].append(psnr)
                metrics['bpp'].append(bppx.mean())
                metrics['bpp_y'].append(bppy.mean())
                metrics['bpp_texty'].append(bpp_texty)
                metrics['loss'].append(loss)
                metrics['lpips'].append(lpips_score)
                metrics['dists'].append(dists_score)
                # 计算MS-SSIM
                ms_ssim_value = ms_ssim(batch_x, pred_x, data_range=1.0, size_average=True, win_size=7)
                msssim = 1-ms_ssim_value
                metrics['ms_ssim'].append(ms_ssim_value.item())
                ms_ssim_db = -10 * torch.log10(msssim + 1e-8)
                metrics['ms_ssim_db'].append(ms_ssim_db.item())
                print("bpp:", bppx.mean().item(), "psnr:", psnr, "ms-ssim:", ms_ssim_value.item(),"lpips:",lpips_score, "dists:",dists_score)
                print("id:",id)

                # results_path = os.path.join('./', 'kitti_stero_compressed_imgs')
                if config.dataset_name == "Cityscape":
                    results_path = os.path.join('./', 'city_compressed_imgs')
                    self.save_image(pred_x[0], batch_x[0], os.path.join(results_path, '{}_images'.format(config.beta)+'city_test_inf{}_'.format(config.alpha)), str(id))
                    # self.save_image(batch_x[0], batch_x[0], os.path.join(results_path, '{}_orange_images'.format(config.beta)), str(id))
                    # self.save_image(batch_y[0], batch_y[0], os.path.join(results_path, '{}_y_orange_images'.format(config.beta)), str(id))
                if config.dataset_name == "KITTI_Stereo":
                    results_path = os.path.join('./', 'kitti_stero_compressed_imgs_nolocal')
                    self.save_image(pred_x[0], batch_x[0], os.path.join(results_path, '{}_images'.format(config.beta)+'kitti_{}_'.format(config.alpha)), str(id))
                # self.save_image(batch_x[0], batch_x[0], os.path.join(results_path, '{}_orange_images'.format(config.beta)), str(id))
                # self.save_image(batch_y[0], batch_y[0], os.path.join(results_path, '{}_y_orange_images'.format(config.beta)), str(id))

                i = iter([id])
                vals = [str(i)] + ['{:.8f}'.format(x) for x in [bppx.mean(), psnr, ms_ssim_value.item(), ms_ssim_db.item(), lpips_score, dists_score]]
                for (name, val) in zip(names, vals):
                    if name not in cols:
                        cols[name] = []
                    cols[name].append(val)

                id = id+1

        print(f"Total Trainable Parameters: {total_params:.2f}M")
        avg_inference_time = sum(inference_times) / len(inference_times)
        print(f"Average Inference Time per Sample: {avg_inference_time:.4f} seconds")

        avg_psnr = sum(metrics['psnr']) / len(metrics['psnr'])
        avg_ms_ssim = sum(metrics['ms_ssim']) / len(metrics['ms_ssim'])
        avg_ms_ssim_db = sum(metrics['ms_ssim_db']) / len(metrics['ms_ssim_db'])
        avg_lpips = sum(metrics['lpips']) / len(metrics['lpips'])
        avg_dists = sum(metrics['dists']) / len(metrics['dists'])

        avg_bpp     = sum(metrics['bpp']) / len(metrics['bpp'])
        avg_bpp_y   = sum(metrics['bpp_y']) / len(metrics['bpp_y'])
        avg_bpp_texty   = sum(metrics['bpp_texty']) / len(metrics['bpp_texty'])
        avg_loss    = sum(metrics['loss']) / len(metrics['loss'])
        print("num_steps:",config.num_denoise_steps)
        print(f"avg bpp: {avg_bpp:.4f}, avg bpp_y: {avg_bpp_y:.4f},  Avg PSNR: {avg_psnr:.2f}, "
            f"Avg MS-SSIM: {avg_ms_ssim:.4f}, Avg MS-SSIM_DB: {avg_ms_ssim_db:.4f}, Avg LPIPS: {avg_lpips:.4f},Avg DISTS: {avg_dists:.4f}, Avg Loss: {avg_loss:.4f} ")
        df = pd.DataFrame.from_dict(cols)
        df.to_csv(os.path.join(results_path, '{}_'.format(config.beta) + '.csv'))
        return {
            'avg_bpp': avg_bpp,
            'avg_bpp_y': avg_bpp_y,
            'avg_bpp_texty':avg_bpp_texty,
            'avg_psnr': avg_psnr,
            'avg_ms_ssim': avg_ms_ssim,
            'avg_ms_ssim_db': avg_ms_ssim_db,  
            'avg_lpips': avg_lpips,
            'avg_dists': avg_dists,       
            'avg_loss': avg_loss
        }