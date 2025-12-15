import torch.nn as nn
from .network_components import ResnetBlock, VBRCondition, FlexiblePrior, Downsample, Upsample, GDN1
from modules.cross_attention import CrossAttention
from .utils import quantize, NormalDistribution
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BertTokenizer, BertModel
from src.unet_2d_perco import UNet2DConditionModel
import numpy as np
import random
import config
from PWAM import PWAM
from einops import rearrange, reduce
from helpers import prob_mask_like, find_linear_layers, update_scheduler, get_pred_original_sample
device = config.device1
device0 = config.device0
device1 = config.device1
import zlib
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator

from torchvision.transforms import GaussianBlur


MODEL_ID = "stable-diffusion-2-1"


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

def tokenize_captions(examples, is_train=True):
            tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", revision=None)
            captions = []
            for caption in examples:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])

            inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids


class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
        image_size = (128,256)
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)] #[3,64,128,256,512]
        # self.dims = [3,64,128,256,324]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))  #[(3,64),(64,128),(128,256),(256,512)]
        self.reversed_dims = list(reversed([out_channels, *map(lambda m: dim * m, dim_mults)])) # [512,256,128,64,3]
        # self.reversed_dims = [324,256,128,64,3]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:])) #[(512,256),(256,128),(128,64),(64,3)]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)] # [512,256,256,256]
        # self.hyper_dims = [324,256,256,256]

        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:])) #[(512,256),(256,256),(256,256)]
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)]) #(256,256,256,1024)
        )
        # self.reversed_hyper_dims = [256,256,256,648]
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:]) #[(256,256),(256,256),(256,1024)]
        )
        self.vbr = vbr
        self.prior_x = FlexiblePrior(self.hyper_dims[-1])
        self.prior_y = FlexiblePrior(self.hyper_dims[-1])
        # self.vl_fusion1 = CrossAttentionFusion(256,256,1024,256,256,h=8)
        # self.vl_fusion2 = CrossAttentionFusion(192,192,1024,192,192,h=16)

        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", revision=None, variant=None
        )

        self.blip2 = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
        self.processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
        self.latent_proj = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                         nn.Conv2d(128, 4, kernel_size=1), 
                                        # LayerNorm(4), nn.ReLU(),                 
        )
        self.latent_proj_back = nn.Sequential(nn.Conv2d(4, 128, kernel_size=1),
                                         nn.Conv2d(128, 256, kernel_size=1),   
                                        # LayerNorm(256), nn.ReLU(),                     
        )
        self.unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        self.noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", beta_schedule="linear",num_train_timesteps=1000)
        self.vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", revision=None, variant=None)
        
    def get_extra_loss(self):
        return self.prior_x.get_extraloss()+self.prior_y.get_extraloss()

    def build_network(self):
        self.enc_x = nn.ModuleList([])
        self.enc_y = nn.ModuleList([])
        self.dec_x = nn.ModuleList([])
        self.dec_xy = nn.ModuleList([])       
        self.dec_y = nn.ModuleList([])
        self.dec_x_cond = nn.ModuleList([])       
        self.dec_y_cond = nn.ModuleList([])
        self.hyper_enc_x = nn.ModuleList([])
        self.hyper_dec_x = nn.ModuleList([])
        self.hyper_enc_y = nn.ModuleList([])
        self.hyper_dec_y = nn.ModuleList([])


    def encode_x(self, input, cond=None):
        for i, (resnet, vbrscaler, down) in enumerate(self.enc_x):
            input = resnet(input)
            if self.vbr:
                input = vbrscaler(input, cond)
            input = down(input)
        latent = input
        for i, (conv, vbrscaler, act) in enumerate(self.hyper_enc_x):
            input = conv(input)
            if self.vbr and i != (len(self.hyper_enc_x) - 1):
                input = vbrscaler(input, cond)
            input = act(input)
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior_x.medians)
        input = q_hyper_latent
        for i, (deconv, vbrscaler, act) in enumerate(self.hyper_dec_x):
            input = deconv(input)
            if self.vbr and i != (len(self.hyper_dec_x) - 1):
                input = vbrscaler(input, cond)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp

    def encode_y(self, input, cond=None):
        for i, (resnet, vbrscaler, down) in enumerate(self.enc_y):
            input = resnet(input)
            if self.vbr:
                input = vbrscaler(input, cond)
            input = down(input)
        latent = input
        for i, (conv, vbrscaler, act) in enumerate(self.hyper_enc_y):
            input = conv(input)
            if self.vbr and i != (len(self.hyper_enc_w) - 1):
                input = vbrscaler(input, cond)
            input = act(input)
        hyper_latent = input
        # q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
       
        for i, (deconv, vbrscaler, act) in enumerate(self.hyper_dec_y):
            input = deconv(input)
            if self.vbr and i != (len(self.hyper_dec_y) - 1):
                input = vbrscaler(input, cond)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        # q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return latent, hyper_latent, state4bpp

    def encode_w(self, x): 
    
        capation={}
        # x = x*0.5+0.5
        x = x* 127.5 + 127.5
        # x = (x+1)/2
        x.to(device)
        # self.processor.to(device)
        self.blip2.to(device)
        self.text_encoder.to(device)
        inputs = self.processor(images=x, return_tensors="pt")

        inputs.to(device)

        generated_ids = self.blip2.generate(**inputs, max_length=32)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text_bpp = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # print(generated_text)
        tokenized_captions = tokenize_captions(generated_text)
        capation['ids'] = tokenized_captions.to(device)
        # capation['ids'] = tokenized_captions
        # print(capation['ids'])
        encoder_hidden_states = self.text_encoder(capation['ids'], return_dict=False)[0]
        # print('encoder_hidden_states:',encoder_hidden_states.shape)
        # encoder_hidden_states = encoder_hidden_states*2-1
        return encoder_hidden_states,generated_text_bpp


    def decode_xy(self, x, y, w=None,cond=None):

        output = []
        output_y = []
        noise_pred = y
        noise_org = y

            # # print("x:", x.shape)  # [4,256,8,16]
            # # print("w:", w.shape)  # [4,77,1024]
        if cond == None:
            self.unet.to(device1)
            self.latent_proj.to(device1)
            self.latent_proj_back.to(device1)
            self.vae.to(device1)
            self.dec_xy.to(device0)
            self.dec_y.to(device0)

            latents = self.latent_proj(x)
            latents = latents * self.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0,self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            if self.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            model_pred = self.unet(noisy_latents, timesteps, w, x, return_dict=False)[0]

            latents = get_pred_original_sample(self.noise_scheduler, timesteps, noisy_latents, model_pred, self.vae)
            noisy_latents = noisy_latents.to(device0)
            model_pred = model_pred.to(device0)
            noise_pred = model_pred
            target = target.to(device0)
            noise_org = target        
            Lx = self.latent_proj_back(latents) # global reconstruction (GR)
            #x = Lx
            Lx = Lx.to(device0)
            
            x = x.to(device0)
            x = x+Lx # the residual information
            # x = Lx

        x = x.to(device0)
        output.append(x)
        output_y.append(y)
        x = torch.cat((y,x),1)

        for i, ((resnet_x, vbrscaler_x, down_x), (resnet_y, vbrscaler_y, down_y)) in enumerate(zip(self.dec_xy, self.dec_y)):

            x = resnet_x(x)
            if self.vbr:
                x = vbrscaler_x(x, cond)
            x = down_x(x)
            y = resnet_y(y)
            if self.vbr:
                y = vbrscaler_y(y, cond)
            y = down_y(y) 
                      
            output.append(x)
            # print(x.shape)
            output_y.append(y)
    
            x = torch.cat((x,y),1) # local constrain (LC)

        return output[::-1],output_y[::-1],noise_pred,noise_org
    
    def compress_text(self,input_text):
        """Compress the input text to bytes using zlib."""
        input_bytes = input_text.encode('utf-8')
        return zlib.compress(input_bytes, level=zlib.Z_BEST_COMPRESSION)
    def calculate_bpp(self,compressed_data, num_pixels, bytes=True, num_bytes=None):
        """Calculate bpp given the compressed text and number of pixels."""
        scaling_factor = 8 if bytes else 1
        if num_bytes:
            return num_bytes * scaling_factor / num_pixels
        return len(compressed_data) * scaling_factor / num_pixels

    def bpp(self, shape, state4bpp_x, state4bpp_y):
        B, _, H, W = shape
        
        latent_x = state4bpp_x["latent"]
        hyper_latent_x = state4bpp_x["hyper_latent"]
        latent_distribution_x = state4bpp_x["latent_distribution"]

        latent_y = state4bpp_y["latent"]
        hyper_latent_y = state4bpp_y["hyper_latent"]
        latent_distribution_y = state4bpp_y["latent_distribution"]

        if self.training:
            q_hyper_latent_x = quantize(hyper_latent_x, "noise")
            q_latent_x = quantize(latent_x, "noise")

        else:
            q_hyper_latent_x = quantize(hyper_latent_x, "dequantize", self.prior_x.medians)
            q_latent_x = quantize(latent_x, "dequantize", latent_distribution_x.mean)

        hyper_rate_x = -self.prior_x.likelihood(q_hyper_latent_x).log2()
        cond_rate_x = -latent_distribution_x.likelihood(q_latent_x).log2()
        bpp_x = (hyper_rate_x.sum(dim=(1, 2, 3)) + cond_rate_x.sum(dim=(1, 2, 3))) / (H * W)

        hyper_rate_y = -self.prior_y.likelihood(hyper_latent_y).log2()
        cond_rate_y = -latent_distribution_y.likelihood(latent_y).log2()
        bpp_y = (hyper_rate_y.sum(dim=(1, 2, 3)) + cond_rate_y.sum(dim=(1, 2, 3))) / (H * W)

        return bpp_x,bpp_y

    def forward(self, input_x,input_y, cond=None):
        q_latent, q_hyper_latent, state4bpp = self.encode_x(input_x, cond)
        latent_y, hyper_latent_y, state4bpp_y = self.encode_y(input_y, cond) 
        w,text_y= self.encode_w(input_y)
        byte_stream_text = self.compress_text(text_y)
        bpp_texty = self.alculate_bpp(byte_stream_text, 128*256)
        # latent_w, hyper_latent_w, state4bpp_w = self.encode_w(input_y, cond)
        bpp_x,bpp_y = self.bpp(input_x.shape, state4bpp, state4bpp_y)
        output,output_y = self.decode_xy(q_latent, latent_y, w, cond)
        return {
            "output": output,
            "bpp": bpp_x,
            "bpp_y":bpp_y,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
            "output_y":output_y,

        }


class BigCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 3, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__(dim, dim_mults, hyper_dims_mults, channels, out_channels, vbr)
        self.build_network()

    def build_network(self):

        self.enc_x = nn.ModuleList([])
        self.enc_w = nn.ModuleList([])
        self.enc_y = nn.ModuleList([])
        self.dec_x = nn.ModuleList([])  
        self.dec_x_cond = nn.ModuleList([])      
        self.dec_y = nn.ModuleList([])
        self.dec_y_cond = nn.ModuleList([])
        self.hyper_enc_x = nn.ModuleList([])
        self.hyper_dec_x = nn.ModuleList([])
        self.dec_xy = nn.ModuleList([])
        self.dec_xy_cond = nn.ModuleList([])
        self.hyper_enc_y = nn.ModuleList([])
        self.hyper_dec_y = nn.ModuleList([])
        # self.hyper_enc_w = nn.ModuleList([])
        # self.hyper_dec_w = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc_x.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        VBRCondition(1, dim_out) if self.vbr else nn.Identity(),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc_y.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        VBRCondition(1, dim_out) if self.vbr else nn.Identity(),
                        Downsample(dim_out),
                    ]
                )
            )

   

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            in0 = ind==0            
            self.dec_xy.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in*2, dim_out if not is_last else dim_in),
                        # ResnetBlock(dim_in, dim_out if not is_last else dim_in),                        
                        VBRCondition(1, dim_out if not is_last else dim_in)
                        if self.vbr
                        else nn.Identity(),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

     
        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            in0 = ind==0
            self.dec_x.append(
                nn.ModuleList(
                    [
                        # ResnetBlock(dim_in*2 if ind>=1 else dim_in, dim_out if not is_last else dim_in),
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        VBRCondition(1, dim_out if not is_last else dim_in)
                        if self.vbr
                        else nn.Identity(),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            in0 = ind==0
            self.dec_y.append(
                nn.ModuleList(
                    [
                        # ResnetBlock(dim_in*2 if in0 else dim_in, dim_out if not is_last else dim_in),
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        VBRCondition(1, dim_out if not is_last else dim_in)
                        if self.vbr
                        else nn.Identity(),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )


        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc_x.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec_x.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc_y.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec_y.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

  




class SimpleCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__(dim, dim_mults, hyper_dims_mults, channels, out_channels, vbr)
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        GDN1(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        GDN1(dim_out, True) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )