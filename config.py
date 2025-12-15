# training config
# n_step = 1000000
n_step = 192000
scheduler_checkpoint_step = 15000
log_checkpoint_step = 197
gradient_accumulate_every = 1
lr = 1e-4
decay = 0.9
minf = 0.5
optimizer = "adamw"  # adamw or adam
n_workers = 4
num_denoise_steps = 20
# load------------------------------------------------------------------------------------------------------------
load_model = True
load_step = False
alpha = 1.0
beta = 0.075
load_beta = 0.075
save_beta = 0.075
device = 0
modal = 'test'

device0 = 2
device1 = 3

save_name = 'pro0'
load_name = 'pro0'
load_idx = 1

loss_type = 'mse'

# diffusion config
pred_mode = 'noise'
loss_type = "l1"
iteration_step = 20000
sample_steps = 200
embed_dim = 64
dim_mults = (1, 2, 3, 4, 5, 6)
hyper_dim_mults = (4, 4, 4)
context_channels = 3
clip_noise = "none"
val_num_of_batch = 1
additional_note = ""
vbr = False
context_dim_mults = (1, 2, 3, 4)
sample_mode = "ddim"
var_schedule = "linear" 
aux_loss_type = "lpips"
compressor = "big"

# dataset----------------------------------------------------------------------------------------
# data_path = 'cityscape_dataset'
# dataset_name = "Cityscape"
data_path = './'
dataset_name = "KITTI_Stereo"
# data_path = './'
# dataset_name = "KITTI_General"

resize = [128,256]


train_batch_size = 16


#---------------------------------------------------------------------------------------
result_root = "./kitti_result_distribute_froze0"
val_path = './kitti_result_distribute_froze0/val_log_alpha1.0_beta0.0108pro0'


# result_root = "./kitti_general_result_distribute_froze"
# val_path = './kitti_general_result_distribute_froze/val_log_alpha1.0_beta0.0106_pro0'


# result_root = "cityscape_result_distribute_froze"
# val_path = './cityscape_result_distribute_froze/val_log_alpha1.0_beta0.0032_pro0'





tensorboard_root = "./result/tensorboard"