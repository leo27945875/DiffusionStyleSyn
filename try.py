import torch
import torch.nn.functional as F

from diffusers import UNet2DConditionModel


model = UNet2DConditionModel(
    in_channels=3, out_channels=3, block_out_channels = (32, 32, 64, 64), cross_attention_dim = (32, 32, 64, 64), norm_num_groups = 4, projection_class_embeddings_input_dim=128,
    class_embed_type = "simple_projection", class_embeddings_concat = True, resnet_time_scale_shift = "ada_group"
).cuda()

