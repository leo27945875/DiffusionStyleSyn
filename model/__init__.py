from .extractor import Extractor, CLIPImageEncoder, VQGAN, ExtractorPlaceholder
from .ensembler import Ensembler
from .unet      import MyUNet, PrecondUNet
from .ema       import ModuleEMA

from type_alias import T_Precond_Func


def BuildModel(PreconditionFunc: T_Precond_Func, nClass: int, baseChannel: int, attnChannel: int, extractorOutChannel: int, extractorCrossAttnChannel: int | None = None):
    """
    Build diffusion model arch in CPU.
    """
    return PrecondUNet(
        GetPrecondSigmas                      = PreconditionFunc,
        in_channels                           = 3 + nClass,
        out_channels                          = 3,
        block_out_channels                    = (baseChannel, baseChannel * 2, baseChannel * 3, baseChannel * 4),
        down_block_types                      = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types                        = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim                    = attnChannel,
        resnet_time_scale_shift               = "ada_group",          # "default", "scale_shift", "ada_group", "spatial"
        class_embed_type                      = "simple_projection",
        class_embeddings_concat               = True,
        cross_attention_dim                   = extractorCrossAttnChannel or (baseChannel, baseChannel * 2, baseChannel * 3, baseChannel * 4),
        projection_class_embeddings_input_dim = extractorOutChannel
    )