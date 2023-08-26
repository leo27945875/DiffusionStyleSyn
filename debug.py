from utils import *


def TestModelArch():
    import torch
    from diffusers import UNet2DConditionModel

    model = UNet2DConditionModel(
        in_channels=3, out_channels=3, block_out_channels = (32, 32, 64, 64), cross_attention_dim = (32, 32, 64, 64), norm_num_groups = 4, projection_class_embeddings_input_dim=128,
        class_embed_type = "simple_projection", class_embeddings_concat = True, resnet_time_scale_shift = "ada_group"
    ).cuda()

    model(torch.zeros([1, 3, 64, 64]).cuda(), torch.tensor(10.).cuda(), None, torch.zeros([1, 128]).cuda())


def TestSeperateDiffusion():
    import matplotlib.pyplot as plt
    from edm import EDM
    from edm.seperate import Seperate

    edm = EDM(1000)
    new = Seperate(edm, 3, 1, otherArgs={"sampleMode": "uniform"})

    print(new.sampleMode)
    print(new.sigmaMin, new.sigmaMax)

    a = [edm.IndexToSigma(i) for i in range(edm.nStep)]
    b = [new.IndexToSigma(i) for i in range(new.nStep)]
    plt.plot(a, 'r-')
    plt.plot([i + new.offsetStep for i in range(new.nStep)], b, 'b-')
    plt.show()


def TestVQGAN():
    import matplotlib.pyplot as plt
    from torchinfo import summary
    from model.extractor import VQGAN

    vqgan = VQGAN()
    vqgan.requires_grad_(False)
    vqgan.eval()

    summary(vqgan)
    print(f"VQGAN encoder input  channel = {vqgan.inChannel}")
    print(f"VQGAN encoder output channel = {vqgan.outChannel}")

    image = vqgan.GetPreprocess()(image=ReadRGBImage("data_80/image/ADE_train_00020130.jpg"))["image"]

    latent = vqgan.EncodeNotQuantizedLatent(image.unsqueeze(0))
    print(latent.shape)
    recons = vqgan.DecodeNotQuantizedLatent(latent)
    print(recons.shape)

    plt.imshow(((recons[0] + 1) / 2).clip(0., 1.).cpu().numpy().transpose(1,2,0))
    plt.show()


def TestTrySomething():
    a = 5
    match a:
        case 1:
            b=1
        case 2:
            b=2
        case _:
            assert False, "AAAAAAAAAAAAAA"
    
    print(b)


if __name__ == '__main__':


    TestTrySomething()