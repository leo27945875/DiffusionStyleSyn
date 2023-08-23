def TestModelArch():
    import torch
    import torch.nn.functional as F
    from diffusers import UNet2DConditionModel

    model = UNet2DConditionModel(
        in_channels=3, out_channels=3, block_out_channels = (32, 32, 64, 64), cross_attention_dim = (32, 32, 64, 64), norm_num_groups = 4, projection_class_embeddings_input_dim=128,
        class_embed_type = "simple_projection", class_embeddings_concat = True, resnet_time_scale_shift = "ada_group"
    ).cuda()

    model(torch.zeros([1, 3, 64, 64]).cuda(), torch.tensor(10.).cuda(), None, torch.zeros([1, 128]).cuda())


def TestSeperateDiffusion():
    from edm import EDM
    from edm.seperate import Seperate
    import matplotlib.pyplot as plt

    edm = EDM(1000)
    new = Seperate(edm, 3, 1, otherArgs={"sampleMode": "uniform"})

    print(new.sampleMode)
    print(new.sigmaMin, new.sigmaMax)

    a = [edm.IndexToSigma(i) for i in range(edm.nStep)]
    b = [new.IndexToSigma(i) for i in range(new.nStep)]
    plt.plot(a, 'r-')
    plt.plot([i + new.offsetStep for i in range(new.nStep)], b, 'b-')
    plt.show()


def TestTrySomething():
    import torch

    a = torch.tensor([3.5, 3])
    print(a.dim())


if __name__ == '__main__':


    TestTrySomething()