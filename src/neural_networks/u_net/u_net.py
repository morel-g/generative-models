# from diffusers import UNet2DModel
from src.neural_networks.u_net.u_net_2d_model import UNet2DModel
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        image_channels=3,
        n_resblocks=8,
        n_channels=128,
        ch_mult=(1, 2, 2, 2),
        block_types=None,
        embedding_type="positional",
        norm_num_groups=32,
    ):
        super().__init__()
        block_out_channels = tuple(n_channels * cm for cm in ch_mult)
        if block_types is None:
            block_types = ["Block2D"] * len(ch_mult)
            block_types[-2] = "AttnBlock2D"
            block_types = tuple(block_types)

        self.model = UNet2DModel(
            sample_size=dim,  # the target image resolution
            in_channels=image_channels,  # the number of input channels, 3 for RGB images
            out_channels=image_channels,  # the number of output channels
            time_embedding_type=embedding_type,
            layers_per_block=n_resblocks,  # how many ResNet layers to use per UNet block
            block_out_channels=tuple(
                n_channels * cm for cm in ch_mult
            ),  # the number of output channes for each UNet block
            # block_out_channels=block_out_channels,
            down_block_types=tuple(
                bt.replace("Block2D", "DownBlock2D") for bt in block_types
            ),
            up_block_types=tuple(
                bt.replace("Block2D", "UpBlock2D") for bt in block_types
            )[::-1],
            norm_num_groups=norm_num_groups,
        )

    def forward(self, x, time):
        return self.model(x, time, return_dict=False)[0]
