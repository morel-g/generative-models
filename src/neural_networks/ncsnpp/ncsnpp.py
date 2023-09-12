# Source: https://github.com/nv-tlabs/CLD-SGM/tree/main/models
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

import torch
import torch.nn as nn
import numpy as np
import functools
from torch.cuda.amp import autocast

from ...case import Case
from . import utils, layers, layerspp, normalization


def string_to_list(string, delimiter=",", type=int):
    return list(map(type, string.split(delimiter)))


def string_to_tuple(string, delimiter=",", type=int):
    return tuple(string_to_list(string, delimiter, type))


ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
# get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


def get_act(act):
    if act == Case.silu:
        return nn.SiLU()
    elif act == Case.tanh:
        return nn.Tanh()
    else:
        raise RuntimeError("Unknown activation function.")


eps_t = 1e-5


@utils.register_model(name="ncsnpp")
class NCSNpp(nn.Module):
    def __init__(
        self,
        dim,
        act=Case.silu,
        v_input=False,
        image_channels=3,
        n_channels=128,
        ch_mult=(1, 2, 2, 2),
        n_resblocks=8,
        attn_resolutions=(16,),
        dropout=0.1,
        resamp_with_conv=True,
        use_fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="none",
        progressive_input="residual",
        progressive_combine="sum",
        embedding_type="fourier",
        init_scale=0.0,
        fourier_scale=16,
        group_norm=True,
    ):
        super().__init__()

        self.act = get_act(act)

        self.v_input = v_input

        self.n_channels = n_channels
        ch_mult = ch_mult
        self.n_resblocks = n_resblocks
        self.attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = len(ch_mult)
        image_size = dim
        self.all_resolutions = [
            image_size // (2**i) for i in range(self.num_resolutions)
        ]

        fir = use_fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type
        self.progressive = progressive
        self.progressive_input = progressive_input
        self.embedding_type = embedding_type
        self.group_norm = group_norm
        init_scale = init_scale
        combine_method = progressive_combine
        combiner = functools.partial(Combine, method=combine_method)

        channels = image_channels
        if self.v_input:
            channels += image_channels

        modules = []
        if self.embedding_type == "no_embedding":
            temb_dim = None
            channels += 1
        else:
            if self.embedding_type == "fourier":
                modules.append(
                    layerspp.GaussianFourierProjection(
                        embedding_size=self.n_channels, scale=fourier_scale
                    )
                )
                embed_dim = 2 * self.n_channels
                modules.append(nn.Linear(embed_dim, self.n_channels * 4))
            elif self.embedding_type == "fourier_modif":
                modules.append(
                    layerspp.GaussianFourierProjection(
                        embedding_size=self.n_channels, scale=fourier_scale
                    )
                )
                embed_dim = 2 * self.n_channels
                modules.append(nn.Linear(embed_dim, self.n_channels * 4))
                channels += 1
            elif self.embedding_type == "positional":
                embed_dim = self.n_channels
                modules.append(nn.Linear(embed_dim, self.n_channels * 4))
            elif self.embedding_type == "linear":
                modules.append(nn.Linear(1, self.n_channels * 4))
                channels += 1
            else:
                raise NotImplementedError(
                    "Embedding type %s is not implemented."
                    % self.embedding_type
                )

            modules[-1].weight.data = default_initializer()(
                modules[-1].weight.shape
            )
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(self.n_channels * 4, self.n_channels * 4))
            modules[-1].weight.data = default_initializer()(
                modules[-1].weight.shape
            )
            nn.init.zeros_(modules[-1].bias)
            temb_dim = 4 * self.n_channels

        modules.append(conv3x3(channels, self.n_channels))
        hs_c = [self.n_channels]
        in_ch = self.n_channels

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp,
            init_scale=init_scale,
            skip_rescale=self.skip_rescale,
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if self.progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif self.progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample,
                fir=fir,
                fir_kernel=fir_kernel,
                with_conv=True,
            )
        elif self.progressive != "none":
            raise NotImplementedError(
                "Progressive method %s is not implemented." % self.progressive
            )

        if self.progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif self.progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample,
                fir=fir,
                fir_kernel=fir_kernel,
                with_conv=True,
            )
        elif self.progressive_input != "none":
            raise NotImplementedError(
                "Progressive input method %s is not implemented."
                % self.progressive_input
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if self.resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=self.act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=self.skip_rescale,
            )
        elif self.resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=self.act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=self.skip_rescale,
                temb_dim=temb_dim,
            )
        else:
            raise NotImplementedError(
                "ResnetBlock %s is not implemented." % self.resblock_type
            )

        if self.progressive_input != "none":
            input_pyramid_ch = channels

        for i_level in range(self.num_resolutions):
            for _ in range(self.n_resblocks):
                out_ch = self.n_channels * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if self.all_resolutions[i_level] in self.attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if self.progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif self.progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(
                            in_ch=input_pyramid_ch, out_ch=in_ch
                        )
                    )
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.n_resblocks + 1):
                out_ch = self.n_channels * ch_mult[i_level]
                modules.append(
                    ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch)
                )
                in_ch = out_ch

            if self.all_resolutions[i_level] in self.attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(
                                in_ch,
                                image_channels,
                                init_scale=init_scale,
                            )
                        )
                        pyramid_ch = image_channels
                    elif self.progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                else:
                    if self.progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(
                                in_ch,
                                image_channels,
                                bias=True,
                                init_scale=init_scale,
                            )
                        )
                        pyramid_ch = channels
                    elif self.progressive == "residual":
                        modules.append(
                            pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch)
                        )
                        pyramid_ch = in_ch

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if self.progressive != "output_skip":
            if self.group_norm:
                modules.append(
                    nn.GroupNorm(
                        num_groups=min(in_ch // 4, 32),
                        num_channels=in_ch,
                        eps=1e-6,
                    )
                )
            modules.append(
                conv3x3(in_ch, image_channels, init_scale=init_scale)
            )

        self.all_modules = nn.ModuleList(modules)

    def forward(self, input, t):
        modules = self.all_modules
        m_idx = 0

        x = input

        if self.embedding_type == "no_embedding":
            temb = t.reshape(-1, 1, 1, 1).expand(
                (x.shape[0], 1, x.shape[2], x.shape[3])
            )
            x = torch.cat((x, temb), dim=1)
            temb = None
            hs = [x]
        else:
            with autocast(False):
                if self.embedding_type == "fourier":
                    t = t + eps_t
                    temb = modules[m_idx](torch.log(t))
                    m_idx += 1
                elif self.embedding_type == "fourier_modif":
                    x = torch.cat(
                        (
                            x,
                            t.reshape(-1, 1, 1, 1).expand(
                                (x.shape[0], 1, x.shape[2], x.shape[3])
                            ),
                        ),
                        dim=1,
                    )
                    t = t + eps_t
                    temb = modules[m_idx](torch.log(t))
                    m_idx += 1
                elif self.embedding_type == "positional":
                    temb = layers.get_timestep_embedding(t, self.n_channels)
                elif self.embedding_type == "linear":
                    temb = t.unsqueeze(-1)
                    x = torch.cat(
                        (
                            x,
                            t.reshape(-1, 1, 1, 1).expand(
                                (x.shape[0], 1, x.shape[2], x.shape[3])
                            ),
                        ),
                        dim=1,
                    )
                temb = modules[m_idx](temb)
                m_idx += 1
                temb = modules[m_idx](self.act(temb))
                m_idx += 1

        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1

        for i_level in range(self.num_resolutions):
            for _ in range(self.n_resblocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.n_resblocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                            h = pyramid

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == "output_skip":
            h = pyramid
        else:
            if self.group_norm:
                h = self.act(modules[m_idx](h))
                m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)
        return h
