'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import torch.nn as nn
import torch


# Removed dropout and changed the transition up layers in the original implementation
# to mitigate the grid patterns of the network output
class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))

    def forward(self, x):
        return super(DenseLayer, self).forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            # we pass all previous activations into each dense layer normally
            # But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)  # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super(TransitionDown, self).forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                       nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop_(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super(Bottleneck, self).forward(x)


def center_crop_(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, feature_length=256):
        super(FCDenseNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        # First Convolution
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        # Final DenseBlock
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=feature_length, kernel_size=1, stride=1,
                                   padding=0, bias=True)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = out / torch.norm(out, dim=1, keepdim=True)

        return out


class FeatureResponseGenerator(nn.Module):
    def __init__(self, scale=20.0, threshold=0.9):
        super(FeatureResponseGenerator, self).__init__()
        self.scale = scale
        self.threshold = threshold

    def forward(self, x):
        source_feature_map, target_feature_map, source_feature_1D_locations, boundaries = x

        # source_feature_map: B x C x H x W
        # source_feature_1D_locations: B x Sampling_size x 1
        batch_size, channel, height, width = source_feature_map.shape
        _, sampling_size, _ = source_feature_1D_locations.shape
        # B x C x Sampling_size
        source_feature_1D_locations = source_feature_1D_locations.view(batch_size, 1,
                                                                       sampling_size).expand(-1, channel, -1)
        # Extend 1D locations to B x C x Sampling_size
        # B x C x Sampling_size
        sampled_feature_vectors = torch.gather(source_feature_map.view(batch_size, channel, height * width), 2,
                                               source_feature_1D_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(batch_size, channel, sampling_size, 1,
                                                               1).permute(0, 2, 1, 3, 4).view(batch_size,
                                                                                              sampling_size,
                                                                                              channel,
                                                                                              1, 1)

        # Do convolution on target_feature_map with the sampled_feature_vectors as the kernels
        # We use the sampled feature vectors in a convolution operation where BC is the input channel dim and
        # Sampling_size as the output channel dim.
        temp = [None for _ in range(batch_size)]
        for i in range(batch_size):
            temp[i] = torch.nn.functional.conv2d(input=target_feature_map[i].view(1, channel, height, width),
                                                 weight=sampled_feature_vectors[i].view(sampling_size, channel,
                                                                                        1,
                                                                                        1),
                                                 padding=0)
        # B x Sampling_size x H x W
        cosine_distance_map = 0.5 * torch.cat(temp, dim=0) + 0.5
        # Normalized cosine distance map
        # B x Sampling_size x H x W
        cosine_distance_map = torch.exp(self.scale * (cosine_distance_map - self.threshold))
        cosine_distance_map = cosine_distance_map / torch.sum(cosine_distance_map, dim=(2, 3), keepdim=True)

        return cosine_distance_map


class FeatureResponseGeneratorNoSoftThresholding(nn.Module):
    def __init__(self):
        super(FeatureResponseGeneratorNoSoftThresholding, self).__init__()

    def forward(self, x):
        source_feature_map, target_feature_map, source_feature_1D_locations, boundaries = x

        # source_feature_map: B x C x H x W
        # source_feature_1D_locations: B x Sampling_size x 1
        batch_size, channel, height, width = source_feature_map.shape
        _, sampling_size, _ = source_feature_1D_locations.shape
        # B x C x Sampling_size
        source_feature_1D_locations = source_feature_1D_locations.view(batch_size, 1,
                                                                       sampling_size).expand(-1, channel, -1)
        # Extend 1D locations to B x C x Sampling_size
        # B x C x Sampling_size
        sampled_feature_vectors = torch.gather(source_feature_map.view(batch_size, channel, height * width), 2,
                                               source_feature_1D_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(batch_size, channel, sampling_size, 1,
                                                               1).permute(0, 2, 1, 3, 4).view(batch_size,
                                                                                              sampling_size,
                                                                                              channel,
                                                                                              1, 1)

        # Do convolution on target_feature_map with the sampled_feature_vectors as the kernels
        # We use the sampled feature vectors in a convolution operation where BC is the input channel dim and
        # Sampling_size as the output channel dim.
        temp = [None for _ in range(batch_size)]
        for i in range(batch_size):
            temp[i] = torch.nn.functional.conv2d(input=target_feature_map[i].view(1, channel, height, width),
                                                 weight=sampled_feature_vectors[i].view(sampling_size, channel,
                                                                                        1, 1), padding=0)
        # B x Sampling_size x H x W
        cosine_distance_map = torch.cat(temp, dim=0)
        return cosine_distance_map
