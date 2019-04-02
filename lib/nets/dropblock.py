#coding:utf-8
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class DropBlock2DMix(nn.Module):
    """
    DropBlock with mixing
    """
    def __init__(self, block_size = 3, drop_prob = 1., test=False, extra_mix=True):
        super(DropBlock2DMix, self).__init__()
        print("[*] using Dropblock mix training={}".format(self.training))
        print("[***]  Setting fixed drop_window")
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.test = test
        self.extra_mix = extra_mix
        self.expandsize = 16

    def forward(self, x, index=None, mode='TRAIN'):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if mode == 'TEST' :  #or self.drop_prob == 0.
            #print("dropblock no training {}".format(self.training))
            #raise ValueError("Dropblock mix, drop_prob > 0 ?")
            return x, None, None
        elif mode == 'TRAIN':
            # sample from a mask
            mask_reduction = self.block_size // 2
            mask_height = x.shape[-2] - mask_reduction
            mask_width = x.shape[-1] - mask_reduction
            mask_sizes = [mask_height, mask_width]

            if any([x <= 0 for x in mask_sizes]):
                raise ValueError('Input of shape {} is too small for block_size {}'
                                 .format(tuple(x.shape), self.block_size))

            # get gamma value
            # gamma = self._compute_gamma(x, mask_sizes)
            # if self.test: print("--- gamma ---\n", gamma)
            # # sample mask
            # mask = Bernoulli(gamma).sample((x.shape[0], *mask_sizes))
            # if self.test: print("---  mask ---\n", mask)
            bs = x.shape[0]
            hw = mask_width
            #rads = torch.randint(0, hw * hw, (bs,)).long()
            rads = torch.randint(0, hw * hw, (1,)).long().repeat(bs)
            rads = torch.unsqueeze(rads, 1)
            mask = torch.zeros(bs, hw*hw).scatter_(1, rads, 1).reshape((bs,hw,hw))

            # place mask on input device
            mask = mask.to(x.device)   # mask.cuda()

            # compute block mask
            block_mask = self._compute_block_mask(mask)
            if self.test: print("--- block mask ---\n", block_mask)

            # apply block mask
            # out = x * block_mask[:, None, :, :]

            batch_size = x.size()[0]
            if index.any() == None:
                index = torch.randperm(batch_size).cuda()
            verse_mask = torch.ones_like(block_mask) - block_mask
            if self.test: print("--- verse_mask ---", verse_mask)


            if self.extra_mix:
                lam = 1 - 0.99
                out = x * block_mask[:, None, :, :] * (1 - lam) + x * verse_mask[:, None, :, :] * lam + \
                      x[index, :] * block_mask[:, None, :, :] * (lam) + \
                      x[index, :] * verse_mask[:, None, :, :] * (1 - lam)
            else:
                out = x * block_mask[:, None, :, :] + \
                      x[index, :] * verse_mask[:, None, :, :] #* 0.1 这里需注意，是否加0.1
            # if self.test: out = x * block_mask[:, None, :, :] + x[index, :] * verse_mask[:, None, :, :] * 0.1
            # scale output
            # out = out * block_mask.numel() / block_mask.sum()

            mask_sight = F.conv2d(block_mask[0, None, :, :].unsqueeze(0), torch.ones((1, 1, self.expandsize, self.expandsize)).to(mask.device), padding=(self.expandsize - 1)).sum().item()
            vmask_sight = F.conv2d(verse_mask[0, None, :, :].unsqueeze(0), torch.ones((1, 1, self.expandsize, self.expandsize)).to(mask.device), padding=(self.expandsize - 1)).sum().item()
            # mask_sight = F.conv2d(block_mask[0, None, :, :].unsqueeze(0), torch.ones((1, 1, self.expandsize, self.expandsize)), padding=(self.expandsize - 1)).sum().item()
            # vmask_sight = F.conv2d(verse_mask[0, None, :, :].unsqueeze(0), torch.ones((1, 1, self.expandsize, self.expandsize)), padding=(self.expandsize - 1)).sum().item()

            lam = mask_sight / (mask_sight + vmask_sight)

            return out, index, lam

    def _compute_block_mask(self, mask):
        block_mask = F.conv2d(mask[:, None, :, :],
                              torch.ones((1, 1, self.block_size, self.block_size)).to(
                                  mask.device),
                              padding=int(np.ceil(self.block_size // 2) + 1))

        delta = self.block_size // 2
        input_height = mask.shape[-2] + delta
        input_width = mask.shape[-1] + delta

        height_to_crop = block_mask.shape[-2] - input_height
        width_to_crop = block_mask.shape[-1] - input_width

        if height_to_crop != 0:
            block_mask = block_mask[:, :, :-height_to_crop, :]

        if width_to_crop != 0:
            block_mask = block_mask[:, :, :, :-width_to_crop]

        block_mask = (block_mask >= 1).to(device=block_mask.device, dtype=block_mask.dtype)
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x, mask_sizes):
        feat_area = x.shape[-2] * x.shape[-1]
        mask_area = mask_sizes[-2] * mask_sizes[-1]
        return (self.drop_prob / (self.block_size ** 2)) * (feat_area / mask_area)
