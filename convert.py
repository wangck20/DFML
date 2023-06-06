#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
from collections import OrderedDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MoCo Pre-Traind Model to DEiT')
    parser.add_argument('--input', default='', type=str, metavar='PATH', required=True,
                        help='path to moco pre-trained checkpoint')
    parser.add_argument('--output', default='', type=str, metavar='PATH', required=True,
                        help='path to output checkpoint in DEiT format')
    args = parser.parse_args()

    # load input
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint['model']
    new_ckpt = OrderedDict()
    # print(state_dict.keys())
    for k in list(state_dict.keys()):
        if 'block' in k:
            if 'attn' in k:
                if 'qkv' in k:
                    if 'weight' in k:
                        value = state_dict[k].view(9, -1, state_dict[k].size()[1])
                        new_ckpt[k[0:-7]+'1.weight'] = torch.cat((value[0], value[3], value[6]), dim=0)
                        new_ckpt[k[0:-7]+'2.weight'] = torch.cat((value[1], value[4], value[7]), dim=0)
                        new_ckpt[k[0:-7]+'3.weight'] = torch.cat((value[2], value[5], value[8]), dim=0)
                    elif 'bias' in k:
                        value = state_dict[k].view(9, -1)
                        new_ckpt[k[0:-5]+'1.bias'] = torch.cat((value[0], value[3], value[6]), dim=0)
                        new_ckpt[k[0:-5]+'2.bias'] = torch.cat((value[1], value[4], value[7]), dim=0)
                        new_ckpt[k[0:-5]+'3.bias'] = torch.cat((value[2], value[5], value[8]), dim=0)
                elif 'proj' in k:
                    if 'weight' in k:
                        value = state_dict[k].view(state_dict[k].size()[0], 3, -1)
                        new_ckpt[k[0:-7]+'1.weight'] = value[:, 0, :]
                        new_ckpt[k[0:-7]+'2.weight'] = value[:, 1, :]
                        new_ckpt[k[0:-7]+'3.weight'] = value[:, 2, :]
                    elif 'bias' in k:
                        new_ckpt[k[0:-5]+'1.bias'] = state_dict[k]/3
                        new_ckpt[k[0:-5]+'2.bias'] = state_dict[k]/3
                        new_ckpt[k[0:-5]+'3.bias'] = state_dict[k]/3
            elif 'mlp' in k:
                if 'fc1' in k:
                    if 'weight' in k:
                        value = state_dict[k].view(3, -1, state_dict[k].size()[1])
                        new_ckpt[k[0:-11]+'1.fc1.weight'] = value[0, :, :]
                        new_ckpt[k[0:-11]+'2.fc1.weight'] = value[1, :, :]
                        new_ckpt[k[0:-11]+'3.fc1.weight'] = value[2, :, :]
                    elif 'bias' in k:
                        value = state_dict[k].view(3, -1)
                        new_ckpt[k[0:-9]+'1.fc1.bias'] = value[0]
                        new_ckpt[k[0:-9]+'2.fc1.bias'] = value[1]
                        new_ckpt[k[0:-9]+'3.fc1.bias'] = value[2]
                elif 'fc2' in k:
                    if 'weight' in k:
                        value = state_dict[k].view(state_dict[k].size()[0], 3, -1)
                        new_ckpt[k[0:-11]+'1.fc2.weight'] = value[:, 0, :]
                        new_ckpt[k[0:-11]+'2.fc2.weight'] = value[:, 1, :]
                        new_ckpt[k[0:-11]+'3.fc2.weight'] = value[:, 2, :]
                    elif 'bias' in k:
                        new_ckpt[k[0:-9]+'1.fc2.bias'] = state_dict[k]/3
                        new_ckpt[k[0:-9]+'2.fc2.bias'] = state_dict[k]/3
                        new_ckpt[k[0:-9]+'3.fc2.bias'] = state_dict[k]/3
            else:
                new_ckpt[k] = state_dict[k]
        else:
            new_ckpt[k] = state_dict[k]
    print(new_ckpt.keys())
        # print(k)
        # print(state_dict[k])
        # retain only base_encoder up to before the embedding layer
        # if k.startswith('norm'):
        #     state_dict["fc_"+k] = state_dict[k]
        # #if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
        #     # remove prefix
        #     #state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # # delete renamed or unused k
        #     del state_dict[k]

    # make output directory if necessary
    #output_dir = os.path.dirname(args.output)
    #if not os.path.isdir(output_dir):
    #    os.makedirs(output_dir)
    # save to output
    torch.save({'model': new_ckpt}, args.output)
