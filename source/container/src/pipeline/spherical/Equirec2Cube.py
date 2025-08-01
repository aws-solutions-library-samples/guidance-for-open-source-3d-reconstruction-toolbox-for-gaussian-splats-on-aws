# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY
#
# Adapted from https://github.com/fuenwang/PanoramaUtility

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Equirec2Cube(nn.Module):
    def __init__(self, equ_h, equ_w, cube_length, FOV=90, RADIUS=128, CUDA=False):
        super(Equirec2Cube, self).__init__()
        batch_size = 1
        R_lst = []
        theta_lst = np.array([-90, 0, 90, 180], float) / 180 * np.pi
        phi_lst = np.array([90, -90], float) / 180 * np.pi
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.CUDA = CUDA
        for theta in theta_lst:
            angle_axis = theta * np.array([0, 1, 0], float)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)

        for phi in phi_lst:
            angle_axis = phi * np.array([1, 0, 0], float)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)
        
        R_lst = [Variable(torch.FloatTensor(x)) for x in R_lst]
        
        self.cube_length = cube_length
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0
        c_x = (cube_length - 1) / 2.0
        c_y = (cube_length - 1) / 2.0
        
        wangle = (180 - FOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(FOV / 2.0)) / np.sin(np.radians(wangle))

        f = RADIUS / w_len * cube_length
        cx = c_x
        cy = c_y
        self.intrisic = {
                    'f': float(f),
                    'cx': float(cx),
                    'cy': float(cy)
                }

        interval = w_len / (cube_length - 1) 
        
        z_map = np.zeros([cube_length, cube_length], float) + RADIUS
        x_map = np.tile((np.arange(cube_length) - c_x) * interval, [cube_length, 1])
        y_map = np.tile((np.arange(cube_length) - c_y) * interval, [cube_length, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([cube_length, cube_length, 3], float)
        xyz[:, :, 0] = (RADIUS / D) * x_map[:, :]
        xyz[:, :, 1] = (RADIUS / D) * y_map[:, :]
        xyz[:, :, 2] = (RADIUS / D) * z_map[:, :]
        xyz = Variable(torch.FloatTensor(xyz))
        
        reshape_xyz = xyz.view(cube_length * cube_length, 3).transpose(0, 1)
        self.batch_size = batch_size # NOTE: Might give an error when batch_size smaller than real batch_size of the batch input
        self.loc = []
        self.grid = []
        for i, R in enumerate(R_lst):
            result = torch.matmul(R, reshape_xyz).transpose(0, 1)
            tmp_xyz = result.contiguous().view(1, cube_length, cube_length, 3)
            self.grid.append(tmp_xyz)
            lon = torch.atan2(result[:, 0] , result[:, 2]).view(1, cube_length, cube_length, 1) / np.pi
            lat = torch.asin(result[:, 1] / RADIUS).view(1, cube_length, cube_length, 1) / (np.pi / 2)

            self.loc.append(torch.cat([lon.repeat(batch_size, 1, 1, 1), lat.repeat(batch_size, 1, 1, 1)], dim=3))

        new_lst = [3, 5, 1, 0, 2, 4]
        self.R_lst = [R_lst[x] for x in new_lst]
        self.grid_lst = []
        for iii in new_lst:
            grid = self.grid[iii].clone()
            scale = self.intrisic['f'] / grid[:, :, :, 2:3]
            grid *= scale
            self.grid_lst.append(grid)

    def _ToCube(self, batch, mode):
        batch_size = batch.size()[0]
        new_lst = [3, 5, 1, 0, 2, 4]
        out = []
        for i in new_lst:
            coor = self.loc[i].cuda() if self.CUDA else self.loc[i]
            result = []
            for ii in range(batch_size):
                tmp = F.grid_sample(batch[ii:ii+1], coor, mode=mode, align_corners=True)
                result.append(tmp)
            result = torch.cat(result, dim=0)
            out.append(result)
        return out

    def GetGrid(self):
        #lst = ['left', 'front', 'right', 'back', 'up', 'down']
        new_lst = [3, 5, 1, 0, 2, 4]
        out = [self.grid[x] for x in new_lst]
        out = torch.cat(out, dim=0)
        return out

    def ToCubeNumpy(self, batch):
        out = self._ToCube(batch)
        result = [x.data.cpu().numpy() for x in out]
        return result

    def ToCubeTensor(self, batch, mode='bilinear'):
        assert mode in ['bilinear', 'nearest']
        batch_size = batch.size()[0]
        cube = self._ToCube(batch, mode=mode)
        out_batch = None
        for batch_idx in range(batch_size):
            for cube_idx in range(6):
                patch = torch.unsqueeze(cube[cube_idx][batch_idx, :, :, :], 0)
                if out_batch is None:
                    out_batch = patch
                else:
                    out_batch = torch.cat([out_batch, patch], dim=0)
        return out_batch

    def forward(self,  batch, mode='bilinear'):
        return self.ToCubeTensor(batch, mode)
