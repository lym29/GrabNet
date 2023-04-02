# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
import os
import trimesh

import time
import numpy as np
import torch

from oikit.oi_shape import OakInkShape

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class LoadOakInkData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 only_params=False,
                 load_on_ram=False
                 ):
        super.__init__()

        self.only_params = only_params

        self.ds_path = os.path.join(dataset_dir, ds_name)
        # self.ds = self._np2torch(os.path.join(self.ds_path, 'grabnet_%s.npz' % ds_name))

        # load the oakink data, use all motion intent and object categories
        self.oi_shape = OakInkShape(data_split=ds_name)

    # def write_obj_info(self):

    def __len__(self):
        return len(self.oi_shape)

    def __getitem__(self, idx):

        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                form_disk = self.load_disk(idx)
                data_out.update(form_disk)
        return data_out

class LoadGRABdata(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='val',
                 dtype=torch.float32):

        super().__init__()

        self.ds_path = os.path.join(dataset_dir, ds_name)
        datasets = glob.glob(self.ds_path+'/*.pt')
        self.ds = self.load(datasets)

        frame_names = np.load(os.path.join(dataset_dir,ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names = [os.path.join(dataset_dir, fname) for fname in frame_names]
        self.frame_sbjs = np.asarray([name.split('/')[-2] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-1].split('_')[0] for name in self.frame_names])

        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(dataset_dir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        # Hand vtemps and betas
        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj] for sbj in self.sbjs]))
        # self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        ## Object data
        self.objs = np.unique(self.frame_objs)
        # self.obj_pts_cloud = torch.from_numpy(np.asarray([sample_points_cloud(self.obj_info[obj]['obj_mesh_file'])
        #                                                   for obj in self.objs])).float()
        self.obj_pts_cloud = torch.from_numpy(np.asarray([self.obj_info[obj]['verts_sample']
                                                          for obj in self.objs])).float()

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

        for idx, name in enumerate(self.objs):
            self.frame_objs[(self.frame_objs == name)] = idx

        self.frame_objs=torch.from_numpy(self.frame_objs.astype(np.int8)).to(torch.long)

        self.ds = self.load_dict()

    def load(self,datasets):
        loaded = {}
        for d in datasets:
            k = os.path.basename(d).split('_')[0]
            loaded[k] = torch.load(d)
        return loaded

    def load_dict(self, source=None):
        if source is None:
            source = self.ds

        out = {}
        for k, v in source.items():
            if isinstance(v,dict):
                out[k] = self.load_dict(v)
            elif isinstance(v, list):
                v = torch.cat(v, dim=0)
                # v = torch.chunk(v, v.shape[0], dim=0)
                out[k] = v
            else:
                out[k] = v

        return out

    def load_idx(self,idx, source=None):

        if source is None:
            source = self.ds

        out = {}
        for k, v in source.items():
            if isinstance(v,dict):
                out[k] = self.load_idx(idx, v)
            elif isinstance(v, list):
                v = torch.cat(v, dim=0)
                # v = torch.chunk(v, v.shape[0], dim=0)
                out[k] = v[idx]
            else:
                out[k] = v[idx]

        return out

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx):

        # data = self.load_idx(idx)
        data = self.ds
        data_out = {}
        data_out['trans_rhand'] = data['rhand']['transl'][idx]
        data_out['global_orient_rhand_rotmat'] = data['rhand']['global_orient'][idx]
        data_out['verts_rhand'] = data['rhand']['verts'][idx]
        data_out['joints_rhand'] = data['rhand']['joints'][idx]
        data_out['rhand_pose'] = data['rhand']['fullpose'][idx]
        data_out['object_contact'] = data['object']['contact'][idx]
        data_out['object_points'] = self.obj_pts_cloud[self.frame_objs[idx]]
        # data_out['object_mesh'] = self.obj_info[self.frame_objs[idx]]['object_mesh']
        data_out['object_global_orient'] = data['object']['global_orient'][idx]
        data_out['object_transl'] = data['object']['transl'][idx]
        data_out['idx'] = torch.from_numpy(np.array(idx, dtype=np.int32))
        data_out['obj_idx'] = torch.from_numpy(np.array(self.frame_objs[idx], dtype=np.int32))
        # data_out['obj_name'] = self.objs[self.frame_objs[idx].item()]
        return data_out

if __name__=='__main__':

    # data_path = '/ghome/l5/ymliu/data/GrabNet_dataset/data'

    oikdata_path = '/ghome/l5/ymliu/data/OakInk/'
    ds = LoadOakInkData(oikdata_path, ds_name='val', only_params=False)

    dataloader = data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=10, drop_last=True)

    s = time.time()
    for i in range(320):
        a = ds[i]
    print(time.time() - s)
    print('pass')

    dl = iter(dataloader)
    s = time.time()
    for i in range(10):
        a = next(dl)
    print(time.time()-s)
    print('pass')

    # mvs = MeshViewers(shape=[1,1])
    #
    # bps_torch = test_ds.bps_torch
    # choice = np.random.choice(range(test_ds.__len__()), 30, replace=False)
    # for frame in choice:
    #     data = test_ds[frame]
    #     rhand = Mesh(v=data['verts_rhand'].numpy(),f=[])
    #     obj = Mesh(v=data['verts_object'].numpy(), f=[], vc=name_to_rgb['blue'])
    #     bps_p = Mesh(v=bps_torch, f=[], vc=name_to_rgb['red'])
    #     mvs[0][0].set_static_meshes([rhand,obj, bps_p])
    #     time.sleep(.4)
    #
    # print('finished')
