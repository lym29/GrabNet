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

import os
import shutil
import sys
sys.path.append('.')
sys.path.append('..')
import json
import numpy as np
import torch
# import mano
from manotorch.manolayer import MANOOutput, ManoLayer

from datetime import datetime

from grabnet.tools.utils import makepath, makelogger, to_cpu
from grabnet.tools.train_tools import EarlyStopping
from grabnet.models.models import CoarseNet, RefineNet
from grabnet.data.dataloader import LoadData
from grabnet.tools.train_tools import point2point_signed

from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from pytorch3d.structures import Meshes
from tensorboardX import SummaryWriter
from oikit.oi_shape.utils import CENTER_IDX


class Trainer:

    def __init__(self,cfg, inference=False):

        
        self.dtype = torch.float32

        # torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)
        logger = makelogger(makepath(os.path.join(cfg.work_dir, '%s.log' % (cfg.expr_ID)), isfile=True)).info
        self.logger = logger

        summary_logdir = os.path.join(cfg.work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger('[%s] - Started training GrabNet, experiment code %s' % (cfg.expr_ID, starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)
        logger('Base dataset_dir is %s' % cfg.dataset_dir)

        # shutil.copy2(os.path.basename(sys.argv[0]), cfg.work_dir)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % cfg.cuda_id if torch.cuda.is_available() else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = torch.cuda.device_count() if cfg.use_multigpu else 1
        if use_cuda:
            logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))



        self.data_info = {}
        self.load_data(cfg, inference)


        with torch.no_grad():
            self.mano_layer = ManoLayer(center_idx=CENTER_IDX, mano_assets_root='assets/mano_v1_2/').to(self.device)
            # self.mano_layer = ManoLayer(center_idx=CENTER_IDX, mano_assets_root='assets/mano_v1_2/').to(self.device)
            
        self.coarse_net = CoarseNet().to(self.device)
        self.refine_net = RefineNet().to(self.device)
        self.refine_net.mano_layer = self.mano_layer

        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.LossL2 = torch.nn.MSELoss(reduction='mean')
        self.KLLoss = lambda mu, logvar: -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if cfg.use_multigpu:
            self.coarse_net = nn.DataParallel(self.coarse_net)
            self.refine_net = nn.DataParallel(self.refine_net)
            logger("Training on Multiple GPU's")

        vars_cnet = [var[1] for var in self.coarse_net.named_parameters()]
        vars_rnet = [var[1] for var in self.refine_net.named_parameters()]

        cnet_n_params = sum(p.numel() for p in vars_cnet if p.requires_grad)
        rnet_n_params = sum(p.numel() for p in vars_rnet if p.requires_grad)
        logger('Total Trainable Parameters for CoarseNet is %2.2f M.' % ((cnet_n_params) * 1e-6))
        logger('Total Trainable Parameters for RefineNet is %2.2f M.' % ((rnet_n_params) * 1e-6))

        self.optimizer_cnet = optim.Adam(vars_cnet, lr=cfg.base_lr, weight_decay=cfg.reg_coef)
        self.optimizer_rnet = optim.Adam(vars_rnet, lr=cfg.base_lr, weight_decay=cfg.reg_coef)

        self.best_loss_cnet = np.inf
        self.best_loss_rnet = np.inf

        self.try_num = cfg.try_num
        self.epochs_completed = 0
        self.cfg = cfg
        self.coarse_net.cfg = cfg


        if cfg.best_cnet is not None:
            self._get_cnet_model().load_state_dict(torch.load(cfg.best_cnet, map_location=self.device), strict=False)
            logger('Restored CoarseNet model from %s' % cfg.best_cnet)
        if cfg.best_rnet is not None:
            self._get_rnet_model().load_state_dict(torch.load(cfg.best_rnet, map_location=self.device), strict=False)
            logger('Restored RefineNet model from %s' % cfg.best_rnet)

        # weights for contact, penetration and distance losses
        self.vpe  = torch.from_numpy(np.load(cfg.vpe_path)).to(self.device).to(torch.long)
        rh_f = self.mano_layer.th_faces.view(1, -1, 3)
        self.rh_f = rh_f.expand(self.cfg.batch_size,-1,-1).to(self.device).to(torch.long)

        v_weights = torch.from_numpy(np.load(cfg.c_weights_path)).to(torch.float32).to(self.device)
        v_weights2 = torch.pow(v_weights, 1.0 / 2.5)
        self.refine_net.v_weights = v_weights
        self.refine_net.v_weights2 = v_weights2
        

        self.v_weights = v_weights
        self.v_weights2 = v_weights2

        self.w_dist = torch.ones([self.cfg.batch_size, self.n_obj_verts]).to(self.device)
        self.contact_v = v_weights > 0.8
        # print(v_weights[self.contact_v])

        self.fit_cnet = False
        self.fit_rnet = False

    def load_data_info(self, ds:LoadData):
        data_info = {}
        data_info['frame_names'] = ds.frame_names
        data_info['frame_sbjs'] = ds.frame_sbjs
        data_info['seq_id'] = ds.seq_id
        return data_info


    def load_data(self,cfg, inference):

        kwargs = {'num_workers': cfg.n_workers,
                  'batch_size':cfg.batch_size,
                  'shuffle':True,
                  'drop_last':True
                  }

        ds_name = 'test'
        ds_test = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name)

        self.data_info['obj_info'] = ds_test.obj_info
        self.data_info['obj_names'] = ds_test.obj_names
        self.data_info[ds_name] = self.load_data_info(ds_test)
        self.ds_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

        # selected = list(range(2*cfg.batch_size))

        if not inference:
            ds_name = 'train'
            ds_train = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name, load_on_ram=cfg.load_on_ram)
            self.data_info[ds_name] = self.load_data_info(ds_train)
            self.data_info['hand_vtmp'] = ds_train.sbj_vtemp
            self.data_info['hand_betas'] = ds_train.sbj_betas
            # ds_train = Subset(ds_train, selected)
            self.ds_train = DataLoader(ds_train, **kwargs)
            

            ds_name = 'val'
            self.data_info[ds_name] = {}
            ds_val = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name, load_on_ram=cfg.load_on_ram)
            self.data_info[ds_name] = self.load_data_info(ds_val)
            # ds_val = Subset(ds_val, selected)
            self.ds_val = DataLoader(ds_val, **kwargs)



            self.logger('Dataset Train, Vald, Test size respectively: %.2f M, %.2f K, %.2f K' %
                   (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset) * 1e-3))

        self.bps = ds_test.bps
        self.n_obj_verts = ds_test[0]['verts_object'].shape[0]

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def _get_cnet_model(self):
        return self.coarse_net.module if isinstance(self.coarse_net, torch.nn.DataParallel) else self.coarse_net

    def save_cnet(self):
        torch.save(self.coarse_net.module.state_dict()
                   if isinstance(self.coarse_net, torch.nn.DataParallel)
                   else self.coarse_net.state_dict(), self.cfg.best_cnet)

    def _get_rnet_model(self):
        return self.refine_net.module if isinstance(self.refine_net, torch.nn.DataParallel) else self.refine_net

    def save_rnet(self):
        torch.save(self.refine_net.module.state_dict()
                   if isinstance(self.refine_net, torch.nn.DataParallel)
                   else self.refine_net.state_dict(), self.cfg.best_rnet)

    def train(self):

        self.coarse_net.train()
        self.refine_net.train()

        save_every_it = len(self.ds_train) / self.cfg.log_every_epoch

        train_loss_dict_cnet = {}
        train_loss_dict_rnet = {}
        torch.autograd.set_detect_anomaly(True)

        for it, dorig in enumerate(self.ds_train):

            dorig = {k: dorig[k].to(self.device) for k in dorig.keys()}

            self.optimizer_cnet.zero_grad()
            self.optimizer_rnet.zero_grad()

            if self.fit_cnet:
                drec_cnet = self.coarse_net(**dorig)
                loss_total_cnet, cur_loss_dict_cnet = self.loss_cnet(dorig, drec_cnet)

                loss_total_cnet.backward()
                self.optimizer_cnet.step()

                train_loss_dict_cnet = {k: train_loss_dict_cnet.get(k, 0.0) + v.item() for k, v in cur_loss_dict_cnet.items()}
                if it % (save_every_it + 1) == 0:
                    cur_train_loss_dict_cnet = {k: v / (it + 1) for k, v in train_loss_dict_cnet.items()}
                    train_msg = self.create_loss_message(cur_train_loss_dict_cnet,
                                                        expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epochs_completed,
                                                        model_name='CoarseNet',
                                                        it=it,
                                                        try_num=self.try_num,
                                                        mode='train')

                    self.logger(train_msg)
                    if self.epochs_completed % 20 == 0:
                        with torch.no_grad():
                            self.vis_training_result(dorig, f'{self.epochs_completed}_{it}')


            if self.fit_rnet:

                params_rnet = self.params_rnet(dorig)
                dorig.update(params_rnet)

                drec_rnet = self.refine_net(**dorig)
                loss_total_rnet, cur_loss_dict_rnet = self.loss_rnet(dorig, drec_rnet)

                loss_total_rnet.backward()
                self.optimizer_rnet.step()

                train_loss_dict_rnet = {k: train_loss_dict_rnet.get(k, 0.0) + v.item() for k, v in cur_loss_dict_rnet.items()}
                if it % (save_every_it + 1) == 0:
                    cur_train_loss_dict_rnet = {k: v / (it + 1) for k, v in train_loss_dict_rnet.items()}
                    train_msg = self.create_loss_message(cur_train_loss_dict_rnet,
                                                        expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epochs_completed,
                                                        model_name='RefineNet',
                                                        it=it,
                                                        try_num=self.try_num,
                                                        mode='train')

                    self.logger(train_msg)

        train_loss_dict_cnet = {k: v / len(self.ds_train) for k, v in train_loss_dict_cnet.items()}
        train_loss_dict_rnet = {k: v / len(self.ds_train) for k, v in train_loss_dict_rnet.items()}

        return train_loss_dict_cnet, train_loss_dict_rnet

    def evaluate(self, ds_name='val'):
        self.coarse_net.eval()
        self.refine_net.eval()

        eval_loss_dict_cnet = {}
        eval_loss_dict_rnet = {}

        data = self.ds_val if ds_name == 'val' else self.ds_test
        # print(len(data))

        with torch.no_grad():
            for dorig in data:
                dorig = {k: dorig[k].to(self.device) for k in dorig.keys()}

                if self.fit_cnet:
                    drec_cnet = self.coarse_net(**dorig)
                    loss_total_cnet, cur_loss_dict_cnet = self.loss_cnet(dorig, drec_cnet)

                    eval_loss_dict_cnet = {k: eval_loss_dict_cnet.get(k, 0.0) + v.item() for k, v in cur_loss_dict_cnet.items()}


                if self.fit_rnet:

                    params_rnet = self.params_rnet(dorig)
                    dorig.update(params_rnet)

                    drec_rnet = self.refine_net(**dorig)
                    loss_total_rnet, cur_loss_dict_rnet = self.loss_rnet(dorig, drec_rnet)

                    eval_loss_dict_rnet = {k: eval_loss_dict_rnet.get(k, 0.0) + v.item() for k, v in cur_loss_dict_rnet.items()}

            eval_loss_dict_cnet = {k: v / len(data) for k, v in eval_loss_dict_cnet.items()}
            eval_loss_dict_rnet = {k: v / len(data) for k, v in eval_loss_dict_rnet.items()}

        return eval_loss_dict_cnet, eval_loss_dict_rnet

    def params_rnet(self,dorig):
        rh_mesh = Meshes(verts=dorig['verts_rhand_f'], faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3)
        rh_mesh_gt = Meshes(verts=dorig['verts_rhand'], faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3)

        o2h_signed, h2o, _ = point2point_signed(dorig['verts_rhand_f'], dorig['verts_object'], rh_mesh)
        o2h_signed_gt, h2o_gt, _ = point2point_signed(dorig['verts_rhand'], dorig['verts_object'], rh_mesh_gt)

        h2o = h2o.abs()
        h2o_gt = h2o_gt.abs()

        return {'h2o_dist': h2o, 'h2o_gt': h2o_gt, 'o2h_gt': o2h_signed_gt}
    
    def get_hand_vertice(self, dorig, drec):
        batch_hand_pose = torch.cat([drec['global_orient'], drec['hand_pose']], dim=1)
        batch_hand_shape = dorig['hand_shape']
        # construct the hand mesh from network output
        out_put:MANOOutput = self.mano_layer(batch_hand_pose, batch_hand_shape)
        verts_rhand = out_put.verts + drec['transl'][:, None, :]
        return verts_rhand


    def loss_rnet(self, dorig, drec):

        # out_put = self.rhm_train(**drec)
        # verts_rhand = out_put.vertices

        verts_rhand = self.get_hand_vertice(dorig, drec)

        rh_mesh = Meshes(verts=verts_rhand, faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3)
        h2o_gt = dorig['h2o_gt']
        o2h_signed, h2o, _ = point2point_signed(verts_rhand, dorig['verts_object'], rh_mesh)
        ######### dist loss
        loss_dist_h = 35 * (1. - self.cfg.kl_coef) * torch.mean(torch.einsum('ij,j->ij', torch.abs(h2o.abs() - h2o_gt.abs()), self.v_weights2))
        ########## verts loss
        loss_mesh_rec_w = 20 * (1. - self.cfg.kl_coef) * torch.mean(torch.einsum('ijk,j->ijk', torch.abs((dorig['verts_rhand'] - verts_rhand)), self.v_weights2))
        ########## edge loss
        loss_edge = 10 * (1. - self.cfg.kl_coef) * self.LossL1(self.edges_for(verts_rhand, self.vpe), self.edges_for(dorig['verts_rhand'], self.vpe))
        ##########

        loss_dict = {
            'loss_edge_r': loss_edge,
            'loss_mesh_rec_r': loss_mesh_rec_w,
            'loss_dist_h_r': loss_dist_h,
        }

        loss_total = 0
        # loss_total = torch.stack(list(loss_dict.values())).sum()
        # loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def loss_cnet(self, dorig, drec):

        device = dorig['verts_rhand'].device
        dtype = dorig['verts_rhand'].dtype

        verts_rhand = self.get_hand_vertice(dorig, drec)

        rh_mesh = Meshes(verts=verts_rhand, faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3)
        rh_mesh_gt = Meshes(verts=dorig['verts_rhand'], faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3)

        o2h_signed, h2o, _ = point2point_signed(verts_rhand, dorig['verts_object'], rh_mesh)
        o2h_signed_gt, h2o_gt, o2h_idx = point2point_signed(dorig['verts_rhand'], dorig['verts_object'], rh_mesh_gt)

        # addaptive weight for penetration and contact verts
        w_dist = (o2h_signed_gt < 0.01) * (o2h_signed_gt > -0.005)
        w_dist_neg = o2h_signed < 0.
        w = self.w_dist.clone()
        w[~w_dist] = .1 # less weight for far away vertices
        w[w_dist_neg] = 1.5 # more weight for penetration
        ######### dist loss
        loss_dist_h = 35 * (1. - self.cfg.kl_coef) * torch.mean(torch.einsum('ij,j->ij', torch.abs(h2o.abs() - h2o_gt.abs()), self.v_weights2))
        loss_dist_o = 30 * (1. - self.cfg.kl_coef) * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed - o2h_signed_gt), w))
        ########## verts loss
        loss_mesh_rec_w = 35 * (1. - self.cfg.kl_coef) * torch.mean(torch.einsum('ijk,j->ijk', torch.abs((dorig['verts_rhand'] - verts_rhand)), self.v_weights))
        ########## edge loss
        loss_edge = 30 * (1. - self.cfg.kl_coef) * self.LossL1(self.edges_for(verts_rhand, self.vpe), self.edges_for(dorig['verts_rhand'], self.vpe))
        ########## KL loss
        loss_kl = self.cfg.kl_coef * self.KLLoss(drec['mu'], drec['logvar'])
        ##########

        loss_dict = {'loss_kl': loss_kl,
                     'loss_edge': loss_edge,
                     'loss_mesh_rec': loss_mesh_rec_w,
                     'loss_dist_h': loss_dist_h,
                     'loss_dist_o': loss_dist_o,
                     }

        loss_total = 0
        # loss_total = torch.stack(list(loss_dict.values())).sum()
        # loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def fit(self, n_epochs=None, message=None):

        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        prev_lr_cnet = np.inf
        prev_lr_rnet = np.inf
        self.fit_cnet = True
        self.fit_rnet = False

        lr_scheduler_cnet = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_cnet, 'min')
        lr_scheduler_rnet = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_rnet, 'min')
        early_stopping_cnet = EarlyStopping(patience=10, trace_func=self.logger)
        early_stopping_rnet = EarlyStopping(patience=10, trace_func=self.logger)

        for epoch_num in range(1, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            eval_loss_dict_cnet , eval_loss_dict_rnet  = self.evaluate()
            train_loss_dict_cnet, train_loss_dict_rnet = self.train()



            if self.fit_cnet:

                lr_scheduler_cnet.step(eval_loss_dict_cnet['loss_total'])
                cur_lr_cnet = self.optimizer_cnet.param_groups[0]['lr']

                if cur_lr_cnet != prev_lr_cnet:
                    self.logger('--- CoarseNet learning rate changed from %.2e to %.2e ---' % (prev_lr_cnet, cur_lr_cnet))
                    prev_lr_cnet = cur_lr_cnet

                with torch.no_grad():
                    eval_msg = Trainer.create_loss_message(eval_loss_dict_cnet, expr_ID=self.cfg.expr_ID,
                                                            epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                            model_name='CoarseNet',
                                                            try_num=self.try_num, mode='evald')
                    if train_loss_dict_cnet['loss_total'] < self.best_loss_cnet:

                        self.cfg.best_cnet = makepath(os.path.join(self.cfg.work_dir, 'snapshots', 'TR%02d_E%03d_cnet.pt' % (self.try_num, self.epochs_completed)), isfile=True)
                        self.save_cnet()
                        self.logger(eval_msg + ' ** ')
                        self.best_loss_cnet = train_loss_dict_cnet['loss_total']

                    else:
                        self.logger(eval_msg)

                    # self.swriter.add_scalars('total_loss_cnet/scalars',
                    #                          {'train_loss_total': train_loss_dict_cnet['loss_total'],
                    #                          'evald_loss_total': eval_loss_dict_cnet['loss_total'], },
                    #                          self.epochs_completed)
                    
                    # self.swriter.add_scalars('kl_loss_cnet/scalars',
                    #                          {'train_loss_kl': train_loss_dict_cnet['loss_kl'],
                    #                          'evald_loss_kl': eval_loss_dict_cnet['loss_kl'], },
                    #                          self.epochs_completed)
                    
                    for k in train_loss_dict_cnet:
                        self.swriter.add_scalars(f'{k}_cnet/scalars',
                                                 {f'train_{k}':train_loss_dict_cnet[k],
                                                  f'evald_{k}':eval_loss_dict_cnet[k]},
                                                 self.epochs_completed)
                    
                    # if 'loss_lap' in train_loss_dict_cnet.keys():
                    #     self.swriter.add_scalars('loss_lap_cnet/scalars',
                    #                             {'train_loss_lap': train_loss_dict_cnet['loss_lap']/30,
                    #                             'evald_loss_lap': eval_loss_dict_cnet['loss_lap']/30, },
                    #                             self.epochs_completed)
                        
                    # if 'loss_anatomy' in train_loss_dict_cnet.keys():
                    #     self.swriter.add_scalars('loss_anatomy_cnet/scalars',
                    #                             {'train_loss_anatomy': train_loss_dict_cnet['loss_anatomy']/30,
                    #                             'evald_loss_anatomy': eval_loss_dict_cnet['loss_anatomy']/30, },
                    #                             self.epochs_completed)

                if early_stopping_cnet(eval_loss_dict_cnet['loss_total'], train_loss_dict_cnet['loss_total'] ):
                    self.fit_cnet = False
                    self.logger('Early stopping CoarseNet training!')

            if self.fit_rnet:

                lr_scheduler_rnet.step(eval_loss_dict_rnet['loss_total'])
                cur_lr_rnet = self.optimizer_rnet.param_groups[0]['lr']

                if cur_lr_rnet != prev_lr_rnet:
                    self.logger('--- RefineNet learning rate changed from %.2e to %.2e ---' % (prev_lr_rnet, cur_lr_rnet))
                    prev_lr_rnet = cur_lr_rnet

                with torch.no_grad():
                    eval_msg = Trainer.create_loss_message(eval_loss_dict_rnet, expr_ID=self.cfg.expr_ID,
                                                           epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                           model_name='RefineNet',
                                                           try_num=self.try_num, mode='evald')
                    if train_loss_dict_rnet['loss_total']<self.best_loss_rnet:

                        self.cfg.best_rnet = makepath(os.path.join(self.cfg.work_dir, 'snapshots', 'TR%02d_E%03d_rnet.pt' % (self.try_num, self.epochs_completed)), isfile=True)
                        self.save_rnet()
                        self.logger(eval_msg + ' ** ')
                        self.best_loss_rnet = train_loss_dict_rnet['loss_total']

                    else:
                        self.logger(eval_msg)

                    for k in train_loss_dict_rnet:
                        self.swriter.add_scalars(f'{k}_rnet/scalars',
                                                 {f'train_{k}':train_loss_dict_rnet[k],
                                                  f'evald_{k}':eval_loss_dict_rnet[k]},
                                                 self.epochs_completed)

                    # self.swriter.add_scalars('total_loss_rnet/scalars',
                    #                          {'train_loss_total': train_loss_dict_rnet['loss_total'],
                    #                           'evald_loss_total': eval_loss_dict_rnet['loss_total'], },
                    #                          self.epochs_completed)
                    
                    # if 'loss_lap' in train_loss_dict_rnet.keys():
                    #     self.swriter.add_scalars('loss_lap_rnet/scalars',
                    #                             {'train_loss_lap': train_loss_dict_rnet['loss_lap']/30.0,
                    #                             'evald_loss_lap': eval_loss_dict_rnet['loss_lap']/30.0, },
                    #                             self.epochs_completed)
                        
                    # if 'loss_anatomy' in train_loss_dict_rnet.keys():
                    #     self.swriter.add_scalars('loss_anatomy_rnet/scalars',
                    #                             {'train_loss_anatomy': train_loss_dict_rnet['loss_anatomy']/30.0,
                    #                             'evald_loss_anatomy': eval_loss_dict_rnet['loss_anatomy']/30.0, },
                    #                             self.epochs_completed)

                if early_stopping_rnet(eval_loss_dict_rnet['loss_total'], train_loss_dict_rnet['loss_total']):
                    self.fit_rnet = False
                    self.logger('Early stopping RefineNet training!')


            self.epochs_completed += 1

            if not self.fit_cnet and not self.fit_rnet:
                self.logger('Stopping the training!')
                break
                

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s!\n' % (endtime - starttime))
        self.logger('Best CoarseNet val total loss achieved: %.2e\n' % (self.best_loss_cnet))
        self.logger('Best CoarseNet model path: %s\n' % self.cfg.best_cnet)

        self.logger(
            'Best RefineNet val total loss achieved: %.2e\n' % (self.best_loss_rnet))
        self.logger('Best RefineNet model path: %s\n' % self.cfg.best_rnet)

    def eval(self):
        self.coarse_net.eval()
        self.refine_net.eval()
        # ds_name = self.cfg.dataset_dir.split('/')[-1]
        ds_name = str(self.cfg.expr_ID)

        total_error_cnet = {}
        total_error_rnet = {}
        for split, ds in [('val', self.ds_val), ('test', self.ds_test), ('train', self.ds_train)]:

            mean_error_cnet = []
            mean_error_rnet = []
            with torch.no_grad():
                for dorig in ds:

                    dorig = {k: dorig[k].to(self.device) for k in dorig.keys()}

                    MESH_SCALER = 1000

                    drec_cnet = self.coarse_net(**dorig)
                    # verts_hand_cnet = self.rhm_train(**drec_cnet).vertices
                    verts_hand_cnet = self.get_hand_vertice(dorig, drec_cnet)

                    mean_error_cnet.append(torch.mean(torch.abs(dorig['verts_rhand'] - verts_hand_cnet) * MESH_SCALER))

                    ########## refine net
                    params_rnet = self.params_rnet(dorig)
                    dorig.update(params_rnet)
                    drec_rnet = self.refine_net(**dorig)
                    # verts_hand_mano = self.rhm_train(**drec_rnet).vertices
                    verts_hand_mano = self.get_hand_vertice(dorig, drec_rnet)

                    mean_error_rnet.append(torch.mean(torch.abs(dorig['verts_rhand'] - verts_hand_mano) * MESH_SCALER))

            total_error_cnet[split] = {'v2v_mae': float(to_cpu(torch.stack(mean_error_cnet).mean()))}
            total_error_rnet[split] = {'v2v_mae': float(to_cpu(torch.stack(mean_error_rnet).mean()))}

        outpath = makepath(os.path.join(self.cfg.work_dir, 'evaluations', 'ds_%s'%ds_name, 
                                        os.path.basename(self.cfg.best_cnet).replace('.pt', '_CoarseNet.json')), 
                            isfile=True)

        with open(outpath, 'w') as f:
            json.dump(total_error_cnet, f)

        with open(outpath.replace('.json', '_RefineNet.json'), 'w') as f:
            json.dump(total_error_rnet, f)

        return total_error_cnet, total_error_rnet

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='CoarseNet', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)
