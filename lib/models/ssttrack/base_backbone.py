from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lib.models.ssttrack.utils import combine_tokens, recover_tokens, split_channel
import torchvision.transforms.functional as tvisf


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384
        self.cat_mode = 'direct'
        self.pos_embed_z = None
        self.pos_embed_x = None
        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None
        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]
        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):
        s_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        t_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        n_p_s = cfg.MODEL.BACKBONE.STRIDE
        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        p_p_e = self.absolute_pos_embed
        p_p_e = p_p_e.transpose(1, 2)
        B, E, Q = p_p_e.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        p_p_e = p_p_e.view(B, E, P_H, P_W)
        H, W = s_size
        new_P_H, new_P_W = H // n_p_s, W // n_p_s
        sea_p_p_e = nn.functional.interpolate(p_p_e, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        sea_p_p_e = sea_p_p_e.flatten(2).transpose(1, 2)
        H, W = t_size
        new_P_H, new_P_W = H // n_p_s, W // n_p_s
        tem_p_p_e = nn.functional.interpolate(p_p_e, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        tem_p_p_e = tem_p_p_e.flatten(2).transpose(1, 2)
        self.pos_embed_z = nn.Parameter(tem_p_p_e)
        self.pos_embed_x = nn.Parameter(sea_p_p_e)
        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x, mask=None):
        avg_p = nn.AvgPool2d(kernel_size=3, stride=3)
        max_p = nn.MaxPool2d(kernel_size=3, stride=3)
        x18 = self.cs_18(x)
        x16 = self.cs_16(x)
        x14 = self.cs_14(x)
        x18 = (avg_p(x18) + max_p(x18)).flatten(2).transpose(1, 2)
        x16 = (avg_p(x16) + max_p(x16)).flatten(2).transpose(1, 2)
        x14 = (avg_p(x14) + max_p(x14)).flatten(2).transpose(1, 2)
        ms_m = self.cs_MP(torch.cat([x18, x16, x14], dim=-2))
        ms_ = ms_m.transpose(1, 2)
        cm = torch.matmul(ms_, ms_m)
        for i in range(16):
            cm[:, i, i] = 0.0
        cm = F.normalize(cm, p=2, dim=-1)
        for i in range(16):
            cm[:, i, i] = 0.0
        w = torch.norm(cm, p=1, dim=-1)
        w_reshape = w.contiguous().view(w.shape[0], w.shape[1], 1, 1)
        x_cg = x * w_reshape.expand_as(x)
        x_cg = x_cg.view(x_cg.shape[0], x_cg.shape[1], -1)
        z_cg = z * w_reshape.expand_as(z)
        z_cg = z_cg.view(z_cg.shape[0], z_cg.shape[1], -1)
        cg_res = [z.view(z.shape[0], z.shape[1], -1), x.view(x.shape[0], x.shape[1], -1), z_cg, x_cg]
        orderY = torch.sort(w, dim=-1, descending=True, out=None)
        x_fc_l, _ = split_channel(x, orderY)
        z_fc_l, _ = split_channel(z, orderY)
        z0, z1, z2, z3, z4 = z_fc_l[0], z_fc_l[1], z_fc_l[2], z_fc_l[3], z_fc_l[4]
        x0, x1, x2, x3, x4 = x_fc_l[0], x_fc_l[1], x_fc_l[2], x_fc_l[3], x_fc_l[4]
        im_m = torch.tensor([0.485, 0.456, 0.406]).cuda()
        im_s = torch.tensor([0.229, 0.224, 0.225]).cuda()
        z0 = tvisf.normalize(z0, im_m, im_s)
        z1 = tvisf.normalize(z1, im_m, im_s)
        z2 = tvisf.normalize(z2, im_m, im_s)
        z3 = tvisf.normalize(z3, im_m, im_s)
        z4 = tvisf.normalize(z4, im_m, im_s)
        x0 = tvisf.normalize(x0, im_m, im_s)
        x1 = tvisf.normalize(x1, im_m, im_s)
        x2 = tvisf.normalize(x2, im_m, im_s)
        x3 = tvisf.normalize(x3, im_m, im_s)
        x4 = tvisf.normalize(x4, im_m, im_s)
        z0 = self.patch_embed(z0)
        x0 = self.patch_embed(x0)
        z1 = self.patch_embed(z1)
        x1 = self.patch_embed(x1)
        z2 = self.patch_embed(z2)
        x2 = self.patch_embed(x2)
        z3 = self.patch_embed(z3)
        x3 = self.patch_embed(x3)
        z4 = self.patch_embed(z4)
        x4 = self.patch_embed(x4)
        for blk in self.blocks[:-self.num_main_blocks]:
            z0, z1, z2, z3, z4 = blk(z0, z1, z2, z3, z4)
            x0, x1, x2, x3, x4 = blk(x0, x1, x2, x3, x4)
        z0 = z0[..., 0, 0, :]
        x0 = x0[..., 0, 0, :]
        z1 = z1[..., 0, 0, :]
        x1 = x1[..., 0, 0, :]
        z2 = z2[..., 0, 0, :]
        x2 = x2[..., 0, 0, :]
        z3 = z3[..., 0, 0, :]
        x3 = x3[..., 0, 0, :]
        z4 = z4[..., 0, 0, :]
        x4 = x4[..., 0, 0, :]
        z0 += self.pos_embed_z
        x0 += self.pos_embed_x
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        z1 += self.pos_embed_z
        x1 += self.pos_embed_x
        z2 += self.pos_embed_z
        x2 += self.pos_embed_x
        z3 += self.pos_embed_z
        x3 += self.pos_embed_x
        z4 += self.pos_embed_z
        x4 += self.pos_embed_x
        x0 = combine_tokens(z0, x0, mode=self.cat_mode)
        x1 = combine_tokens(z1, x1, mode=self.cat_mode)
        x2 = combine_tokens(z2, x2, mode=self.cat_mode)
        x3 = combine_tokens(z3, x3, mode=self.cat_mode)
        x4 = combine_tokens(z4, x4, mode=self.cat_mode)
        x0 = self.pos_drop(x0)
        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)
        x3 = self.pos_drop(x3)
        x4 = self.pos_drop(x4)
        for blk in self.blocks[-self.num_main_blocks:]:
            x0, x1, x2, x3, x4 = blk(x0, x1, x2, x3, x4)
        x = x0 + x1 + x2 + x3 + x4
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)
        aux_dict = {"attn": None}
        x = self.norm_(x)
        return x, aux_dict, cg_res, orderY

    def forward(self, z, x, **kwargs):
        x, aux_dict, cg_res, orderY = self.forward_features(z, x, )
        return x, aux_dict, cg_res, orderY
