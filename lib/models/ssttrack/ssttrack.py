import os
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.ssttrack.hivit import hivit_small, hivit_base
from lib.models.layers.transformer_dec import build_transformer_dec
from lib.models.layers.position_encoding import build_position_encoding


class Fovea(nn.Module):
    def __init__(self, smooth=True):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.smooth = smooth
        if smooth:
            self.stms = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h * w)
        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        return output.contiguous().view(b, c, h, w)


class ssttrack(nn.Module):
    def __init__(self, transformer, box_head, transformer_dec, position_encoding, aux_loss=False,
                 head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            cs: channel selection
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        self.transformer_dec = transformer_dec
        self.position_encoding = position_encoding
        self.query_embed = nn.Embedding(num_embeddings=1, embedding_dim=512)
        self.stmf = Fovea()

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                return_last_attn=False,
                training=True,
                tgt_pre=None):
        b0, num_search = template[0].shape[0], len(search)
        if training:
            template = template[0].repeat(num_search, 1, 1, 1)
            search = torch.cat(search, dim=0)
        x, aux_dict, cg_res, orderY = self.backbone(z=template, x=search, return_last_attn=return_last_attn, )
        input_dec = x
        xdecs = []
        batches = [[] for _ in range(b0)]
        for m, iinput in enumerate(input_dec):
            batches[m % b0].append(iinput.unsqueeze(0))
        query_embed = self.query_embed.weight
        assert len(query_embed.size()) in [2, 3]
        if len(query_embed.size()) == 2:
            query_embeding = query_embed.unsqueeze(1)
        for k, batch in enumerate(batches):
            if len(batch) == 0:
                continue
            tgt_all = [torch.zeros_like(query_embeding) for _ in range(num_search)]
            for j, input in enumerate(batch):
                pos_embed = self.position_encoding(1)
                tgt_q = tgt_all[j]
                tgt_kv = torch.cat(tgt_all[:j + 1], dim=0)
                if not training and len(tgt_pre) != 0:
                    tgt_kv = torch.cat(tgt_pre, dim=0)
                tgt = [tgt_q, tgt_kv]
                tgt_out = self.transformer_dec(input.transpose(0, 1), tgt, self.feat_len_s, pos_embed, query_embeding)
                xdecs.append(tgt_out[0])
                tgt_all[j] = tgt_out[0]
            if not training:
                if len(tgt_pre) < 3:
                    tgt_pre.append(tgt_out[0])
                else:
                    tgt_pre.pop(0)
                    tgt_pre.append(tgt_out[0])
        batch0 = []
        if not training:
            batch0.append(xdecs[0])
        else:
            batch0 = [xdecs[i + j * num_search] for j in range(b0) for i in range(num_search)]
        flast = x
        if isinstance(x, list):
            flast = x[-1]
        xdec = torch.cat(batch0, dim=1)
        out = self.forward_head(flast, xdec, None)
        out.update(aux_dict)
        out['tgt'] = tgt_pre
        out['cg_res'] = cg_res
        out['bs_order'] = orderY
        return out

    def forward_head(self, cat_feature, out_dec=None, gt_score_map=None):
        eopt = cat_feature[:, -self.feat_len_s:]
        att = torch.matmul(eopt, out_dec.transpose(0, 1).transpose(1, 2))
        opt = (eopt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        optfa = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.stmf(optfa) + optfa
        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out


def build_ssttrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('ssttrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    if cfg.MODEL.BACKBONE.TYPE == 'hivit_small':
        backbone = hivit_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    transformer_dec = build_transformer_dec(cfg, hidden_dim)
    position_encoding = build_position_encoding(cfg, sz=1)
    box_head = build_box_head(cfg, hidden_dim)
    model = ssttrack(
        backbone,
        box_head,
        transformer_dec,
        position_encoding,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE)

    if 'ssttrack' in cfg.MODEL.PRETRAIN_FILE and training:
        wgt = 'ssttrack_ep150_full_256.pth.tar'
        ckpt = os.path.join(os.path.abspath(__file__), '../../../../pretrained_models/' + wgt)
        checkpoint = torch.load(ckpt, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from:')
        print(cfg.MODEL.PRETRAIN_FILE, '\n/pretrained_models/' + wgt)
        print('Missing_keys:')
        for val in missing_keys:
            print(val)
        print('Unexpected_keys:')
        for val in unexpected_keys:
            print(val)

    return model
