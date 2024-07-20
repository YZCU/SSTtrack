from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap


class ssttrackActor(BaseActor):

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        template_list, search_list = [], []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)
        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_list.append(search_img_i)
        out_dict = self.net(template=template_list, search=search_list, return_last_attn=False)
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = gt_dict['search_anno'].view(-1, 4)
        gts = gt_bbox.unsqueeze(0)
        gt_gaussian_maps = generate_heatmap(gts, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)

        try:
            giou_ls, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_ls, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        l1_ls = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        if 'score_map' in pred_dict:
            location_ls = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_ls = torch.tensor(0.0, device=l1_ls.device)

        cg_loss = (self.objective['cg'](pred_dict['cg_res'][1], pred_dict['cg_res'][3])) + (self.objective['cg'](
            pred_dict['cg_res'][0], pred_dict['cg_res'][2]))
        loss = self.loss_weight['giou'] * giou_ls + self.loss_weight['l1'] * l1_ls + self.loss_weight[
            'focal'] * location_ls + self.loss_weight['cg'] * cg_loss
        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_ls.item(),
                      "Loss/l1": l1_ls.item(),
                      "Loss/location(cls)": location_ls.item(),
                      "Loss/cg(l1)": cg_loss.item(),
                      "Mean batch IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
