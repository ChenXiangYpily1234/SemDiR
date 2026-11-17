"""Controls use the same single-forward segmentation and GT region definition."""
from copy import deepcopy
import torch
from torch.nn import functional as F
import torch.distributed as dist
from mmseg.registry import MODELS
from research.ad_pcpf_v2.adapter import ADPCPFRegionPreservingSegmentor,ADPCPFPaperSegmentor
from research.ad_pcpf_v2.regions import ordinary_interior_mask

@MODELS.register_module()
class CompetitorVanillaRSP(ADPCPFRegionPreservingSegmentor):
    def __init__(self,region_preservation,**kwargs):
        # Explicitly registered 2x2 control: vanilla with the identical RSP loss.
        ADPCPFPaperSegmentor.__init__(self,**kwargs)
        self.region_preservation=deepcopy(region_preservation)
        self._rsp_iteration=0
        object.__setattr__(self,'_rsp_mask_pool',None)

@MODELS.register_module()
class CompetitorRegionCE(ADPCPFRegionPreservingSegmentor):
    def loss(self,inputs,data_samples):
        options=self.region_preservation
        labels=torch.stack([s.gt_sem_seg.data[0] for s in data_samples]).to(inputs.device,torch.long)
        ordinary=ordinary_interior_mask(labels.detach().cpu(),self.decode_head.num_classes,
            options['boundary_radius'],options['small_area_fraction'],255).to(inputs.device)
        losses,logits=self._loss_and_logits(inputs,data_samples)
        with torch.autocast(device_type=inputs.device.type,enabled=False):
            aligned=F.interpolate(logits.float(),labels.shape[-2:],mode='bilinear',align_corners=False)
            ce=F.cross_entropy(aligned,labels,ignore_index=255,reduction='none')
            mask=ordinary & (labels!=255);safe=labels.clamp(0,aligned.shape[1]-1).reshape(-1)
            sums=ce.new_zeros(aligned.shape[1]).scatter_add(0,safe,(ce*mask).reshape(-1))
            counts=torch.zeros_like(sums,dtype=torch.int64).scatter_add(0,safe,mask.reshape(-1).long())
            report=sums.detach().clone();world=dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            if world>1:dist.all_reduce(counts);dist.all_reduce(report)
            number=(counts>0).sum().clamp_min(1);denom=counts.clamp_min(1).to(sums)
            loss=world*(sums/denom).sum()/number
            weight=options['max_weight']*min(1.,self._rsp_iteration/options['warmup_iters'])
            losses.update(loss_region_ce=weight*loss,region_ce_raw=(report/denom).sum()/number,
                          region_ce_inner_pixels=counts.sum().float())
        return losses
