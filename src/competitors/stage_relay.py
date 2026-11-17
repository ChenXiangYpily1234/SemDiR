"""Declared M21 extension to heterogeneous stage widths/resolutions.

New adaptation: stage-specific tied channel projections to a 192-channel,
stride-16 grid. No fabricated CLS tokens and no shallow-first prior.
Original stage outputs remain the semantic anchors.
"""
import torch
from torch import nn
from torch.nn import functional as F
from research.ad_pcpf_v2.variants import VariantDepthContext
from research.ad_pcpf_v2.relay import TaskGatedAllDepthRelay
from research.pcpf_v6.core import contrast_correction


class StageAlignedRelay(nn.Module):
    novelty=False
    branch_budget=rgb_budget=None
    residual_scale=1.
    use_rgb=use_attenuation=True
    plain_cnn=False
    def __init__(self,channels,latent=192,*,transport=False,transport_hidden=32):
        super().__init__()
        self.in_proj=nn.ModuleList(nn.Conv2d(c,latent,1,bias=False) for c in channels)
        self.context=VariantDepthContext(latent,32,0.,4.,None,'learned','measured')
        self.coarse_dw=nn.Conv2d(latent,latent,3,padding=1,groups=latent,bias=False)
        self.coarse_out=nn.ModuleList(nn.Conv2d(latent,latent,1,bias=False) for _ in range(4))
        self.raw_alpha=nn.Parameter(torch.zeros(4))
        self.stem=nn.Sequential(nn.Conv2d(3,48,3,2,1),nn.GELU(),nn.Conv2d(48,48,3,2,1),nn.GELU())
        self.detail_dw=nn.Conv2d(48,48,3,padding=1,groups=48,bias=False)
        self.semantic_proj=nn.Conv2d(channels[0],48,1,bias=False)
        self.detail_gate=nn.Conv2d(98,1,1)
        self.detail_out=nn.Conv2d(48,channels[0],1,bias=False)
        self.last_diagnostics={}
        nn.init.dirac_(self.coarse_dw.weight,groups=latent)
        nn.init.dirac_(self.detail_dw.weight,groups=48)
        for p in self.coarse_out:nn.init.zeros_(p.weight)
        nn.init.zeros_(self.detail_out.weight);nn.init.zeros_(self.detail_gate.weight);nn.init.zeros_(self.detail_gate.bias)
        self.transport=None
        if transport:
            from research.ad_pcpf_v2.transport import CoarseFineResidualTransport
            self.transport=CoarseFineResidualTransport(
                latent,transport_hidden,rgb_channels=channels[0],output_channels=channels[0])

    def forward(self,image,features,layers):
        grid=features[2].shape[-2:]
        projected=[F.interpolate(self.in_proj[s](x),grid,mode='bilinear',align_corners=False) for s,x in layers]
        refined=[];supports=[];diagnostics=[];coarse_residuals=[]
        for i,base in enumerate(features):
            semantic=F.interpolate(self.in_proj[i](base),grid,mode='bilinear',align_corners=False)
            anchor,detail,support,diag=self.context(projected,semantic,return_diagnostics=True)
            update=self.coarse_out[i](F.gelu(self.coarse_dw(detail.to(semantic.dtype))))
            update=update-update.mean((-2,-1),keepdim=True)
            update=update+contrast_correction(semantic.float().movedim(1,-1).contiguous(),update.float().movedim(1,-1).contiguous(),anchor.float().movedim(1,-1).contiguous(),mode='persistent',offset=(0,0)).movedim(-1,1).to(update.dtype)
            coarse_residuals.append(update)
            update=F.conv2d(update,self.in_proj[i].weight.transpose(0,1))
            update=F.interpolate(update,base.shape[-2:],mode='bilinear',align_corners=False)
            update=update-update.mean((-2,-1),keepdim=True)
            refined.append(base+(2*self.raw_alpha[i].sigmoid()).to(base.dtype)*update)
            supports.append(support);diagnostics.append(diag)
        self.last_diagnostics=dict(beta=diagnostics[0]['beta'],weights=diagnostics[0]['weights'],alpha=2*self.raw_alpha.detach().sigmoid())
        if self.transport is None:
            refined[0]=TaskGatedAllDepthRelay.refine_highres(self,image,refined[0],supports[0])
        else:
            high,rgb=TaskGatedAllDepthRelay.refine_highres(
                self,image,refined[0],supports[0],return_update=True)
            extra=self.transport(coarse_residuals,rgb)
            refined[0]=high+extra
            self.last_diagnostics.update(self.transport.last_diagnostics)
        return tuple(refined)
