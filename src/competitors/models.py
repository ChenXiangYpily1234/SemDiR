"""Independent competing backbones with explicit, checked initialization."""
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from mmseg.registry import MODELS
from mmseg.models.necks import Feature2Pyramid
from research.ad_pcpf_v2.artifacts import sha256
from research.ad_pcpf_v2.adapter import ADPCPFPaperBackbone
from .vit_comer.official import OfficialViTCoMer
from .locatvit.vit import LocAtViT
from .locatvit.swin import LocAtSwin
from research.ad_pcpf_v2.transport import CoarseFineResidualTransport


class CapacityMatchedPlainFusion(CoarseFineResidualTransport):
    """Same learned tensors as CFRT; ordinary spatial gated residual fusion."""
    def configure_conditioning(self, conditioning='both'):
        if conditioning not in ('both', 'no_coarse', 'no_rgb'):
            raise ValueError(f'Unknown plain-fusion conditioning: {conditioning}')
        self.conditioning = conditioning
        if conditioning == 'no_coarse':
            # A shared learned token replaces all sample/spatial coarse content.
            # Copy the existing bias to preserve RNG and common initial tensors;
            # a nonzero token also avoids a dead score/value initialization.
            self.coarse_token = nn.Parameter(self.coarse.bias.detach().clone())
            del self.coarse
        elif conditioning == 'no_rgb':
            del self.rgb

    def forward(self, residuals, rgb, q=None):
        conditioning = getattr(self, 'conditioning', 'both')
        if conditioning == 'no_coarse':
            b, _, h, w = residuals[0].shape
            c = self.coarse_token[None, :, None, None].expand(b, -1, h, w)
        else:
            c = self.coarse(torch.cat(residuals, 1))
        c = self.norm(c.movedim(1, -1)).movedim(-1, 1)
        height, width = rgb.shape[-2:]
        _, pos = self.geometry(height, width, *c.shape[-2:], rgb.device)
        expanded = F.interpolate(c, (height, width), mode='bilinear', align_corners=False)
        if q is None:
            q = torch.zeros_like(expanded) if conditioning == 'no_rgb' else self.rgb(rgb)
        score = self.score(torch.cat((expanded*q, (expanded-q).abs(), q,
            pos[None].expand(rgb.shape[0], -1, -1, -1).to(q.dtype)), 1))
        value = F.interpolate(self.value(c), (height, width), mode='bilinear', align_corners=False)
        extra = (score.float().tanh() * value.float()).to(rgb.dtype)
        self.last_diagnostics = dict(transport_update_rms=extra.detach().float().square().mean().sqrt(),
            transport_value_rms=value.detach().float().square().mean().sqrt(),
            transport_allocation_rms=score.detach().float().tanh().square().mean().sqrt())
        return extra


@MODELS.register_module()
class CompetitorM23Backbone(ADPCPFPaperBackbone):
    def __init__(self, fusion='cfrt', conditioning='both', **kwargs):
        super().__init__(method='adpcpf_r9', **kwargs)
        if fusion == 'plain':
            # Preserve every parameter and RNG state of the CFRT initialization.
            self.relay.transport.__class__ = CapacityMatchedPlainFusion
            self.relay.transport.configure_conditioning(conditioning)
            # Historical Full configs share the sole active Exact forward.
            # Dedicated ablations retain their own generic relay semantics.
            from .semdir_exact import convert_relay, supports_exact_relay
            if supports_exact_relay(self.relay):
                convert_relay(self.relay)
        elif fusion != 'cfrt':
            raise ValueError(f'Unknown M23 fusion: {fusion}')
        elif conditioning != 'both':
            raise ValueError('Conditioning interventions require plain fusion')


def checked_state(path, digest):
    if not path or not digest or sha256(path) != digest:
        raise ValueError('Explicit local checkpoint and matching SHA256 are required')
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(str(path),device='cpu')
    x=torch.load(path,map_location='cpu',weights_only=True)
    return x.get('model',x.get('state_dict',x))


@MODELS.register_module()
class CompetitorViTController(ADPCPFPaperBackbone):
    def __init__(self, **kwargs):
        super().__init__(method='readout_only', **kwargs)


@MODELS.register_module()
class CompetitorCoMer(OfficialViTCoMer):
    """Complete author MRFP/CTI, with MMCV's identical deformable-attention op."""
    method='vit_comer'
    def __init__(self,pretrained_checkpoint,pretrained_sha256,drop_path_rate=.2,**kwargs):
        super().__init__(embed_dim=384,depth=12,num_heads=6,patch_size=16,conv_inplane=64,
            n_points=4,deform_num_heads=6,cffn_ratio=.25,deform_ratio=1.,dim_ratio=1.5,
            interaction_indexes=[[0,2],[3,5],[6,8],[9,11]],window_attn=[False]*12,
            window_size=[None]*12,pretrained=None,drop_path_rate=drop_path_rate,**kwargs)
        self.pretrained_checkpoint=pretrained_checkpoint
        self.pretrained_sha256=pretrained_sha256
        self.loaded=False

    def init_weights(self,pretrained=None):
        # The upstream base invokes this before the child constructor finishes.
        if not hasattr(self,'pretrained_checkpoint') or self.loaded:return
        state=checked_state(self.pretrained_checkpoint,self.pretrained_sha256)
        allowed={'pos_embed'}|{k for k in self.state_dict() if k.startswith(('blocks.','patch_embed.')) and not k.endswith(('gamma1','gamma2'))}
        state={k:v for k,v in state.items() if k in allowed}
        if set(state)!=allowed:raise ValueError('Incomplete CoMer pretrained plain-ViT mapping')
        self.load_state_dict(state,strict=False)
        self.loaded=True;self._is_init=True

    def forward(self,image):
        # The author's 1/8,1/16,1/32 sequence assumes exact scale ratios.
        # Padding occurs on already-normalized input, and is shared by all paths.
        h,w=image.shape[-2:]
        image=F.pad(image,(0,(-w)%32,0,(-h)%32))
        return super().forward(image)

    def check_zero_initialization(self,image):
        pass  # CoMer is not a zero-residual extension of the paper baseline.


@MODELS.register_module()
class CompetitorLocAt(nn.Module):
    method='locatvit'
    def __init__(self,pretrained_checkpoint,pretrained_sha256,size='tiny',architecture='vit',
                 frozen=True,linear_probe=True,drop_path_rate=0.):
        super().__init__()
        self.pretrained_checkpoint=pretrained_checkpoint;self.pretrained_sha256=pretrained_sha256
        self.frozen=frozen;self.linear_probe=linear_probe;self.architecture=architecture
        if architecture=='vit':
            dim={'tiny':192,'small':384,'base':768}[size];heads=dim//64
            self.encoder=LocAtViT(patch_size=16,embed_dim=dim,depth=12,num_heads=heads,
                num_classes=1000,dynamic_img_size=True,dynamic_img_pad=True,
                store_metrics=False,drop_path_rate=drop_path_rate)
            self.indices=(11,) if linear_probe else (2,5,8,11)
            self.pyramid=None if linear_probe else Feature2Pyramid(embed_dim=dim,rescales=[4,2,1,.5],norm_cfg=dict(type='SyncBN'))
        elif architecture=='swin' and size=='tiny':
            self.encoder=LocAtSwin(patch_size=4,window_size=7,embed_dim=96,depths=(2,2,6,2),
                num_heads=(3,6,12,24),strict_img_size=False,drop_path_rate=drop_path_rate)
            self.encoder.patch_embed.dynamic_img_pad=True
            self.indices=(3,) if linear_probe else (0,1,2,3);self.pyramid=None
        else:raise ValueError('No declared LocAt variant for this backbone')
        # Classification head/PRR are required in the checkpoint, but the author's
        # segmentation readout extracts intermediates BEFORE PRR/classification.
        for name in ('head','prr','norm','fc_norm'):
            module=getattr(self.encoder,name,None)
            if isinstance(module,nn.Module):module.requires_grad_(False)
        if frozen:self.encoder.requires_grad_(False)
        self.loaded=False

    def init_weights(self):
        if self.loaded:return
        state=checked_state(self.pretrained_checkpoint,self.pretrained_sha256)
        self.encoder.load_state_dict(state,strict=True)
        if not any('gaug.log_var' in k for k in state):raise ValueError('Vanilla weights are not LocAt pretraining')
        self.loaded=True

    def train(self,mode=True):
        super().train(mode)
        if self.frozen:self.encoder.eval()
        return self

    def forward(self,image):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            out=self.encoder.forward_intermediates(image,indices=self.indices,norm=False,
                output_fmt='NCHW',intermediates_only=True)
        return self.pyramid(tuple(out)) if self.pyramid is not None else tuple(out)

    def check_zero_initialization(self,image):pass


@MODELS.register_module()
class CompetitorHierarchical(nn.Module):
    """MMSeg Swin/MiT baseline or explicit stage-aligned DCPA adaptation."""
    def __init__(self,architecture,pretrained_checkpoint,pretrained_sha256,local_relay=False,
                 transport=False,fusion='cfrt'):
        super().__init__()
        from mmseg.models.backbones import SwinTransformer,MixVisionTransformer
        self.architecture=architecture;self.pretrained_checkpoint=pretrained_checkpoint
        self.pretrained_sha256=pretrained_sha256;self.method='adpcpf_r9' if local_relay else 'vanilla'
        if architecture=='swin_t':
            self.encoder=SwinTransformer(pretrain_img_size=224,embed_dims=96,depths=(2,2,6,2),
                num_heads=(3,6,12,24),window_size=7,out_indices=(0,1,2,3),drop_path_rate=.1)
            channels=(96,192,384,768)
        elif architecture=='mit_b0':
            self.encoder=MixVisionTransformer(embed_dims=32,num_layers=(2,2,2,2),
                num_heads=(1,2,5,8),patch_sizes=(7,3,3,3),sr_ratios=(8,4,2,1),
                out_indices=(0,1,2,3),drop_path_rate=.1)
            channels=(32,64,160,256)
        else:raise ValueError(architecture)
        self.channels=channels;self.loaded=False;self._capture=[];self.handles=[]
        if transport and not local_relay:
            raise ValueError('CFRT transport requires the stage-aligned local relay')
        if fusion not in ('cfrt','plain') or (fusion=='plain' and not transport):
            raise ValueError('Plain fusion requires the stage-aligned transport path')
        if local_relay:
            from .stage_relay import StageAlignedRelay
            with torch.random.fork_rng(devices=[]):
                self.relay=StageAlignedRelay(channels,transport=transport)
            if fusion=='plain':
                self.relay.transport.__class__=CapacityMatchedPlainFusion
                self.relay.transport.configure_conditioning('both')
            if architecture=='swin_t':
                for stage,layer in enumerate(self.encoder.stages):
                    for block in layer.blocks:self.handles.append(block.register_forward_hook(self._hook(stage)))
            else:
                for stage,layer in enumerate(self.encoder.layers):
                    for block in layer[1]:self.handles.append(block.register_forward_hook(self._hook(stage)))

    def _hook(self,stage):
        def hook(module,args,output):
            tokens=output
            shape=args[1]
            h,w=map(int,shape)
            self._capture.append((stage,tokens.transpose(1,2).reshape(tokens.shape[0],-1,h,w)))
        return hook

    def init_weights(self):
        if self.loaded:return
        state=checked_state(self.pretrained_checkpoint,self.pretrained_sha256)
        # Published OpenMMLab converted files use backbone.* prefixes.
        state={k.removeprefix('backbone.'):v for k,v in state.items() if not k.startswith(('head.','decode_head.','auxiliary_head.'))}
        state={k:v for k,v in state.items() if not k.endswith('attn_mask')}
        missing,unexpected=self.encoder.load_state_dict(state,strict=False)
        if self.architecture=='swin_t':
            expected={f'norm{i}.{p}' for i in range(4) for p in ('weight','bias')}
            missing=[k for k in missing if k not in expected]
        # Some classification checkpoints include an unused final classification norm.
        unexpected=[k for k in unexpected if k not in ('norm.weight','norm.bias')]
        if missing or unexpected:raise ValueError(f'Hierarchical pretrained mapping: {missing}, {unexpected}')
        self.loaded=True

    def forward(self,image):
        self._capture=[]
        try:
            out=self.encoder(image)
            return self.relay(image,out,self._capture) if hasattr(self,'relay') else out
        finally:self._capture=[]

    def check_zero_initialization(self,image):
        if not hasattr(self,'relay'):return
        training=self.training;self.eval()
        try:
            with torch.no_grad():
                self._capture=[];base=self.encoder(image);out=self.relay(image,base,self._capture)
                if not all(torch.equal(x,y) for x,y in zip(base,out)):raise RuntimeError('Stage relay initialization differs')
        finally:self._capture=[];self.train(training)


@MODELS.register_module()
class CompetitorPlainProbe(nn.Module):
    method='vanilla_probe'
    def __init__(self,pretrained_checkpoint,pretrained_sha256):
        super().__init__()
        import timm
        self.encoder=timm.create_model('deit_tiny_patch16_224',pretrained=False,dynamic_img_size=True,dynamic_img_pad=True)
        self.encoder.requires_grad_(False)
        self.pretrained_checkpoint=pretrained_checkpoint;self.pretrained_sha256=pretrained_sha256
        self.loaded=False
    def init_weights(self):
        if not self.loaded:
            self.encoder.load_state_dict(checked_state(self.pretrained_checkpoint,self.pretrained_sha256),strict=True);self.loaded=True
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,x):
        with torch.no_grad():return tuple(self.encoder.forward_intermediates(x,indices=(11,),norm=False,output_fmt='NCHW',intermediates_only=True))
    def check_zero_initialization(self,image):pass


from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

@MODELS.register_module()
class CompetitorLinearHead(BaseDecodeHead):
    """Author LocAt frozen readout: pointwise Conv + GN(1) + GELU + cls."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.conv=ConvModule(self.in_channels,self.channels,1,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
    def forward(self,inputs):return self.cls_seg(self.conv(inputs[0]))
