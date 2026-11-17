"""Explicit DCPA extensions for main-table cells absent from author ADE20K tables.

Controller: per-stage channel-wise depth softmax using pooled block tokens.
CoMer: upstream CNN/MRFP/CTI modules, bidirectional interaction around each native
stage through a shared stride-16 projection; native Swin/MiT blocks are retained.
FDAM: masked window AttInv plus both frequency paths. LocAt: GAug fine-tuning
from vanilla ImageNet weights; no author GAug/PRR classification pretraining.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mmseg.registry import MODELS
from mmseg.models.backbones.swin import WindowMSA, SwinBlock
from mmseg.models.backbones.mit import EfficientMultiheadAttention, TransformerEncoderLayer
from research.ad_pcpf_v2.fdam import GroupDynamicScale, StarReLU
from research.ad_pcpf_v2.adapter import ADPCPFPaperBackbone
from .models import CompetitorHierarchical
from .locatvit.locat import GaussianAugment
from .locatvit.vit import GaussianAugmentedAttention
from .vit_comer.modules import CNN, CTIBlock, deform_inputs, deform_inputs_only_one


def nchw(x, shape):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], *shape)


def add_attinv(module, channels, heads):
    module.dy_freq = nn.Linear(channels, heads)
    module.dy_freq_2 = nn.Linear(channels, heads)
    module.dy_freq_starrelu = StarReLU()
    module.lf_gamma = nn.Parameter(torch.full((channels,), 1e-5))
    module.hf_gamma = nn.Parameter(torch.full((channels,), 1e-5))


def invert_attention(module, x, context, values, heads):
    feat = module.dy_freq_starrelu(x)
    low = module.dy_freq_2(feat).tanh().repeat_interleave(x.shape[-1] // heads, -1)
    high2 = F.softplus(module.dy_freq(feat)).square()
    high = (2 * high2 / (high2 + .3678)).repeat_interleave(x.shape[-1] // heads, -1)
    return context + context * low * module.lf_gamma + (values-context) * high * module.hf_gamma


class AugmentedWindowMSA(WindowMSA):
    def forward(self, x, mask=None):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
        scores = (q * self.scale) @ k.transpose(-2,-1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(n,n,-1).permute(2,0,1)
        scores = scores + bias[None]
        if mask is not None:
            nw = mask.shape[0]
            scores = (scores.reshape(b//nw,nw,self.num_heads,n,n) + mask[None,:,None]).reshape(b,self.num_heads,n,n)
        if self.augmentation == 'locat':
            scores = scores.float() + self.gaug(q).float()
        attention = self.attn_drop(scores.softmax(-1)).to(v.dtype)
        context = (attention @ v).transpose(1,2).reshape(b,n,c)
        if self.augmentation == 'fdam':
            context = invert_attention(self,x,context,v.transpose(1,2).reshape(b,n,c),self.num_heads)
        return self.proj_drop(self.proj(context))


class AugmentedSRA(EfficientMultiheadAttention):
    def forward(self, x, hw_shape, identity=None):
        b,n,c = x.shape
        kv = self.norm(self.sr(nchw(x,hw_shape)).flatten(2).transpose(1,2)) if self.sr_ratio>1 else x
        wq,wk,wv = self.attn.in_proj_weight.chunk(3)
        biases = self.attn.in_proj_bias.chunk(3) if self.attn.in_proj_bias is not None else (None,None,None)
        q = F.linear(x,wq,biases[0]).reshape(b,n,self.num_heads,-1).transpose(1,2)
        k = F.linear(kv,wk,biases[1]).reshape(b,-1,self.num_heads,c//self.num_heads).transpose(1,2)
        v = F.linear(kv,wv,biases[2]).reshape(b,-1,self.num_heads,c//self.num_heads).transpose(1,2)
        addition = None
        if self.augmentation == 'locat':
            h,w = hw_shape; sr = self.sr_ratio
            self.gaug.set_grid_size((h,w))
            with torch.autocast(device_type=x.device.type,enabled=False):
                var,alpha = self.gaug.get_var_and_alpha(q.float())
                yy,xx = torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
                ky,kx = torch.meshgrid(torch.arange(h//sr,device=x.device)*sr+(sr-1)/2,
                                      torch.arange(w//sr,device=x.device)*sr+(sr-1)/2,indexing='ij')
                dy = (yy.reshape(-1,1)-ky.reshape(1,-1)).square()
                dx = (xx.reshape(-1,1)-kx.reshape(1,-1)).square()
                addition = alpha * (-.5*(dy/(var[...,0,None]+self.gaug.EPS)+dx/(var[...,1,None]+self.gaug.EPS))).exp()
        out = F.scaled_dot_product_attention(q,k,v,attn_mask=addition,
            dropout_p=self.attn.dropout if self.training else 0.).transpose(1,2).reshape(b,n,c)
        if self.augmentation == 'fdam':
            values = F.linear(x,wv,biases[2])
            out = invert_attention(self,x,out,values,self.num_heads)
        out = self.attn.out_proj(out)
        return (x if identity is None else identity) + self.dropout_layer(self.proj_drop(out))


class FDAMSwinBlock(SwinBlock):
    def forward(self, x, hw_shape):
        if hasattr(self, 'fdam_attn_drop'):
            a = self.attn(self.norm1(x),hw_shape)
            a = a + self.freq_scale_1(nchw(a,hw_shape)).flatten(2).transpose(1,2)
            x = x + self.fdam_attn_drop(a)
            m = self.ffn(self.norm2(x),identity=torch.zeros_like(x))
            m = m + self.freq_scale_2(nchw(m,hw_shape)).flatten(2).transpose(1,2)
            return x + self.fdam_ffn_drop(m)
        # Keep archived configs/checkpoints on their registered legacy path.
        a = self.attn(self.norm1(x),hw_shape)
        x = x + a + self.freq_scale_1(nchw(a,hw_shape)).flatten(2).transpose(1,2)
        m = self.ffn(self.norm2(x),identity=torch.zeros_like(x))
        return x + m + self.freq_scale_2(nchw(m,hw_shape)).flatten(2).transpose(1,2)


class FDAMMiTBlock(TransformerEncoderLayer):
    def forward(self, x, hw_shape):
        if hasattr(self, 'fdam_attn_drop'):
            a = self.attn(self.norm1(x),hw_shape,identity=torch.zeros_like(x))
            a = a + self.freq_scale_1(nchw(a,hw_shape)).flatten(2).transpose(1,2)
            x = x + self.fdam_attn_drop(a)
            m = self.ffn(self.norm2(x),hw_shape,identity=torch.zeros_like(x))
            m = m + self.freq_scale_2(nchw(m,hw_shape)).flatten(2).transpose(1,2)
            return x + self.fdam_ffn_drop(m)
        a = self.attn(self.norm1(x),hw_shape,identity=torch.zeros_like(x))
        x = x + a + self.freq_scale_1(nchw(a,hw_shape)).flatten(2).transpose(1,2)
        m = self.ffn(self.norm2(x),hw_shape,identity=torch.zeros_like(x))
        return x + m + self.freq_scale_2(nchw(m,hw_shape)).flatten(2).transpose(1,2)


@MODELS.register_module()
class AdaptedLocAtDeiT(ADPCPFPaperBackbone):
    """GAug downstream adaptation, explicitly without LocAt classification pretraining."""
    def __init__(self, **kwargs):
        super().__init__(method='vanilla',**kwargs)
        self.init_weights()
        for block in self.encoder.blocks:
            attn = block.attn
            attn.__class__ = GaussianAugmentedAttention
            attn.gaug = GaussianAugment(attn.qkv.in_features,attn.num_heads)
            attn.gaug.command((14,14),self.encoder.num_prefix_tokens,store_metrics=False)

    def _features(self,image):
        grid = tuple(map(int,self.encoder.patch_embed.dynamic_feat_size(image.shape[-2:])))
        for block in self.encoder.blocks:
            block.attn.gaug.set_grid_size(grid)
        return super()._features(image)


@MODELS.register_module()
class AdaptedHierarchicalCompetitor(CompetitorHierarchical):
    def __init__(self, adaptation, fdam_order='legacy_post_drop', **kwargs):
        if fdam_order not in ('legacy_post_drop', 'pre_drop'):
            raise ValueError('Unknown FDAM frequency/DropPath order')
        if adaptation != 'fdam' and fdam_order != 'legacy_post_drop':
            raise ValueError('Frequency/DropPath order applies only to FDAM')
        super().__init__(local_relay=False,**kwargs)
        # Load the vanilla backbone strictly before adding new operators. All
        # parameters exist before MMEngine builds DDP and the optimizer.
        self.init_weights()
        self.adaptation = adaptation
        self.fdam_order = fdam_order
        self._stage_blocks = [list(s.blocks) for s in self.encoder.stages] if self.architecture=='swin_t' else [list(s[1]) for s in self.encoder.layers]
        self._observed = [[] for _ in range(4)]
        if adaptation == 'controller':
            for i,blocks in enumerate(self._stage_blocks):
                for block in blocks:
                    self.handles.append(block.register_forward_hook(self._capture_controller(i)))
        elif adaptation in ('fdam','locat'):
            h = w = 512
            mit_gaug_grids = []
            if adaptation == 'locat' and self.architecture == 'mit_b0':
                # ADE20K training crops are 512x512. Keep this reference grid
                # fixed and update only current_grid_size for dynamic inputs.
                for layer in self.encoder.layers:
                    projection = layer[0].projection
                    kernel = projection.kernel_size
                    stride = projection.stride
                    padding = projection.padding
                    dilation = projection.dilation
                    h = (h + 2*padding[0] - dilation[0]*(kernel[0]-1) - 1)//stride[0] + 1
                    w = (w + 2*padding[1] - dilation[1]*(kernel[1]-1) - 1)//stride[1] + 1
                    mit_gaug_grids.append((h,w))
            for stage,blocks in enumerate(self._stage_blocks):
                for block in blocks:
                    if self.architecture=='swin_t':
                        attn = block.attn.w_msa
                        attn.__class__ = AugmentedWindowMSA
                        channels,heads = attn.embed_dims,attn.num_heads
                    else:
                        attn = block.attn
                        attn.__class__ = AugmentedSRA
                        channels,heads = attn.embed_dims,attn.num_heads
                    attn.augmentation = adaptation
                    if adaptation == 'fdam':
                        add_attinv(attn,channels,heads)
                        block.__class__ = FDAMSwinBlock if self.architecture=='swin_t' else FDAMMiTBlock
                        block.freq_scale_1 = GroupDynamicScale(channels)
                        block.freq_scale_2 = GroupDynamicScale(channels)
                        if fdam_order == 'pre_drop':
                            # Move the original stochastic-depth operators to
                            # the combined branch: D(a + F(a)), not D(a)+F(D(a)).
                            if self.architecture == 'swin_t':
                                block.fdam_attn_drop = block.attn.drop
                                block.attn.drop = nn.Identity()
                            else:
                                block.fdam_attn_drop = block.attn.dropout_layer
                                block.attn.dropout_layer = nn.Identity()
                            block.fdam_ffn_drop = block.ffn.dropout_layer
                            block.ffn.dropout_layer = nn.Identity()
                    else:
                        attn.gaug = GaussianAugment(channels,heads)
                        if self.architecture=='swin_t':
                            attn.gaug.command(attn.window_size,0,store_metrics=False)
                        elif self.architecture=='mit_b0':
                            attn.gaug.command(mit_gaug_grids[stage],0,store_metrics=False)
        elif adaptation == 'comer':
            self.cnn = CNN(inplanes=64,embed_dim=192)
            self.to_shared = nn.ModuleList(nn.Conv2d(c,192,1) for c in self.channels)
            self.from_shared = nn.ModuleList(nn.Conv2d(192,c,1) for c in self.channels)
            self.interactions = nn.ModuleList(CTIBlock(dim=192,num_heads=6,init_values=0.,
                dim_ratio=1.5,cnn_feature_interaction=True,extra_CTI=(i==3)) for i in range(4))
            for i,blocks in enumerate(self._stage_blocks):
                self.handles.append(blocks[0].register_forward_pre_hook(self._inject(i)))
                self.handles.append(blocks[-1].register_forward_hook(self._extract(i)))
        else:
            raise ValueError(f'Unknown adaptation {adaptation}')

    def _capture_controller(self,stage):
        def hook(module,args,output):
            self._observed[stage].append(output)
        return hook

    def _shared_tokens(self,stage,x,shape):
        y = F.interpolate(self.to_shared[stage](nchw(x,shape)),self._grid,mode='bilinear',align_corners=False)
        return y.flatten(2).transpose(1,2)

    def _inject(self,stage):
        def hook(module,args):
            x,shape = args[:2]; h,w = self._grid; layer = self.interactions[stage]
            shared = self._shared_tokens(stage,x,shape)
            c = layer.mrfp(self._cnn_tokens,h,w)
            c = torch.cat((c[:,:h*w*4],c[:,h*w*4:h*w*5]+shared,c[:,h*w*5:]),1)
            refs = deform_inputs_only_one(shared,h*16,w*16)
            refined = layer.cti_tov(
                query=shared,reference_points=refs[0],feat=c,spatial_shapes=refs[1],level_start_index=refs[2],H=h,W=w)
            self._cnn_tokens = c
            delta = F.interpolate(self.from_shared[stage](nchw(refined-shared,(h,w))),shape,mode='bilinear',align_corners=False)
            return (x+delta.flatten(2).transpose(1,2),*args[1:])
        return hook

    def _extract(self,stage):
        def hook(module,args,output):
            h,w = self._grid; layer = self.interactions[stage]
            shared = self._shared_tokens(stage,output,args[1]); refs = self._deform2
            self._cnn_tokens = layer.cti_toc(query=self._cnn_tokens, reference_points=refs[0], feat=shared, spatial_shapes=refs[1], level_start_index=refs[2], H=h, W=w)
            if layer.extra_CTIs is not None:
                for cti in layer.extra_CTIs:
                    self._cnn_tokens = cti(query=self._cnn_tokens, reference_points=refs[0], feat=shared, spatial_shapes=refs[1], level_start_index=refs[2], H=h, W=w)
        return hook

    def forward(self,image):
        self._observed = [[] for _ in range(4)]
        if self.adaptation == 'comer':
            image = F.pad(image,(0,(-image.shape[-1])%32,0,(-image.shape[-2])%32))
            self._grid = (image.shape[-2]//16,image.shape[-1]//16)
            c1,c2,c3,c4 = self.cnn(image)
            self._cnn_tokens = torch.cat((c2,c3,c4),1)
            _,self._deform2 = deform_inputs(image)
        try:
            out = self.encoder(image)
            if self.adaptation == 'controller':
                result = []
                for i,blocks in enumerate(self._observed):
                    stack = torch.stack(blocks,0)
                    weights = stack.mean(2).softmax(0)
                    fused = (stack*weights[:,:,None]).sum(0)
                    norm = getattr(self.encoder,f'norm{i}') if self.architecture=='swin_t' else self.encoder.layers[i][2]
                    result.append(nchw(norm(fused),out[i].shape[-2:]))
                return tuple(result)
            if self.adaptation == 'comer':
                h,w = self._grid; n = h*w
                c = self._cnn_tokens
                maps = [c1,nchw(c[:,:n*4],(h*2,w*2)),nchw(c[:,n*4:n*5],(h,w)),nchw(c[:,n*5:],(h//2,w//2))]
                return tuple(x+F.interpolate(self.from_shared[i](maps[i]),x.shape[-2:],mode='bilinear',align_corners=False) for i,x in enumerate(out))
            return out
        finally:
            self._observed = [[] for _ in range(4)]
            if self.adaptation=='comer':
                self._cnn_tokens = None

    def check_zero_initialization(self,image):
        pass  # Adapted competitor architectures are not zero-residual M23 arms.
