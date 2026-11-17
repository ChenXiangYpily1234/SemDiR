"""New IDs for missing workbook cells; all measured M IDs remain immutable."""
from dataclasses import dataclass
from pathlib import Path
import os,json
from research.ad_pcpf_v2.experiments import Experiment
ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT.parent/'models'
BUDGET=int(os.environ.get('DCPA_COMPETITOR_BUDGET','160000'))
PROTOCOL=f'DCPA-COMPETITORS-R1-{BUDGET}'
PRESERVED=('M01','M02','M03','M04','M08','M16','M17','M21','M22','M23')
RETIRED=('M05','M06','M07','M09','M10','M11','M12','M13','M14','M15','M18','M19','M20')

@dataclass(frozen=True)
class CompetitorExperiment(Experiment):
    recipe: str='matched'
    comparison: str='M24'
    seed_required: int=1
    @property
    def protocol(self):return PROTOCOL
    @property
    def steps(self):return 80000 if self.dataset=='context59' else 20000 if self.recipe=='native_probe' else BUDGET

EXPERIMENTS={
 'M24':CompetitorExperiment('M24','vanilla',backbone='deit_s',comparison=''),
 'M25':CompetitorExperiment('M25','vitcontroller',backbone='deit_s'),
 'M26':CompetitorExperiment('M26','vit_comer',backbone='deit_s'),
 'M27':CompetitorExperiment('M27','vanilla',backbone='swin_t',comparison=''),
 'M28':CompetitorExperiment('M28','m21_adapted',backbone='swin_t',comparison='M27'),
 'M29':CompetitorExperiment('M29','vanilla',backbone='mit_b0',comparison=''),
 'M30':CompetitorExperiment('M30','m21_adapted',backbone='mit_b0',comparison='M29'),
 'M31':CompetitorExperiment('M31','fdam_mit',backbone='mit_b0',comparison='M29'),
 'M32':CompetitorExperiment('M32','locatvit',backbone='locat_tiny',recipe='native_probe',comparison='M33'),
 'M33':CompetitorExperiment('M33','vanilla_probe',backbone='vit_tiny_native',recipe='native_probe',comparison=''),
 'M34':CompetitorExperiment('M34','locatvit',backbone='locat_swin',recipe='native_probe',comparison=''),
 'M35':CompetitorExperiment('M35','locatvit',backbone='locat_small',comparison='M24'),
 'M36':CompetitorExperiment('M36','vanilla_rsp',backbone='deit_s',comparison='M03'),
 'M37':CompetitorExperiment('M37','r9_region_ce',backbone='deit_s',comparison='M21'),
 'M38':CompetitorExperiment('M38','m21',backbone='deit_s',comparison='M39',seed_required=0),
 'M39':CompetitorExperiment('M39','vanilla',backbone='deit_s',comparison='',seed_required=0),
 'M23-Swin-T':CompetitorExperiment('M23-Swin-T','m23_stage_aligned',backbone='swin_t',comparison='M28'),
 'M23-MiT-B0':CompetitorExperiment('M23-MiT-B0','m23_stage_aligned',backbone='mit_b0',comparison='M30'),
 'M23-NoRSP':CompetitorExperiment('M23-NoRSP','m23_no_rsp',backbone='deit_s',comparison='M23'),
 'M23-PlainFusion':CompetitorExperiment('M23-PlainFusion','m23_plain',backbone='deit_s',comparison='M23'),
 'M23-PlainFusion-Swin-T':CompetitorExperiment('M23-PlainFusion-Swin-T','m23_plain_stage_aligned',backbone='swin_t',comparison='M27'),
 'M23-PlainFusion-Seed0':CompetitorExperiment('M23-PlainFusion-Seed0','m23_plain',backbone='deit_s',comparison='M38',seed_required=0),
 'Context59-Vanilla':CompetitorExperiment('Context59-Vanilla','vanilla',backbone='deit_s',dataset='context59',comparison=''),
 'Context59-FDAM':CompetitorExperiment('Context59-FDAM','fdam',backbone='deit_s',dataset='context59',comparison='M08'),
 'SemDiR-Clean':CompetitorExperiment('SemDiR-Clean','semdir_clean',backbone='deit_s',comparison='M23-PlainFusion'),
 'SemDiR-Clean-Shared':CompetitorExperiment('SemDiR-Clean-Shared','semdir_shared',backbone='deit_s',comparison='SemDiR-Clean'),
 'SemDiR-Lite':CompetitorExperiment('SemDiR-Lite','semdir_lite',backbone='deit_s',comparison='SemDiR-Clean-Shared'),
 'M23-Seed0':CompetitorExperiment('M23-Seed0','m23',backbone='deit_s',comparison='M23',seed_required=0),
 'M23-Context59':CompetitorExperiment('M23-Context59','m23',backbone='deit_s',dataset='context59',comparison='M08'),
}
# New numbered rows are local adaptations; prior measured IDs are unchanged.
EXPERIMENTS.update({
 'M40':CompetitorExperiment('M40','adapted_locat',backbone='deit_s'),
 'M41':CompetitorExperiment('M41','adapted_controller',backbone='swin_t',comparison='M27'),
 'M42':CompetitorExperiment('M42','adapted_comer',backbone='swin_t',comparison='M27'),
 'M43':CompetitorExperiment('M43','adapted_fdam',backbone='swin_t',comparison='M27'),
 'M44':CompetitorExperiment('M44','adapted_locat',backbone='swin_t',comparison='M27'),
 'M45':CompetitorExperiment('M45','adapted_controller',backbone='mit_b0',comparison='M29'),
 'M46':CompetitorExperiment('M46','adapted_comer',backbone='mit_b0',comparison='M29'),
 'M47':CompetitorExperiment('M47','adapted_fdam',backbone='mit_b0',comparison='M29'),
 'M48':CompetitorExperiment('M48','adapted_locat',backbone='mit_b0',comparison='M29'),
 'M48-R2':CompetitorExperiment('M48-R2','adapted_locat',backbone='mit_b0',comparison='M29'),
})
# Author-reported scores stay in the workbook's separate literature sheet.
# M44/M47 local matched training was explicitly authorized on 2026-11-11.
LITERATURE_ONLY={}
MAIN_MISSING_TRAIN=('M40','M41','M42','M43','M44','M45','M46','M47','M48')
PLAIN_ABLATIONS=('M23-PlainFusion-NoRSP','M23-PlainFusion-NoCoarse','M23-PlainFusion-NoRGB')
EXPERIMENTS.update({
 'M23-PlainFusion-NoRSP':CompetitorExperiment('M23-PlainFusion-NoRSP','m23_plain_no_rsp',backbone='deit_s',comparison='M23-PlainFusion'),
 'M23-PlainFusion-NoCoarse':CompetitorExperiment('M23-PlainFusion-NoCoarse','m23_plain_no_coarse',backbone='deit_s',comparison='M23-PlainFusion'),
 'M23-PlainFusion-NoRGB':CompetitorExperiment('M23-PlainFusion-NoRGB','m23_plain_no_rgb',backbone='deit_s',comparison='M23-PlainFusion'),
})
# Historical identities stay readable for sealed checkpoints and evaluation.
# Their training entrypoints are retired, rather than silently repurposed.
READ_ONLY_TRAINING_IDS=frozenset(('M23','M23-NoRSP','M23-Seed0','M23-Context59',
                                'M23-Swin-T','M23-MiT-B0'))

NOT_APPLICABLE={
 ('swin_t','vitcontroller'):'No CLS token; a new layer controller would be a different method.',
 ('mit_b0','vitcontroller'):'No CLS and heterogeneous stage widths; no original implementation.',
 ('swin_t','vit_comer'):'Plain-ViT CTI/feature geometry is not defined for Swin in the paper.',
 ('mit_b0','vit_comer'):'No author MiT MRFP/CTI architecture.',
 ('swin_t','fdam'):'Author Swin AttInv/FreqScale adaptation is not provided.',
 ('mit_b0','locatvit'):'No author MiT GAug/PRR implementation/checkpoint.',
}
BLOCKERS={
 'M31':'Author repository exposes only deit_fdam.py; no released MiT spatial-reduction AttInv/FreqScale source. Do not substitute plain FFT or remove SRA.',
 'M33':'Matched vanilla Tiny ImageNet pretraining control checkpoint is not released. Train/load an independently audited checkpoint under the same classification recipe.',
 'M34':'Author LocAtSwin code available, but no public corresponding pretrained checkpoint found.',
 'M35':'No public LocAtViT-Small/DeiT-S-width GAug+PRR pretraining checkpoint. Vanilla DeiT weights cannot substitute.',
}

def default_weight(spec):
    lock=json.loads((ROOT/'competitors/weights.json').read_text())
    if spec.backbone in lock:
        row=lock[spec.backbone]
        return MODELS/row['file'],row['sha256']
    raise ValueError(BLOCKERS.get(spec.id,'Explicit initialization checkpoint required'))
