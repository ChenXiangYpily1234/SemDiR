"""Canonical R7 registry. M01-M02 remain sealed R3 historical records.

Paper labels such as P-B0 and CTX-B are aliases of one numbered run; they do
not create duplicate trainings. The new R7 study starts at M03 in a new output
directory and uses the workbook's no-SI matched protocol.
"""
from __future__ import annotations
from dataclasses import dataclass

PROTOCOL = 'DCPA-ADPCPF-PAPER-V2-R7-RUNTIME'
FDAM_BASELINE_PROTOCOL = 'FDAM-DEIT-S-UPERNET-ADE20K-160K-REPLICA-V1'  # historical M03 protocol
R9_PROTOCOL = 'DCPA-ADPCPF-PAPER-V3-R9-PYRAMID'
R9_IDS = ('M15', 'M16', 'M17', 'M18')
R10_PROTOCOL = 'DCPA-ADPCPF-PAPER-V4-R10-NBLF'
R10_IDS = ('M19',)
RSP_PROTOCOL = 'DCPA-ADPCPF-PAPER-V6-R9-RSP'
RSP_IDS = ('M21',)
CFRT_PROTOCOL = 'DCPA-M23-COFIRT-RSP-V1'
SEED, WORLD, PER_GPU = 1, 8, 2
PRETRAIN_SHA256 = 'cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f'

HISTORICAL = {
    'M01': dict(method='vanilla', si=.05, protocol='R3', status='preserved_frozen'),
    'M02': dict(method='adpcpf', si=.05, protocol='R3', status='preserved_frozen'),
}


@dataclass(frozen=True)
class Experiment:
    id: str
    method: str
    si: float = 0.0
    dataset: str = 'ade20k'
    backbone: str = 'small'
    reference: str | None = None
    intervention: str = 'full'
    optional: bool = False

    @property
    def protocol(self):
        if self.id == 'M23':
            return CFRT_PROTOCOL
        if self.id in RSP_IDS:
            return RSP_PROTOCOL
        if self.id in R10_IDS:
            return R10_PROTOCOL
        return R9_PROTOCOL if self.id in R9_IDS else PROTOCOL

    @property
    def steps(self):
        return 80000 if self.dataset == 'context59' else 160000

    @property
    def crop(self):
        return 480 if self.dataset == 'context59' else 512

    @property
    def classes(self):
        return 59 if self.dataset == 'context59' else 150


# M01-M02 are intentionally absent here: those R3 checkpoints and receipts
# remain immutable in their historical study directory and are never retrained.
EXPERIMENTS = {
    # ADE20K / DeiT-S main matched rows: P-B0, P-F0, P-O0.
    'M03': Experiment('M03', 'vanilla'),
    'M04': Experiment('M04', 'fdam', reference='M03'),
    'M05': Experiment('M05', 'adpcpf', reference='M03'),
    # ADE20K / DeiT-B matched pair. ViT-S table rows reuse M03/M05 when the
    # checkpoint and config identity are the same; they are not duplicated.
    'M06': Experiment('M06', 'vanilla', backbone='base'),
    'M07': Experiment('M07', 'adpcpf', backbone='base', reference='M06'),
    # Context-59 matched pair.
    'M08': Experiment('M08', 'vanilla', dataset='context59'),
    'M09': Experiment('M09', 'adpcpf', dataset='context59', reference='M08'),
    # No-SI readout control uses M03; local ablations use the shared full run M05.
    'M10': Experiment('M10', 'readout_only', reference='M03', intervention='readout_only'),
    'M11': Experiment('M11', 'adpcpf', reference='M05', intervention='no_projection'),
    'M12': Experiment('M12', 'adpcpf', reference='M05', intervention='uniform'),
    'M13': Experiment('M13', 'adpcpf', reference='M05', intervention='no_rgb'),
    'M14': Experiment('M14', 'adpcpf', reference='M05', intervention='no_gate', optional=True),
}

EXPERIMENTS.update({
    'M15': Experiment('M15', 'vanilla'),
    'M16': Experiment('M16', 'adpcpf_r9', reference='M15'),
    'M17': Experiment('M17', 'vanilla', backbone='base'),
    'M18': Experiment('M18', 'adpcpf_r9', backbone='base', reference='M17'),
    # External comparison baseline is completed M03. Do not mix historical
    # protocols in this new study or automatically retrain a baseline.
    'M19': Experiment('M19', 'adpcpf_r9'),
    'M21': Experiment('M21', 'adpcpf_r9'),
    'M23': Experiment('M23', 'adpcpf_r9'),
})

ALIASES = {
    'vanilla_nosi':'M03', 'readout_only':'M10', 'ours_nosi':'M05', 'fdam_nosi':'M04',
    'DB-B':'M06', 'DB-O':'M07',
    'R9-B0':'M15', 'R9-O0':'M16', 'R9-DB-B':'M17', 'R9-DB-O':'M18',
    'R10-O0':'M19', 'nblf':'M19',
    'R9-RSP':'M21', 'r9_rsp_full':'M21',
    'P-B0': 'M03', 'P-F0': 'M04', 'P-O0': 'M05',
    'ViT-S-B0': 'M03', 'ViT-S-O0': 'M05',
    'ViT-B-B0': 'M06', 'ViT-B-O0': 'M07',
    'CTX-B': 'M08', 'CTX-O': 'M09',
    'P-R0': 'M10', 'P-A1': 'M11', 'P-A2': 'M12', 'P-A3': 'M13', 'P-A4': 'M14',
}

CORE_IDS = tuple(f'M{i:02d}' for i in range(3, 14))
GROUPS = {
    'r8': ('M03', 'M10', 'M05', 'M04'),
    'r8_main': ('M03', 'M10', 'M05', 'M04'),
    'r8_all': ('M03', 'M10', 'M05', 'M04', 'M11', 'M12', 'M13', 'M14', 'M08', 'M09', 'M06', 'M07'),
    'generalization': ('M08', 'M09', 'M06', 'M07'),
    'r9': ('M15', 'M16'),
    'r9_backbone': ('M17', 'M18'),
    'r9_all': R9_IDS,
    'r10': R10_IDS,
    'r9_rsp': RSP_IDS,
    'current': R10_IDS,
    'main': R10_IDS,
    'legacy_main': ('M03', 'M04', 'M05'),
    'backbone': ('M06', 'M07'),
    'context': ('M08', 'M09'),
    'ablations': ('M10', 'M11', 'M12', 'M13'),
    'optional': ('M14',),
    'all': CORE_IDS,
}


def get_experiment(name):
    key = ALIASES.get(name, name)
    if key not in EXPERIMENTS:
        if key in ('M01', 'M02'):
            raise ValueError(f'{key} is a preserved R3 historical run, not an R7 training config')
        raise ValueError(f'Unknown R7 experiment {name!r}; use the list command')
    return EXPERIMENTS[key]


def expand(names, with_references=True):
    result = []

    def add(token):
        spec = get_experiment(token)
        if with_references and spec.reference:
            add(spec.reference)
        if spec.id not in result:
            result.append(spec.id)

    for token in names:
        for key in GROUPS.get(token, (token,)):
            add(key)
    return result


def relay_options(spec):
    if spec.method == 'readout_only':
        return {}
    options = dict(stem_channels=48, gate_hidden=32, beta_init=0., beta_bound=4., fixed_beta=None,
                   gate_mode='learned', support_mode='measured', use_projection=True,
                   use_rgb=True, use_attenuation=True, plain_cnn=False, readout_only=False)
    patches = {
        'full': {},
        'readout_only': dict(readout_only=True),
        'no_projection': dict(use_projection=False),
        'uniform': dict(fixed_beta=0.),
        'no_rgb': dict(use_rgb=False),
        'no_gate': dict(gate_mode='half'),
        'no_attenuation': dict(use_attenuation=False),
    }
    options.update(patches[spec.intervention])
    if spec.protocol in (R9_PROTOCOL, R10_PROTOCOL, RSP_PROTOCOL, CFRT_PROTOCOL):
        options.pop('readout_only')
    if spec.protocol == R10_PROTOCOL:
        options.update(novelty=True, branch_budget=(.1, .1, .1, .1), rgb_budget=.05,
                       residual_scale=1., fusion_eps=1e-6)
    if spec.id == "M23":
        options.update(transport=True, transport_hidden=32)
    return options


def study_protocol(names):
    protocols={get_experiment(name).protocol for name in names}
    if len(protocols)!=1:
        raise ValueError('Different protocol experiment IDs require separate study/output directories')
    return protocols.pop()
