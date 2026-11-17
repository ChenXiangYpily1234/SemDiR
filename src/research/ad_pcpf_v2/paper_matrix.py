"""Canonical M01-M14 run IDs and paper-row aliases from the R7 workbook."""
from __future__ import annotations
from .experiments import EXPERIMENTS, R9_PROTOCOL, R10_PROTOCOL, RSP_PROTOCOL, CFRT_PROTOCOL


HISTORICAL = (
    dict(run_id='M01', method='Vanilla + SI', si=.05, ss_miou=45.56, ss_macc=56.65, ss_aacc=81.77,
         ms_miou=46.25, ms_macc=56.00, ms_aacc=82.30,
         checkpoint_sha256='0840ef6eb0df09a6b5857b803e7f663521918c903ad9d2f5e63e5359d8e4b247',
         study='ADPCPF-PAPER-V2-R3-MT8-IOU-COST-CPUFLOPS'),
    dict(run_id='M02', method='AD-PCPF V2 Full + SI', si=.05, ss_miou=45.44, ss_macc=56.39, ss_aacc=81.73,
         ms_miou=45.82, ms_macc=55.55, ms_aacc=82.31,
         checkpoint_sha256='fc845c1444a09fc50f6575f4e673e6fd0799cd6121e8aed858e866143dcd4f90',
         study='ADPCPF-PAPER-V2-R3-MT8-IOU-COST-CPUFLOPS'),
)

LITERATURE_ADE20K = (
    dict(backbone='DeiT-S/16', method='DeiT-S', params=52.1, flops=360, ss=42.9, ms=43.8,
         source='FDAM Table 2'),
    dict(backbone='DeiT-S/16', method='DeiT-S + FDAM', params=52.6, flops=363, ss=44.3, ms=45.0,
         source='FDAM Table 2'),
    dict(backbone='ViT-S/16', method='ViT-S', params=53.6, flops=None, ss=44.6, ms=45.7,
         source='META Table 3'),
    dict(backbone='ViT-S/16', method='ViT-Adapter-S', params=57.6, flops=None, ss=46.2, ms=47.1,
         source='META Table 3'),
    dict(backbone='ViT-S/16', method='LoSA-S', params=54.9, flops=None, ss=45.8, ms=46.6,
         source='META Table 3'),
    dict(backbone='ViT-S/16', method='META-S', params=54.7, flops=None, ss=47.2, ms=48.0,
         source='META Table 3'),
    dict(backbone='ViT-B/16', method='ViT-B', params=127.3, flops=None, ss=46.1, ms=47.1,
         source='META Table 3'),
    dict(backbone='ViT-B/16', method='ViT-Adapter-B', params=133.9, flops=None, ss=48.8, ms=49.7,
         source='META Table 3'),
    dict(backbone='ViT-B/16', method='LoSA-B', params=131.2, flops=None, ss=47.3, ms=48.5,
         source='META Table 3'),
    dict(backbone='ViT-B/16', method='META-B', params=129.1, flops=None, ss=49.4, ms=50.5,
         source='META Table 3'),
)

# One run ID appears in more than one paper table only when its dataset,
# backbone checkpoint, config SHA, SI weight, and terminal checkpoint are the same.
TABLES = (
    ('ADE20K DeiT-S main comparison', (
        dict(paper_id='P-B0', run_id='M03', method='Vanilla', registry_method='vanilla',
             dataset='ade20k', backbone='small', steps=160000, si=0., reference=None),
        dict(paper_id='P-F0', run_id='M04', method='FDAM matched adaptation', registry_method='fdam',
             dataset='ade20k', backbone='small', steps=160000, si=0., reference='P-B0'),
        dict(paper_id='P-O0', run_id='M05', method='AD-PCPF V2 Full', registry_method='adpcpf',
             dataset='ade20k', backbone='small', steps=160000, si=0., reference='P-B0'),
    )),
    ('ADE20K local backbone pairs', (
        dict(paper_id='ViT-S Vanilla', run_id='M03', method='Vanilla (matched)', registry_method='vanilla',
             dataset='ade20k', backbone='small', steps=160000, si=0., reference=None),
        dict(paper_id='ViT-S AD-PCPF', run_id='M05', method='AD-PCPF V2', registry_method='adpcpf',
             dataset='ade20k', backbone='small', steps=160000, si=0., reference='ViT-S Vanilla'),
        dict(paper_id='ViT-B Vanilla', run_id='M06', method='Vanilla (matched)', registry_method='vanilla',
             dataset='ade20k', backbone='base', steps=160000, si=0., reference=None),
        dict(paper_id='ViT-B AD-PCPF', run_id='M07', method='AD-PCPF V2', registry_method='adpcpf',
             dataset='ade20k', backbone='base', steps=160000, si=0., reference='ViT-B Vanilla'),
    )),
    ('PASCAL-Context-59', (
        dict(paper_id='CTX-B', run_id='M08', method='Vanilla', registry_method='vanilla',
             dataset='context59', backbone='small', steps=80000, si=0., reference=None),
        dict(paper_id='CTX-O', run_id='M09', method='AD-PCPF V2 Full', registry_method='adpcpf',
             dataset='context59', backbone='small', steps=80000, si=0., reference='CTX-B'),
    )),
    ('ADE20K ablation', (
        dict(paper_id='P-B0', run_id='M03', method='Vanilla reference', registry_method='vanilla',
             intervention='full', dataset='ade20k', backbone='small', steps=160000, si=0., reference='P-O0'),
        dict(paper_id='P-R0', run_id='M10', method='Semantic readout only', registry_method='readout_only',
             intervention='readout_only', dataset='ade20k', backbone='small', steps=160000, si=0., reference='P-O0'),
        dict(paper_id='P-A1', run_id='M11', method='No PCPF correction', registry_method='adpcpf',
             intervention='no_projection', dataset='ade20k', backbone='small', steps=160000, si=0., reference='P-O0'),
        dict(paper_id='P-A2', run_id='M12', method='Uniform depth weights', registry_method='adpcpf',
             intervention='uniform', dataset='ade20k', backbone='small', steps=160000, si=0., reference='P-O0'),
        dict(paper_id='P-A3', run_id='M13', method='No HR-RGB branch', registry_method='adpcpf',
             intervention='no_rgb', dataset='ade20k', backbone='small', steps=160000, si=0., reference='P-O0'),
        dict(paper_id='P-A4', run_id='M14', method='Fixed gate 0.5 (optional)', registry_method='adpcpf',
             intervention='no_gate', dataset='ade20k', backbone='small', steps=160000, si=0., reference='P-O0',
             optional=True),
        dict(paper_id='P-O0', run_id='M05', method='AD-PCPF V2 Full', registry_method='adpcpf',
             intervention='full', dataset='ade20k', backbone='small', steps=160000, si=0., reference=None),
    )),
)


def _format(value):
    return '' if value is None else f'{float(value):.2f}'


def render_tables(manifest, scores, multi_scores, costs, evaluations):
    if manifest["protocol"] in (R9_PROTOCOL, R10_PROTOCOL, RSP_PROTOCOL, CFRT_PROTOCOL):
        return render_r9_tables(manifest, scores, multi_scores, costs, evaluations)
    prepared = {row['id']: row for row in manifest['experiments']}
    runtime = {key: manifest[key] for key in ('world_size', 'per_gpu', 'global_batch', 'seed')}
    lines = [
        f'Execution parameters: {runtime}. Compare runs only within the same registered study.', '',
        '# R7 spreadsheet-facing results', '',
        '> M01-M02 are historical R3 results and are not rerun. New R7 runs begin at M03. '
        'P-/CTX labels are paper-row aliases, not extra training IDs.', '',
    ]
    for title, rows in TABLES:
        lines += [f'## {title}', '',
                  '| Run ID | Paper ID | Method / intervention | SS mIoU (%) | MS mIoU (%) | ΔSS (pp) | Params (M) | FLOPs (G) | Status |',
                  '|---|---|---|---:|---:|---:|---:|---:|---|']
        valid_scores = {}
        for row in rows:
            run_id = row['run_id']
            prepared_row = prepared.get(run_id)
            if prepared_row and scores.get(run_id) is not None:
                spec = prepared_row['spec']
                matches = (spec['dataset'] == row['dataset'] and spec['backbone'] == row['backbone']
                           and spec['method'] == row['registry_method']
                           and spec['intervention'] == row.get('intervention', 'full')
                           and float(spec['si']) == row['si']
                           and int(prepared_row['steps']) == row['steps'])
                if matches:
                    valid_scores[row['paper_id']] = float(scores[run_id])
        for row in rows:
            run_id = row['run_id']
            prepared_row = prepared.get(run_id)
            ss = ms = params = flops = delta = None
            status = 'not prepared'
            if not prepared_row:
                status = 'not registered in this study'
            else:
                spec = prepared_row['spec']
                matches = (spec['dataset'] == row['dataset'] and spec['backbone'] == row['backbone']
                           and spec['method'] == row['registry_method']
                           and spec['intervention'] == row.get('intervention', 'full')
                           and float(spec['si']) == row['si']
                           and int(prepared_row['steps']) == row['steps'])
                if not matches:
                    status = 'config mismatch'
                elif scores.get(run_id) is None:
                    status = 'prepared; not sealed'
                else:
                    ss = float(scores[run_id])
                    ms = multi_scores.get(run_id)
                    cost = costs.get(run_id) or {}
                    params = cost.get('total_parameters')
                    params = params / 1e6 if params else None
                    flops = cost.get('flops_counted_G')
                    status = 'SS sealed; MS pending' if ms is None else 'ready'
            if ss is not None and row.get('reference') in valid_scores:
                delta = ss - valid_scores[row['reference']]
            lines.append(f"| {run_id} | {row['paper_id']} | {row['method']} / {row.get('intervention', '—')} | "
                         f"{_format(ss)} | {_format(ms)} | {_format(delta)} | {_format(params)} | {_format(flops)} | {status} |")
        lines.append('')
    lines += [
        '## Preserved historical runs', '',
        '| Run ID | Method | SI | SS mIoU | MS mIoU | Status |',
        '|---|---|---:|---:|---:|---|',
    ]
    for row in HISTORICAL:
        lines.append(f"| {row['run_id']} | {row['method']} | {row['si']:.2f} | "
                     f"{row['ss_miou']:.2f} | {row['ms_miou']:.2f} | preserved R3 checkpoint / receipt |")
    lines += ['', 'M01-M02 must not fill R7 no-SI rows. The ADE20K primary local claim is P-O0 − P-B0; '
              'Context-59 deltas remain within that dataset. FLOPs values require coverage and operator-scope notes.', '']
    return '\n'.join(lines)


def render_r9_tables(manifest,scores,multi_scores,costs,evaluations):
    layout={k:manifest[k] for k in ('world_size','per_gpu','global_batch','seed')}
    title = 'R10 NBLF results' if manifest.get('protocol') == R10_PROTOCOL else 'R9 semantic-pyramid results'
    if manifest.get('protocol') == RSP_PROTOCOL:
        title = 'R9-RSP results (single-forward region class-balanced margin)'
    if manifest.get('protocol') == CFRT_PROTOCOL:
        title = 'M23 CoFiRT-RSP results'
    lines=['# '+title,'',f'Execution: {layout}', '',
           '| ID | Backbone | Method | SS mIoU | MS mIoU | Params (M) | FLOPs (G) | FLOP coverage |',
           '|---|---|---|---:|---:|---:|---:|---|']
    def fmt(x):
        return '' if x is None else f'{float(x):.2f}'
    for row in manifest['experiments']:
        name=row['id']; spec=EXPERIMENTS[name]; cost=costs.get(name) or {}
        params=cost.get('total_parameters')
        lines.append(f'| {name} | {spec.backbone} | {spec.method} | {fmt(scores.get(name))} | '
                     f'{fmt(multi_scores.get(name))} | {fmt(params/1e6 if params else None)} | '
                     f'{fmt(cost.get("flops_counted_G"))} | {cost.get("flop_status", "")} |')
    if manifest.get('protocol') == R10_PROTOCOL:
        lines += ['', 'External baseline: preserved M03; historical R9 comparison: M16. '
                  'Inference lambda scans are exploratory diagnostics, not trained M19 results.']
    if manifest.get('protocol') == RSP_PROTOCOL:
        lines += ['', 'Fresh ImageNet-1K initialization; no M03 teacher or extra network forward. '
                  'GT regions and margin are training-only; compare with preserved M03 and M16.']
    return '\n'.join(lines)+'\n'
