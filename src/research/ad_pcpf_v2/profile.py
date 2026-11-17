"""Full-model cost measurements on a sealed model. Does not train.

Unsupported FLOP operators are reported as incomplete coverage, never as zero
cost. Inference FLOPs exclude the auxiliary training head and SI objective.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
from .artifacts import write_once, sha256
from .evaluate import load_sealed_model


def measure_model_cost(model, cfg, *, device, checkpoint_sha256, warmup=100, repeats=1000):
    import torch
    from torch import nn
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Latency protocol requires a CUDA device')
    torch.cuda.set_device(torch.device(device))
    torch.backends.cudnn.benchmark = False
    class ForwardOnly(nn.Module):
        def __init__(self, segmentor):
            super().__init__()
            self.segmentor = segmentor
        def forward(self, x):
            return self.segmentor._forward(x, data_samples=None)
    wrapped = ForwardOnly(model).eval()
    report = dict(experiment=cfg.paper.experiment, checkpoint_sha256=checkpoint_sha256,
        profiler_sha256=sha256(__file__), device=torch.cuda.get_device_name(), torch=torch.__version__,
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        parameter_scope='all registered parameters including auxiliary head; no buffers',
        flop_input=[1,3,512,2048], flop_convention='MMEngine FMA=1; CPU trace; inference network, no SI/aux/slide/TTA',
        latency_input=[1,3,512,512], precision='BF16 autocast / FP32 master parameters',
        warmup=warmup, repeats=repeats,
        publication_measurement=warmup == 100 and repeats == 1000)
    with torch.inference_mode():
        x = torch.randn(1,3,512,512,device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(warmup):
                wrapped(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(repeats):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                wrapped(x)
                end.record()
                end.synchronize()
                durations.append(begin.elapsed_time(end))
        report.update(latency_median_ms=float(np.median(durations)), latency_p90_ms=float(np.quantile(durations,.9)),
            peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9,
            memory_scope='includes resident model/input and inference intermediates; decimal GB')
        del x
        try:
            from mmengine.analysis import FlopAnalyzer
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(min(previous_threads, 4))
            try:
                # Keep tracing device-consistent. FDAM's bicubic spectrum resize
                # uses index_select internally; MMEngine's CUDA trace creates CPU
                # indices for that path, while a CPU trace counts it successfully.
                wrapped.to('cpu').eval()
                image = torch.zeros(1,3,512,2048,device='cpu')
                analyzer = FlopAnalyzer(wrapped, (image,))
                total = analyzer.total()
                unsupported = dict(analyzer.unsupported_ops())
                report.update(flops_counted_G=total/1e9, unsupported_operators=unsupported,
                    flop_status='incomplete_operator_coverage' if unsupported else 'analyzer_covered',
                    flops_by_operator={str(k):float(v) for k,v in analyzer.by_operator().items()},
                    flop_analysis_device='CPU')
            finally:
                torch.set_num_threads(previous_threads)
        except Exception as exc:
            report.update(flops_counted_G=None, flop_status='failed', flop_error=repr(exc))
    return report, durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute or args.warmup < 1 or args.repeats < 1:
        parser.error('Pass --execute and positive warmup/repeats')
    model, cfg, manifest, reference = load_sealed_model(args.study.resolve(), args.experiment, args.device)
    report, durations = measure_model_cost(model, cfg, device=args.device,
        checkpoint_sha256=reference['checkpoint_sha256'], warmup=args.warmup, repeats=args.repeats)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / 'profile.json', report)
    np.save(out / 'latency_ms.npy', np.asarray(durations), allow_pickle=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
