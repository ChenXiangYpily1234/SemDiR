"""Training iteration clock retained without depending on historical run receipts."""
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SemDiRIterationHook(Hook):
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if hasattr(model, 'set_region_preservation_step'):
            model.set_region_preservation_step(runner.iter + 1)
