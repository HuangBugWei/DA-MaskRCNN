import torch
import time
import numpy as np
from detectron2.engine import AMPTrainer, SimpleTrainer
from PIL import Image

class customSimpleTrainer(SimpleTrainer):
    def __init__(self, 
                 model, 
                 data_loader,
                 target_data_loader,
                 optimizer):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        # unsupported = "AMPTrainer does not support single-process multi-device training!"
        # if isinstance(model, DistributedDataParallel):
        #     assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        # assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer)

        self.target_data_loader = target_data_loader
        # to access the data loader iterator, call `self._data_loader_iter`
        self._target_data_loader_iter_obj = None

    @property
    def _target_data_loader_iter(self):
        if self._target_data_loader_iter_obj is None:
            self._target_data_loader_iter_obj = iter(self.target_data_loader)
        
        return self._target_data_loader_iter_obj

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        if self.target_data_loader is None:
            # print("no target")
            pass
        else:
            # print("have target")
            targetData = next(self._target_data_loader_iter)
            # print(targetData[0])
            # print(targetData[0]["image"].shape)
            # Image.fromarray(targetData[0]["image"].permute(1, 2, 0).numpy()).show()
            data += targetData
        data_time = time.perf_counter() - start
        
        p = float(self.iter / self.max_iter)
        alpha = 2. / ( 1. + np.exp(-10 * p)) - 1
        # alpha = 2. / ( 1. + np.exp(-2 * p)) - 1
        # alpha3 = alpha if alpha < 0.5 else 0.5
        # alpha3 = alpha
        # alpha4 = alpha if alpha < 0.5 else 0.5
        # alpha5 = alpha if alpha < 0.1 else 0.1
        alpha2 = alpha
        alpha3 = alpha
        alpha4 = alpha
        alpha5 = alpha

        loss_dict = self.model(data, None, False, alpha2, alpha3, alpha4, alpha5)
        
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()


class customAMPTrainer(AMPTrainer):
    def __init__(self, 
                 model, 
                 data_loader,
                 target_data_loader,
                 optimizer, 
                 grad_scaler=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        # unsupported = "AMPTrainer does not support single-process multi-device training!"
        # if isinstance(model, DistributedDataParallel):
        #     assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        # assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, grad_scaler)

        self.target_data_loader = target_data_loader
        # to access the data loader iterator, call `self._data_loader_iter`
        self._target_data_loader_iter_obj = None

    @property
    def _target_data_loader_iter(self):
        if self._target_data_loader_iter_obj is None:
            self._target_data_loader_iter_obj = iter(self.target_data_loader)
        
        return self._target_data_loader_iter_obj

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        targetData = next(self._target_data_loader_iter)
        p = float(self.iter / self.max_iter)
        alpha = 2. / ( 1. + np.exp(-10 * p)) - 1
        alpha3 = alpha if alpha < 0.5 else 0.5
        alpha4 = alpha if alpha < 0.5 else 0.5
        alpha5 = alpha if alpha < 0.1 else 0.1

        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data, False, alpha3, alpha4, alpha5)
            loss_dict_target = self.model(targetData, True, alpha3, alpha4, alpha5)
            loss_dict["loss_r3"] += loss_dict_target["loss_r3"]
            loss_dict["loss_r4"] += loss_dict_target["loss_r4"]
            loss_dict["loss_r5"] += loss_dict_target["loss_r5"]

            loss_dict["loss_r3"] *= 0.5
            loss_dict["loss_r4"] *= 0.5
            loss_dict["loss_r5"] *= 0.5
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()