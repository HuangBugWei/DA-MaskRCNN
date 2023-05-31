from typing import List, Dict, Optional
from detectron2.structures import Instances
import torch
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.utils.events import get_event_storage

from discriminator import DiscriminatorRes3, DiscriminatorRes4, DiscriminatorRes5


class customRCNN(GeneralizedRCNN):
    def __init__(self,
                *,
                backbone,
                proposal_generator,
                roi_heads,
                pixel_mean,
                pixel_std,
                input_format,
                vis_period: int = 0):
        
        super().__init__(backbone=backbone,
                        proposal_generator=proposal_generator,
                        roi_heads=roi_heads,
                        pixel_mean=pixel_mean,
                        pixel_std=pixel_std,
                        input_format=input_format,
                        vis_period=vis_period)

        self.discriminatorRes3 = DiscriminatorRes3()
        self.discriminatorRes4 = DiscriminatorRes4()
        self.discriminatorRes5 = DiscriminatorRes5()

    def forward(self, 
                batched_inputs: List[Dict[str, torch.Tensor]], 
                on_domain_target = False, 
                alpha3 = 1,
                alpha4 = 1,
                alpha5 = 1):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        images = self.preprocess_image(batched_inputs)
        features, res = self.backbone(images.tensor)
        gt_instances = []
        selected_idx = []
        domain_label = []
        for batch_idx, x in enumerate(batched_inputs):
            if "instances" in x:
                gt_instances.append(x["instances"].to(self.device))
                selected_idx.append(batch_idx)
                domain_label.append([0])
            else:
                domain_label.append([1])
        if len(gt_instances) == 0:
            gt_instances = None

        domain_label = torch.tensor(domain_label).float().to(self.device)
        
        # modified features, filter those from target domain data
        for key in features.keys():
            features[key] = torch.index_select(features[key], 0, torch.tensor(selected_idx).to(self.device))
        
        batched_inputs = [x for batch_idx, x in enumerate(batched_inputs) if batch_idx in selected_idx]
        del images
        images = self.preprocess_image(batched_inputs)

        loss_res3 = self.discriminatorRes3(res["res3"], domain_label, alpha3)
        loss_res4 = self.discriminatorRes4(res["res4"], domain_label, alpha4)
        loss_res5 = self.discriminatorRes5(res["res5"], domain_label, alpha5)
        
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update({"loss_r3": loss_res3, "loss_r4": loss_res4, "loss_r5": loss_res5})

        return losses
    
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features, _ = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results