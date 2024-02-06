from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY


def scale_boxes_relative_to_center(boxes, scale_x, scale_y):
    """
    Input: type Boxes from detectron2
    Assume:
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height of image].
    """
    widths = boxes.tensor[:, 2] - boxes.tensor[:, 0]
    heights = boxes.tensor[:, 3] - boxes.tensor[:, 1]

    new_widths = scale_x * widths
    new_heights = scale_y * heights

    width_diffs = new_widths - widths
    height_diffs = new_heights - heights

    boxes.tensor[:, 0] -= width_diffs / 2.0  # x0
    boxes.tensor[:, 2] += width_diffs / 2.0  # x1
    boxes.tensor[:, 1] -= height_diffs / 2.0  # y0
    boxes.tensor[:, 3] += height_diffs / 2.0  # y1


@META_ARCH_REGISTRY.register()
class CytoRCNN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        nuclei_proposal_generator: nn.Module,
        cell_proposal_generator: nn.Module,
        nuclei_roi_heads: nn.Module,
        cell_roi_heads: nn.Module,
        nuclei_classes: List[int],
        cell_classes: List[int],
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone

        self.nuclei_classes = nuclei_classes
        self.cell_classes = cell_classes
        self.nuclei_proposal_generator = nuclei_proposal_generator
        self.cell_proposal_generator = cell_proposal_generator
        self.nuclei_roi_heads = nuclei_roi_heads
        self.cell_roi_heads = cell_roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert (
                input_format is not None
            ), "input_format is required for visualization!"

        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "nuclei_proposal_generator": build_proposal_generator(
                cfg, backbone.output_shape()
            ),
            "cell_proposal_generator": build_proposal_generator(
                cfg, backbone.output_shape()
            ),
            "nuclei_roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "cell_roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "nuclei_classes": cfg.MODEL.NUCLEI_CLASSES,
            "cell_classes": cfg.MODEL.CELL_CLASSES,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input_dict, prop in zip(batched_inputs, proposals):
            img = input_dict["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input_dict["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break

    def split_gt(self, gt_instances):
        """
        Returns: nuc_gt_instances, cell_gt_instances.
        Remember that bg_label = num_classes
        """
        storage = get_event_storage()
        num_cells = 0
        num_nuclei = 0

        nuclei_instances = []
        cell_instances = []
        for instances in gt_instances:
            ind = instances.gt_classes == self.nuclei_classes[0]
            for nucleus_class in self.nuclei_classes:
                ind = ind | (instances.gt_classes == nucleus_class)
            nuclei = Instances(
                image_size=instances.image_size,
                gt_boxes=instances.gt_boxes[ind],
                gt_classes=instances.gt_classes[ind],
                gt_masks=instances.gt_masks[ind],
            )
            cells = Instances(
                image_size=instances.image_size,
                gt_boxes=instances.gt_boxes[~ind],
                gt_classes=instances.gt_classes[~ind],
                gt_masks=instances.gt_masks[~ind],
            )
            cells.gt_classes[cells.gt_classes == 1] = 0
            num_nuclei += len(nuclei)
            num_cells += len(cells)

            nuclei_instances.append(nuclei)
            cell_instances.append(cells)

        storage.put_scalar("train_num_nuclei", num_nuclei)
        storage.put_scalar("train_num_cells", num_cells)

        return nuclei_instances, cell_instances

    def combine_instances(self, nuclei_instances, cell_instances):
        """
        Merge and ensure the class labels are correct
        """
        all_instances = []
        for i in range(len(nuclei_instances)):
            nuclei = nuclei_instances[i]
            nuclei.pred_classes[nuclei.pred_classes == 1] = 2

            cells = cell_instances[i]
            cells.pred_classes[cells.pred_classes == 1] = 2
            cells.pred_classes[cells.pred_classes == 0] = 1

            merged = Instances.cat([nuclei, cells])
            all_instances.append(merged)
        return all_instances

    def generate_cell_proposals(self, nuclei_instances):
        """
        Input: list of Instances. Each must contain attribute proposal_boxes (in training) or pred_boxes (in inference)
        Outout: list of Instance with attributes: proposal_boxes
        """
        all_instances = []
        for i in range(len(nuclei_instances)):
            nuclei = nuclei_instances[i]

            if self.training:
                cell_pred_boxes = nuclei.proposal_boxes.clone()
            else:
                cell_pred_boxes = nuclei.pred_boxes.clone()

            scale_boxes_relative_to_center(cell_pred_boxes, 2, 2)
            cell_pred_boxes.clip(nuclei.image_size)

            cells = Instances(
                image_size=nuclei.image_size, proposal_boxes=cell_pred_boxes
            )

            all_instances.append(cells)
        return all_instances

    def calculate_objectness_for_cell_proposals(self, cell_proposals):
        """
        Input: list of Instances with attribute proposal_boxes
        Output: Tuple of instances and losses
        Instances have new attribute `objectness_logits`
        """
        for i in range(len(cell_proposals)):
            cells = cell_proposals[i]
            num_cells = len(cells)
            objectness_logits = torch.zeros(num_cells).to(self.device)
            cells.objectness_logits = objectness_logits
        return cell_proposals, None

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
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
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # Split up gt instances in nuclei instances and cell instances
        nuclei_gt_instances, cell_gt_instances = self.split_gt(gt_instances)

        # 1 Train nuclei proposals
        features = self.backbone(images.tensor)
        nuclei_proposals, nuclei_proposal_losses = self.nuclei_proposal_generator(
            images, features, nuclei_gt_instances
        )
        nuclei_proposal_losses = {
            "nuc_" + key: value for key, value in nuclei_proposal_losses.items()
        }

        # 2 Train Nuclei CLS score, bbox and mask.
        nuclei_predictions, nuclei_detector_losses = self.nuclei_roi_heads(
            images, features, nuclei_proposals, nuclei_gt_instances
        )
        nuclei_detector_losses = {
            "nuc_" + key: value for key, value in nuclei_detector_losses.items()
        }

        # 3 Generate cell proposals
        with torch.no_grad():
            cell_proposals = self.generate_cell_proposals(nuclei_predictions)

        # 4 Region score
        cell_proposals, _ = self.calculate_objectness_for_cell_proposals(cell_proposals)

        # 5 Train cell bbox and mask
        _, cell_detector_losses = self.cell_roi_heads(
            images, features, cell_proposals, cell_gt_instances
        )
        cell_detector_losses = {
            "cell_" + key: value for key, value in cell_detector_losses.items()
        }

        losses = {}
        losses.update(nuclei_proposal_losses)
        losses.update(nuclei_detector_losses)
        losses.update(cell_detector_losses)

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
        features = self.backbone(images.tensor)

        nuclei_proposals, _ = self.nuclei_proposal_generator(images, features, None)
        nuclei_results, _ = self.nuclei_roi_heads(
            images, features, nuclei_proposals, None
        )

        cell_proposals = self.generate_cell_proposals(nuclei_results)
        cell_results, _ = self.cell_roi_heads(images, features, cell_proposals, None)

        results = self.combine_instances(nuclei_results, cell_results)

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            return CytoRCNN._postprocess(results, batched_inputs, images.image_sizes)

        return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(
        instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes
    ):
        """
        Rescale the output instances to the target size.
        """
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            results = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": results})
        return processed_results
