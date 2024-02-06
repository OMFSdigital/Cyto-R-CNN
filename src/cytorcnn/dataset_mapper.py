from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
import torch


class DatasetMapperWithExtraKeys(DatasetMapper):
    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        """
        This is just the default implementation while also adding the cell_id to the target objects
        """
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(
                obj,
                transforms,
                image_shape,
                keypoint_hflip_indices=self.keypoint_hflip_indices,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        cell_ids = [anno["gt_cell_id"] for anno in annos]
        cell_ids = torch.tensor(cell_ids, dtype=torch.int64)
        instances.gt_cell_ids = cell_ids

        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
