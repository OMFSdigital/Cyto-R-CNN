import math
from typing import Optional
import cv2
from detectron2.engine import DefaultPredictor
from .dataset import Dataset
from .dataset_register import register_coco_instances_with_extra_keys
from .detectron_config import detectron_base_config
from .utilities import (
    write_dict_to_file,
    create_coco_file_from_detectron_instances,
)
from .custom_trainer import CustomTrainer


class CytoRCNN:

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path
        self.config_name = "cytorcnn"
        self.config = detectron_base_config(self.config_name)

    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        total_num_images = train_dataset.size
        single_iteration = (
            self.config.SOLVER.NUM_GPUS * self.config.SOLVER.IMS_PER_BATCH
        )

        iterations_for_one_epoch = int(math.ceil(total_num_images / single_iteration))
        max_iter = iterations_for_one_epoch * 300
        self.config.SOLVER.MAX_ITER = max_iter
        self.config.SOLVER.STEPS = (int(max_iter / 4), int(max_iter / 2))
        self.config.SOLVER.CHECKPOINT_PERIOD = iterations_for_one_epoch * 100
        self.config.TEST.EVAL_PERIOD = iterations_for_one_epoch

        extra_keys = ["gt_cell_id"]
        register_coco_instances_with_extra_keys(
            f"{self.config_name}_train",
            {},
            train_dataset.coco_file_path,
            train_dataset.image_folder_path,
            extra_keys,
        )
        register_coco_instances_with_extra_keys(
            f"{self.config_name}_val",
            {},
            val_dataset.coco_file_path,
            val_dataset.image_folder_path,
            extra_keys,
        )

        # Start training
        trainer = CustomTrainer(self.config)
        should_resume = self.weights_path is not None
        trainer.resume_or_load(resume=should_resume)

        trainer.train()

    def predict(self, image_path):
        if self.weights_path is None:
            raise RuntimeError("Error: Load a model before calling .predict()")

        self.config.MODEL.DEVICE = "cuda"
        self.config.MODEL.WEIGHTS = self.weights_path
        self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

        predictor = DefaultPredictor(self.config)
        file_paths = [image_path]

        instances_for_each_image = []
        for path in file_paths:
            image = cv2.imread(path)
            result = predictor(image)
            instances = result["instances"]
            instances_for_each_image.append(instances)

        coco = create_coco_file_from_detectron_instances(
            instances_for_each_image, file_paths
        )
        write_dict_to_file(coco, "prediction.json")
