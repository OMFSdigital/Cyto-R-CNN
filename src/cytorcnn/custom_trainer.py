import os
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import hooks, DefaultTrainer
import detectron2.data.transforms as T
import wandb

from .dataset_mapper import DatasetMapperWithExtraKeys
from .utilities import ExtendedCOCOEvaluator
from .validation_hook import ValidationHook
from .wandbwriter import WAndBWriter


class CustomTrainer(DefaultTrainer):
    def __init__(self, config, *args, notes="", **kwargs):
        self.wandbwriter = WAndBWriter(window_size=1)
        super().__init__(config, *args, **kwargs)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Build a standard COCO image evaluator used for validation
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ExtendedCOCOEvaluator(
            dataset_name,
            cfg,
            True,
            output_folder,
            max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
        )

    def build_hooks(self):
        """
        Register a custom hook for validation rounds, were we calculate the validation_loss.
        This is helpful because the built-in CocoEvaluator only offers AP measures
        """
        new_hooks = super().build_hooks()

        mapper = DatasetMapperWithExtraKeys(
            self.cfg, True, augmentations=[T.NoOpTransform()]
        )

        new_hooks.insert(
            -1,
            ValidationHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST[0], mapper=mapper
                ),
            ),
        )

        if wandb.run is not None:
            new_hooks.append(hooks.PeriodicWriter([self.wandbwriter], period=1))

        return new_hooks

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        augmentations = [T.NoOpTransform()]
        mapper = DatasetMapperWithExtraKeys(
            cfg, is_train=False, augmentations=augmentations
        )
        return build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = [T.NoOpTransform()]
        if cfg.TEST.AUG.FLIP:
            augmentations.append(T.RandomFlip(horizontal=True, vertical=False))
            augmentations.append(T.RandomFlip(horizontal=False, vertical=True))
        if cfg.TEST.AUG.ROT:
            augmentations.append(T.RandomRotation([-45, 45], expand=False))

        mapper = DatasetMapperWithExtraKeys(
            cfg, is_train=True, augmentations=augmentations
        )
        return build_detection_train_loader(cfg, mapper=mapper)
