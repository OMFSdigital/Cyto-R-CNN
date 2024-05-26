from detectron2 import model_zoo
from detectron2.config import get_cfg

from .architecture import cytorcnn  # pylint: disable=unused-import


def detectron_base_config(name):
    dataset_name = name

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.META_ARCHITECTURE = "CytoRCNN"
    cfg.MODEL.BACKBONE.FREEZE_AT = 1
    cfg.MODEL.NUCLEI_CLASSES = [0]  # The classes that correspond to nuclei
    cfg.MODEL.CELL_CLASSES = [1]  # The classes that correspond to cells
    cfg.MODEL.CELL_SCALES = [
        2
    ]  # The factor by which each nuclei should be scaled to create a proposal for the whole cell
    cfg.MODEL.ROI_BOX_HEAD.CELL_BBOX_REG_LOSS_WEIGHT = 1
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [8, 16, 32, 64]
    cfg.MODEL.ROI_HEADS.NMS = 0.4
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

    cfg.DATASETS.TRAIN = (dataset_name + "_train",)
    cfg.DATASETS.TEST = (dataset_name + "_val",)
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.INPUT.MIN_SIZE_TEST = 256
    cfg.INPUT.MAX_SIZE_TEST = 256
    cfg.INPUT.MIN_SIZE_TRAIN = 256
    cfg.INPUT.MAX_SIZE_TRAIN = 256

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )

    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.NUM_GPUS = 1

    cfg.TEST.DETECTIONS_PER_IMAGE = 200
    cfg.TEST.AUG.FLIP = True
    cfg.TEST.AUG.ROT = True

    return cfg
