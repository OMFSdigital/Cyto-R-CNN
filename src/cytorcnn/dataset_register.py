import os
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog


def register_coco_instances_with_extra_keys(
    name, metadata, json_file, image_root, extra_keys=None
):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    if extra_keys is None:
        extra_keys = []

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name, extra_keys)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )
