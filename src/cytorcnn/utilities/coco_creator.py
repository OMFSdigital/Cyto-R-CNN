import json
import numpy as np
from pycocotools import mask

from . import mask_util


def write_dict_to_file(dictionary, output_path):
    output_string = json.dumps(dictionary, indent=4)
    with open(output_path, "w") as file:
        print(output_string, file=file)


def create_coco_file_from_detectron_instances(list_of_instances, list_of_images):
    """
    Input:
    - A list of filenames
    - A list of Instances
    """
    coco = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Add images
    for i in range(len(list_of_images)):
        image_file_name = list_of_images[i]

        width = 256
        height = 256
        coco["images"].append(
            {
                "file_name": image_file_name,
                "height": height,
                "width": width,
                "id": i + 1,
            }
        )

    # Add categories
    category_nucleus = {"supercategory": "Cell", "id": 0, "name": "TUM_NUC"}
    category_cell = {"supercategory": "Cell", "id": 1, "name": "TUM_CELL"}

    coco["categories"].append(category_nucleus)
    coco["categories"].append(category_cell)

    # Add annotations
    annotation_id = 1
    for i in range(len(list_of_instances)):
        instances = list_of_instances[i]
        if len(instances) == 0:
            continue

        for k in range(len(instances)):
            obj_mask = instances.pred_masks[k].cpu().detach().numpy()  # Binary mask.
            category = instances.pred_classes[k].item()  # Binary mask.
            category = int(category)

            segmentation = mask_util.binary_mask_to_polygon(obj_mask)
            if len(segmentation) == 0:
                continue
            if len(segmentation[0]) < 3:
                continue  # Has to be a polygon

            binary_mask_encoded = mask.encode(
                np.asfortranarray(obj_mask.astype(np.uint8))
            )
            bounding_box = mask.toBbox(binary_mask_encoded)
            area = mask.area(binary_mask_encoded)
            score = instances.get("scores")[k].cpu().item()
            coco["annotations"].append(
                {
                    "segmentation": segmentation,
                    "area": area.tolist(),
                    "bbox": bounding_box.tolist(),
                    "iscrowd": 0,
                    "image_id": i + 1,
                    "category_id": category,
                    "id": annotation_id,
                    "score": score,
                }
            )
            annotation_id += 1

    return coco
