import json
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table
import numpy as np
from tabulate import tabulate


class ExtendedCOCOEvaluator(COCOEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_fast_impl = True

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn(  # pylint: disable=deprecated-method
                "No predictions from the model!"
            )
            return {metric: float("nan") for metric in metrics}

        results = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            f"Evaluation results for {iou_type}: \n" + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        precisions = coco_eval.eval["precision"]
        recalls = coco_eval.eval["recall"]

        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = {}

        info_dict = {}
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            name = str(name)
            results_per_category[name] = {}
            results_per_category[name]["precisions"] = []
            results_per_category[name]["recalls"] = []

            # Compute precision and recall per class
            # https://github.com/facebookresearch/detectron2/issues/1877

            for ap_threshold_index in range(9):
                precision = precisions[ap_threshold_index, :, idx, 0, -1]
                recall = recalls[ap_threshold_index, idx, 0, -1]

                ap_threshold = round(coco_eval.params.iouThrs[ap_threshold_index] * 100)
                info_dict[f"precision_iou_{ap_threshold}_class_{idx}"] = (
                    precision.tolist()
                )
                info_dict[f"recall_iou_{ap_threshold}_class_{idx}"] = (
                    coco_eval.params.recThrs.tolist()
                )

                # When there is no groundtruth of specified class, then recall will be -1
                recall = recall[recall != -1]
                recall = np.mean(recall)
                results_per_category[name]["recalls"].append(float(recall * 100))

                precision = precision[precision > -1]
                average_precision = (
                    np.mean(precision) if precision.size else float("nan")
                )
                results_per_category[name]["precisions"].append(
                    float(average_precision * 100)
                )

        with open("coco_eval.json", "w") as file:
            json.dump(info_dict, file, indent=4)

        n_cols = 10
        precisions_2d = []
        recalls_2d = []
        for idx, name in enumerate(class_names):
            precisions_2d.append([name] + results_per_category[name]["precisions"])
            recalls_2d.append([name] + results_per_category[name]["recalls"])

        precision_table = tabulate(
            precisions_2d,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[
                "Category",
                "AP50",
                "AP55",
                "AP60",
                "AP65",
                "AP70",
                "AP75",
                "AP80",
                "AP85",
                "AP90",
            ]
            * (n_cols // 2),
            numalign="left",
        )
        self._logger.info(f"Per-category {iou_type} precisions: \n" + precision_table)

        recall_table = tabulate(
            recalls_2d,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[
                "Category",
                "AR50",
                "AR55",
                "AR60",
                "AR65",
                "AR70",
                "AR75",
                "AR80",
                "AR85",
                "AR90",
            ]
            * (n_cols // 2),
            numalign="left",
        )
        self._logger.info(f"Per-category {iou_type} recalls: \n" + recall_table)

        return results
