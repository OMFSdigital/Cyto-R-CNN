from detectron2.utils.events import (
    EventWriter,
    get_event_storage,
)
import wandb


class WAndBWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    """

    def __init__(self, window_size: int = 1):
        self._window_size = window_size

    def write(self):
        storage = get_event_storage()
        stats = {}
        for key, value in storage.latest_with_smoothing_hint(self._window_size).items():
            stats[key.replace("/", "-")] = value[0]
        wandb.log(stats, step=storage.iter)
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self._writer.add_image(img_name, img, step_num)
            storage.clear_images()

    def close(self):
        wandb.finish()
