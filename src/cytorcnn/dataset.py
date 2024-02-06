import os


class Dataset:
    def __init__(self, image_folder, coco_path):
        """
        Describe the folder structure and file format
        that this method expects
        """
        self.image_folder_path = image_folder
        self.coco_file_path = coco_path
        self.size = len(list(os.listdir(image_folder)))
