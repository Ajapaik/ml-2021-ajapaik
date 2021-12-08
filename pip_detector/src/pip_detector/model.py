"""
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF
"""
import copy
import json
import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

setup_logger()


class Model:
    dataset_name: Optional[str] = None
    dataset_name_train: Optional[str] = None
    dataset_name_validation: Optional[str] = None
    device: Optional[str] = None  # 'gpu' | 'cpu'
    dataset_location: Optional[Path] = None

    def __init__(self, dataset_location=None, device=None):
        self.dataset_location = Path(dataset_location)
        self.dataset_name = self.dataset_location.name
        self.dataset_name_train = self.dataset_location.name + '_train'
        self.dataset_name_validation = self.dataset_location.name + '_validation'
        self.device = device

    def train(self):
        self._register_dataset()
        cfg = self._make_train_config()
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def detect(self, image_path: Path, threshold: float = 0.5, show_in_window=False):
        self._register_dataset()
        cfg = get_cfg()
        if self.device:
            cfg.MODEL.DEVICE = self.device
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)

        im = cv2.imread(image_path.__str__())

        # if show_in_window:
        # cv2.imshow('ImageWindow', im)
        # cv2.waitKey()

        outputs = predictor(im)
        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        print(f'Class: {outputs["instances"].pred_classes}')
        print(f'Bbox: {outputs["instances"].pred_boxes}')

        metadata = MetadataCatalog.get(self.dataset_name)
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1,
                       # instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_image = out.get_image()[:, :, ::-1]
        cv2.imwrite('detected.jpg', output_image)

    def inference_and_validation(self):
        cfg = get_cfg()
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)

        metadata = MetadataCatalog.get(self.dataset_name_train)

        _, _, validation_path, validation_labels_path = self._make_paths()
        with validation_labels_path.open("r") as f:
            data = json.load(f)
        data = self._add_bbox_mode(data)
        img_data = data["images"][0]

        im = cv2.imread(str(validation_path / img_data["file_name"]))
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=1,
            # instance_mode=ColorMode.IMAGE_BW
            # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow("Preview", out.get_image()[:, :, ::-1])
        # cv2.waitKey()
        print(f'Class: {outputs["instances"].pred_classes}')
        print(f'Bbox: {outputs["instances"].pred_boxes}')

    def _add_bbox_mode(data: dict) -> dict:
        data = copy.copy(data)
        for note in data["annotations"]:
            note["bbox_mode"] = BoxMode.XYXY_ABS
        return data

    def _register_dataset(self):
        train_path, train_labels_path, validation_path, validation_labels_path = self._make_paths()
        register_coco_instances(self.dataset_name_train, {}, str(train_labels_path), str(train_path))
        register_coco_instances(self.dataset_name_validation, {}, str(validation_labels_path), str(validation_path))

    def _make_train_config(self) -> CfgNode:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = (self.dataset_name_train,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

        if self.device:
            cfg.MODEL.DEVICE = self.device
        return cfg

    def _make_paths(self) -> Tuple[Path, Path, Path, Path]:
        train_path = self.dataset_location / "train"
        validation_path = self.dataset_location / "validation"
        train_labels_path = train_path / "labels.json"
        validation_labels_path = validation_path / "labels.json"
        return train_path, train_labels_path, validation_path, validation_labels_path


# def visualize_image(img_path: Path, annotations: dict, metadata: Metadata):
#     img = cv2.imread(str(img_path))
#     viz = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
#     out = viz.draw_dataset_dict(annotations)
#     cv2.imshow("Preview", out.get_image()[:, :, ::-1])
#     cv2.waitKey()


# def visualize_dataset(dir_path: Path, dataset_name: str):
#     train_path, train_labels_path, _, _ = make_paths(dir_path)
#     metadata = MetadataCatalog.get(dataset_name)
#
#     with train_labels_path.open("r") as f:
#         data = json.load(f)
#
#     data = add_bbox_mode(data)
#
#     img_data = data["images"][0]
#     for note in data["annotations"]:
#         if note["image_id"] == img_data["id"]:
#             annotation = note
#
#     visualize_image(train_path / img_data["file_name"], {"annotations": [annotation]}, metadata)


def main():
    model = Model("data/set2")
    model.train()
    model.detect("data/color targets/412726.jpg")


# def midsize_dataset_training():
#     dataset_name = "photo_train"
#     validation_name = "photo_validation"
#     register_dataset(Path("data/set2"), dataset_name, validation_name)
#     train(dataset_name)
#     # detect("photo_train")
#     # inference_and_validation(Path("data/set1"), "photo_train")
#
#
# def midsize_dataset_detection():
#     dataset_name = "photo_train"
#     validation_name = "photo_validation"
#     register_dataset(Path("data/set2"), dataset_name, validation_name)
#     # train(dataset_name)
#     detect(dataset_name)
#     # inference_and_validation(Path("data/set1"), "photo_train")


if __name__ == '__main__':
    main()
