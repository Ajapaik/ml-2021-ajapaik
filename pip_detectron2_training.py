"""
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF
"""
import copy
import json
import os
from pathlib import Path
from typing import Tuple

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, Metadata
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode

setup_logger()


def add_bbox_mode(data: dict) -> dict:
    data = copy.copy(data)
    for note in data["annotations"]:
        note["bbox_mode"] = BoxMode.XYXY_ABS
    return data


def visualize_image(img_path: Path, annotations: dict, metadata: Metadata):
    img = cv2.imread(str(img_path))
    viz = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
    out = viz.draw_dataset_dict(annotations)
    cv2.imshow("Preview", out.get_image()[:, :, ::-1])
    cv2.waitKey()


def make_paths(dir_path: Path) -> Tuple[Path, Path, Path, Path]:
    train_path = dir_path / "train"
    validation_path = dir_path / "validation"
    train_labels_path = train_path / "labels.json"
    validation_labels_path = validation_path / "labels.json"
    return train_path, train_labels_path, validation_path, validation_labels_path


def register_dataset(dir_path: Path, train_name: str, validation_name: str):
    train_path, train_labels_path, validation_path, validation_labels_path = make_paths(dir_path)
    register_coco_instances(train_name, {}, str(train_labels_path), str(train_path))
    register_coco_instances(validation_name, {}, str(validation_labels_path), str(validation_path))


def visualize_dataset(dir_path: Path, dataset_name: str):
    train_path, train_labels_path, _, _ = make_paths(dir_path)
    metadata = MetadataCatalog.get(dataset_name)

    with train_labels_path.open("r") as f:
        data = json.load(f)

    data = add_bbox_mode(data)

    img_data = data["images"][0]
    for note in data["annotations"]:
        if note["image_id"] == img_data["id"]:
            annotation = note

    visualize_image(train_path / img_data["file_name"], {"annotations": [annotation]}, metadata)


def make_config(dataset_name: str) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 12
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 20  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.MODEL.DEVICE = 'cpu'
    return cfg


def train(dataset_name: str):
    cfg = make_config(dataset_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def inference_and_validation(dir_path: Path, dataset_name: str):
    cfg = make_config(dataset_name)
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(dataset_name)

    _, _, validation_path, validation_labels_path = make_paths(dir_path)
    with validation_labels_path.open("r") as f:
        data = json.load(f)
    data = add_bbox_mode(data)
    img_data = data["images"][0]

    im = cv2.imread(str(validation_path / img_data["file_name"]))
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=1,
                   # instance_mode=ColorMode.IMAGE_BW
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Preview", out.get_image()[:, :, ::-1])
    cv2.waitKey()


def detect():
    im = cv2.imread("./data/color targets/412726.jpg")
    # cv2.imshow('ImageWindow', im)
    # cv2.waitKey()

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)


def main():
    # detect()
    register_dataset(Path("data/sample2"), "photo_train", "photo_validation")
    # visualize_dataset(Path("data/sample2"), "photo_train")
    # train("photo_train")
    inference_and_validation(Path("data/sample2"), "photo_train")
    pass


if __name__ == '__main__':
    main()
