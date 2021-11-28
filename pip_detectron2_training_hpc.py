"""
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF
"""
import os
from pathlib import Path
from typing import Tuple

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

setup_logger()


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


def main():
    register_dataset(Path("data/set2"), "pip_train", "pip_validation")
    train("pip_train")


if __name__ == '__main__':
    main()
