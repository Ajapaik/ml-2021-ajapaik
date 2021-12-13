# Picture-in-Picture Detection (PIP)

Artefacts that aren't committed to the repository:

- **Trained model** files: https://owncloud.ut.ee/owncloud/index.php/s/4Hf8f6FABEmSKo6 (~190 MB)

The tool uses [detectron2](https://github.com/facebookresearch/detectron2) to train it on detection of picture-in-picture areas from photographs with additional decorative elements, frames, collages, e.g., https://ajapaik.ee/?album=56263&photo=276552&order1=time&order2=added&page=1.

## Setup

```shell
# getting the source
$ git clone https://github.com/iharsuvorau/ml-2021-ajapaik.git
$ cd pip_detector
# pytorch and detectron installation, see https://pytorch.org
$ virtualenv venv
(venv) $ source venv/bin/activate
(venv) $ python -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html 
(venv) $ python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
# installing pip_detector
(venv) $ python -m pip install .
```

## Usage

```shell
(venv) $ pip_detector detect-batch -p data/set2 -i data/set2/validation -o results/set2_validation -d cpu
```

If a user provides a path to a folder with images and an output folder, the tool saves  visualized predictions on images and a JSON file with a list of objects. Each object has only 2 attributes: `file_name` and `bbox`. `bbox` is an list of lists, because one image can have multiple images inside (e.g., collages like https://ajapaik.ee/?album=56263&photo=276552&order1=time&order2=added&page=1)

This is a machine-readable sample output for PIP detection tool:

```json
[
  {
    "file_name": "437873.jpg",
    "bbox": [
      [
        60.952701568603516,
        57.085933685302734,
        337.5085754394531,
        256.60528564453125
      ]
    ]
  },
  {
    "file_name": "432878.jpg",
    "bbox": [
      [
        67.50006103515625,
        46.967918395996094,
        339.67572021484375,
        252.1700439453125
      ]
    ]
  }
]
```

Use `--help` flag to get more help:

```shell
(venv) $ pip_detector --help
(venv) $ pip_detector <command_name> --help
```

## Quick help

```
Usage: pip_detector [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  detect
  detect-batch
  train
```

```
Usage: pip_detector detect [OPTIONS]

Options:
  -m, --model_path PATH
  -p, --dataset_path PATH  [required]
  -i, --input_path PATH    [required]
  -t, --threshold FLOAT
  -d, --device TEXT
```

```
Usage: pip_detector detect-batch [OPTIONS]

Options:
  -m, --model_path PATH
  -p, --dataset_path PATH  [required]
  -i, --input_path PATH    [required]
  -o, --output_path PATH
  -t, --threshold FLOAT
  -d, --device TEXT
```

```
Usage: pip_detector train [OPTIONS]

Options:
  -p, --dataset_path PATH  [required]
  -d, --device TEXT
```
