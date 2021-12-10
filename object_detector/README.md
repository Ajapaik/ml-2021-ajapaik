# Object Detection
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Setup
```
# clone the project
$ git clone https://github.com/iharsuvorau/ml-2021-ajapaik.git
$ cd object_detector
```
## Usage
```
//Use the below command to run dectection on a folder containing the images
py object_detector.py -l "/darknet/cfg/coco.names" -c "/darknet/cfg/yolov4.cfg" -w "/darknet/yolov4.weights" -d /AjapaikImages/

//Use the below command to run detection on a single image
py object_detector.py -l "../../darknet/cfg/coco.names" -c "../../darknet/cfg/yolov4.cfg" -w "../../darknet/yolov4.weights" -i "../../darknet/data/ajaipik.jpg
```


## Quick Help
To get inform on the commands to use ,run:
```sh
python3 object_detector.py -h
```
The above produce the below information:
```sh

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH, -c CONFIG_PATH
                        Path to yolov4.cfg
  --weight_path WEIGHT_PATH, -w WEIGHT_PATH
                        Path to yolov4.weights.
  --label_path LABEL_PATH, -l LABEL_PATH
                        Path to coco.names.
  --image_path IMAGE_PATH, -i IMAGE_PATH
                        Path to Image file. Leaving Blank Searches the current directory
  --threshold THRESHOLD, -t THRESHOLD
                        Detection Confidence Threshold to apply
  --image-dir DIR, -d DIR
                        Directory containing image file
```
## Output Sample
Single File detection sample
```
```
Direction Detection Sample
```

```
## statistics

