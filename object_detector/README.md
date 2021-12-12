
#  Object Detection
This tool uses [Yolo4](https://github.com/AlexeyAB/darknet) and OpenCV's [DNN Module](https://docs.opencv.org/3.4/d6/d0f/group__dnn.html) to detect common objects in images.

##  Setup
```
# clone the project
$ git clone https://github.com/iharsuvorau/ml-2021-ajapaik.git
$ cd object_detector
$ virtualenv venv
(venv) $ source venv/bin/activate
(venv) $ python -m pip install .
# Download some Yolo4 default Config files
$ wget https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
$ wget https://github.com/AlexeyAB/darknet/blob/master/cfg/coco.names
# ... and training weight file
$ wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```
##  Usage
```
# Running Object Detection of folder containing images containing images with [jpg,jpeg,gifs] as extensions.
# if coco.names, yolov4.{cfg,weights} are relative to the cli too, they would be read automatically
(venv) object_detector -d /path/to/image/folder/ -o /path/to/where/results/would/be/stored
# else
(venv) object_detector -l "/path/to/coco.names" -c "/path/to/yolov4.cfg" -w "/path/to/yolov4.weights" -d /path/to/image/folder/ -o /path/to/where/results/would/be/stored
# single image detection
(venv) object_detector -i /path/to/single/image/file
```
``-d`` and ``-o`` are optional args, if there are not provided, the directory relative to the cli tool would be used.
As output for these commands, a json file containing the bounding boxes coordinates for each image is saved in ``-o`` as ``image_name.ext.json``

This is a machine-readable sample output for the object detection tool:
```
[{
"detection_count": 22,
"file_name": "/AjapaikImages/hakaniementori.jpg",
"confidence_threshold": 0.25
}, {
"label": "person",
"confidence": "96.04",
"left_x": 773,
"top_y": 547,
"width": 83,
"height": 123
},{
"label": "horse",
"confidence": "67.89",
"left_x": 914,
"top_y": 408,
"width": 86,
"height": 87
},{...}]
```
##  Quick Help
```
(venv) $ object_detector --help
```
```
usage: object_detector [-h] [--config_path CONFIG_PATH] [--weight_path WEIGHT_PATH] [--label_path LABEL_PATH] [--image_path IMAGE_PATH] [--threshold THRESHOLD]
[--image-dir INPUT_DIR] [--output-dir OUTPUT_DIR]

optional arguments:
-h, --help show this help message and exit
--config_path CONFIG_PATH, -c CONFIG_PATH Path to yolov4.cfg

--weight_path WEIGHT_PATH, -w WEIGHT_PATH Path to yolov4.weights.

--label_path LABEL_PATH, -l LABEL_PATH Path to coco.names.

--image_path IMAGE_PATH, -i IMAGE_PATH Path to Image file. Leaving Blank Searches the current directory

--threshold THRESHOLD, -t THRESHOLD Detection Confidence Threshold to apply

--image-dir INPUT_DIR, -d INPUT_DIR Directory containing image file

--output-dir OUTPUT_DIR, -o OUTPUT_DIR Directory where output should be stored
```

##  Statistics

Object detection is ran against the coco dataset(80 [classes](https://github.com/AlexeyAB/darknet/blob/master/cfg/coco.names), 328k images). We could detect windmills, churches, lighthouses, people, bicycles, cars, among others objects in the random partner's images .