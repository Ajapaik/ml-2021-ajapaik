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
[{"detection_count": 3, "file_name": "/Users/dev/Downloads/AjapaikImages/VM_VMF_F_1548672.jpg", "confidence_threshold": 0.25}, {"label": "person", "confidence": "99.39", "left_X": 17, "top_y": 223, "width": 170, "heigth": 454}, {"label": "person", "confidence": "98.90", "left_X": 179, "top_y": 101, "width": 257, "heigth": 602}, {"label": "baseball bat", "confidence": "73.33", "left_X": 261, "top_y": 229, "width": 231, "heigth": 132}]

```

Directory Detection Sample
```
[{"detection_count": 20, "file_name": "/Users/dev/Downloads/AjapaikImages/muis_ocgMZCm.jpg", "confidence_threshold": 0.25}, {"label": "person", "confidence": "97.89", "left_X": 4670, "top_y": 1726, "width": 567, "heigth": 1665}, {"label": "person", "confidence": "96.02", "left_X": 1324, "top_y": 1627, "width": 339, "heigth": 1180}, {"label": "person", "confidence": "95.80", "left_X": 1914, "top_y": 1746, "width": 303, "heigth": 1226}, {"label": "person", "confidence": "94.38", "left_X": 2489, "top_y": 1674, "width": 499, "heigth": 1504}, {"label": "person", "confidence": "93.46", "left_X": 2948, "top_y": 1757, "width": 348, "heigth": 1356}, {"label": "person", "confidence": "92.38", "left_X": 3527, "top_y": 1855, "width": 448, "heigth": 1400}, {"label": "person", "confidence": "91.22", "left_X": 2169, "top_y": 1589, "width": 340, "heigth": 1443}, {"label": "person", "confidence": "86.37", "left_X": 1052, "top_y": 1554, "width": 275, "heigth": 965}, {"label": "person", "confidence": "84.95", "left_X": 3913, "top_y": 1813, "width": 381, "heigth": 1364}, {"label": "person", "confidence": "84.34", "left_X": 4223, "top_y": 1814, "width": 298, "heigth": 1122}, {"label": "person", "confidence": "76.23", "left_X": 3317, "top_y": 1755, "width": 313, "heigth": 1328}, {"label": "person", "confidence": "73.16", "left_X": 703, "top_y": 1507, "width": 378, "heigth": 1001}, {"label": "person", "confidence": "67.75", "left_X": 543, "top_y": 1640, "width": 214, "heigth": 752}, {"label": "person", "confidence": "64.49", "left_X": 1728, "top_y": 1539, "width": 266, "heigth": 1290}, {"label": "person", "confidence": "58.08", "left_X": 324, "top_y": 1628, "width": 200, "heigth": 529}, {"label": "person", "confidence": "47.70", "left_X": 5, "top_y": 1383, "width": 5190, "heigth": 1765}, {"label": "handbag", "confidence": "45.03", "left_X": 2907, "top_y": 2571, "width": 186, "heigth": 254}, {"label": "person", "confidence": "39.16", "left_X": 1588, "top_y": 1658, "width": 207, "heigth": 1043}, {"label": "bicycle", "confidence": "39.10", "left_X": 180, "top_y": 1834, "width": 162, "heigth": 270}, {"label": "tie", "confidence": "26.99", "left_X": 837, "top_y": 1655, "width": 103, "heigth": 214}]

```

## statistics
  
We run image detection against the Coco dataset of about 80 classes and its sizes is 328k images, The keywords that could be detected accurately are windmills, churches, lighthouses, person, bicycle, car, motorbike, aeroplane among others. The accuracy of the prediction varies between 90 - 97%
