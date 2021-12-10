import numpy as np
import time
import cv2
import argparse
import sys
import os
import glob
import json
from pathlib import Path 

class ObjectDetector:
# Check if required modules are installed
    def modules_installed():
        if "numpy" and "cv2" and "glob" and "json" and "argparse" in sys.modules:
            return True
        else:
            print("You might need to check to confirm that Numpy and CV2 are installed")
            return False

    def file_exist(file_names_list: list) -> bool:
        if all(list(map(os.path.isfile,file_names_list))):
            return True
        else:
            print("Please check one of the Config Files does not exist")
            return False

    # if the script is run from the darknet folder, these defaults should be sane
    # if run from a different location or if different naming conventions are used, please update below
    def set_default_config():
        LABELS_FILE='data/coco.names'
        CONFIG_FILE='cfg/yolov4.cfg'
        WEIGHTS_FILE='yolov4.weights'
        CONFIDENCE_THRESHOLD=0.25

        file_names_list = [LABELS_FILE,CONFIG_FILE,WEIGHTS_FILE]
        
        # Check if the provided file paths exists
        if ObjectDetector.file_exist(file_names_list=file_names_list):
            return LABELS_FILE,CONFIG_FILE,WEIGHTS_FILE
    
    # load all files matching ext  from im_dir
    def load_image_files(dir=""):
        # defaults to "current" dir
        imdir = dir
        # various extensions of files that can be fetched
        ext = ['png', 'jpg', 'gif','jpeg']
        files = []
        [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
        return files

    def save_to_json(file_name,data):
        full_file_path = file_name + ".json"
        with open(full_file_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"{full_file_path} with bounding cordinates saved")

    def object_detection(image_path,label_path=None,config_path=None,weight_path=None,threshold=0.25):
        INPUT_FILE= image_path
        CONFIDENCE_THRESHOLD = threshold
        
        LABELS_FILE,_,_ = ObjectDetector.set_default_config() if label_path == None else label_path,None,None
        _,CONFIG_FILE,_ = ObjectDetector.set_default_config() if config_path == None else None,config_path,None
        _,_,WEIGHTS_FILE = ObjectDetector.set_default_config() if weight_path == None else None,None,weight_path

        LABELS = open(LABELS_FILE).read().strip().split("\n")

        np.random.seed(4)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")


        net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

        image = cv2.imread(INPUT_FILE)
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()


        #Maybe output some time Metrics?
        #print(" took {:.6f} seconds".format(end - start))


        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE_THRESHOLD:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
            CONFIDENCE_THRESHOLD)

        # how name objects were detected
        data = [{"detection_count": len(idxs), "file_name": image_path, "confidence_threshold": CONFIDENCE_THRESHOLD}]

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                data.append({"label":LABELS[classIDs[i]],  "confidence":"{:.2f}".format(confidences[i]*100), "left_X":x, "top_y":y,"width":w,"heigth":h})
        # save file ot image_path.json 
        # Maybe a flag to save results of a batch to same json?
        _file_name = Path(image_path).name
        # safe file to json 
        ObjectDetector.save_to_json(_file_name,data)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to Object Detection with Yolo4."
    )
    parser.add_argument(
        "--config_path", "-c", dest="config_path",
        default="cfg/yolov4.cfg", help="Path to yolov4.cfg"
    )
    parser.add_argument(
        "--weight_path", "-w", dest="weight_path",
        default="yolov4.weights", help="Path to yolov4.weights."
    )
    parser.add_argument(
        "--label_path", "-l", dest="label_path",
        default="cfg/coco.names", help="Path to coco.names."
    )

    parser.add_argument(
        "--image_path", "-i", dest="image_path",
        default=None, help="Path to Image file. Leaving Blank Searches the current directory"
    )

    parser.add_argument(
        "--threshold", "-t", dest="threshold",
        default=float(0.25), help="Detection Confidence Threshold to apply"
    )

    parser.add_argument(
        "--image-dir", "-d", dest="dir",
        default=os.getcwd(), help="Directory containing image file"
    )

    

    if ObjectDetector.modules_installed():
        args = parser.parse_args()
        if args.image_path is None:
           print(f"--image_path not provided, searching {args.dir} for image files...")
            image_files = ObjectDetector.load_image_files(args.dir)
            if len(image_files) <=0:
                print("No Image file(s) found")
            for i,image in enumerate(image_files,1):
                print(f"Running Object Detection on {i} of {len(image_files)} images")
                ObjectDetector.object_detection(
                        image,
                        args.label_path, 
                        args.config_path, 
                        args.weight_path,
                        float(args.threshold)
                      )
        else:
            ObjectDetector.object_detection(
                        args.image_path,
                        args.label_path, 
                        args.config_path, 
                        args.weight_path,
                        float(args.threshold)
                      )

                    
       