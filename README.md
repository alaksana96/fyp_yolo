# YOLO Detector ROS Node

## Node: yolo_detector 

My implementation of a ROS Node that utilizes the Python API of the original Darknet YOLO detector.

## Custom Messages

Open the ``src/fyp_yolo/msg`` folder to see the custom messages:

* **`BoundingBox`** Contains the individual detections.

    ```
    string Class
    float64 probability
    int64 xmin
    int64 ymin
    int64 xmax
    int64 ymax
    ```

* **`BoundingBoxes`** Contains the list of detections.

    ```
    Header header
    Header image_header
    BoundingBox[] bounding_boxes
    ```

## Topics

### Subscribed Topics

* **`/image_transport/compressed`** ([sensor_msg::CompressedImage])
  
    Subscribe to the compressed image topic from the Hololens in the JPEG format.

### Published Topics

* **`/yolo_detector/output/compressed`** ([sensor_msg::CompressedImage])

    Publish the same compressed image that the node subscribes too.

* **`/yolo_detector/output/detections`** ([fyp_yolo::BoundingBoxes])

    Publish the detections from the YOLO detector.


# Useful Commands

## Kill all instances of this node still running on GPU
* Sometimes the node wont shutdown properly, and the GPU fills up with previous instances of the node.
```
nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9
```