# YOLO Detector ROS Node

My implementation of a ROS Node that utilizes the Python API of the original Darknet YOLO detector.


# Useful Commands

## Kill all instances of this node still running on GPU
* Sometimes the node wont shutdown properly, and the GPU fills up with previous instances of the node.
```
nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9
```