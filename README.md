# PoseNet_TFLite_Runtime
Python library that uses a ready-made TFLite model and spits poses

The library was based on the [TensorFlow pose estimation example for Mobile & IoT](https://www.tensorflow.org/lite/models/pose_estimation/overview).  
The TFLite model that is automatically downloaded by the library can be found [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite)



## Installation:

### 1 Cloning
```
$ git clone https://github.com/ricardodeazambuja/PoseNet_TFLite_Runtime.git
$ cd YAMSPy
$ sudo pip3 install .
```

### 2 pip
```
$ sudo pip3 install git+git://github.com/ricardodeazambuja/PoseNet_TFLite_Runtime --upgrade
```

### TFLite Runtime
Check your python version:
```
$ python --version
```
Then [select the proper tflite_runtime version](https://www.tensorflow.org/lite/guide/python). E.g. `Python 3.6.9`:
```
$ pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp36-cp36m-linux_x86_64.whl --upgrade
```

### Usage:
It expects a RGB image measuring 257x257 pixels, therefore array shape will be (257,257,3).
```
from tflite_posenet import Pose
pose = Pose()
output_dict = pose.calc(img) # img => PIL or numpy array

min_score = 0.8
img = pose.draw_pose(output_dict, img, 
                        threshold=min_score, 
                        marker_color='green', 
                        color='yellow', 
                        marker_size=10, 
                        thickness=2)
```

