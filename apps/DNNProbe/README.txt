Purpose
=============
- This is an OpenCV application to visualize layer outputs of a deep neural network.

Build
=============
- This project depends on OpenCV package.
- Apply patch 0001-Add-function-to-get-layer-output-blob.patch to opencv.
- Build and install opencv.
- Make a build folder and run cmake.
- Build DNNProbe.

Download network data
=============
- Get opencv_extra
cd .../algo_core/third_party
git clone https://github.com/opencv/opencv_extra.git


Run
=============
cd .../algo_core/third_party/opencv_extra/testdata/dnn
../../../../dpocv/apps/DNNProbe/build/DNNProbe --config=yolov3.cfg --model=yolov3.weights --width=416 --height=416 --scale=0.00392 --input=/media/photon1/DataDrive/DataSets/MOT2015/train/PETS09-S2L1/video.mp4 --layer=conv_0

- User can select which layer's output to display by program option --layer=layer_name.
- User can control which channel of the layer's output feature map to display when program is running.
Press "a" to increase channel number, press "d" to decrease channel number.
