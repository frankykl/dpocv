#Overview
DetectNet features object detection and classifications based on RetinaNet feature pyramid architecture. In the initial version, it focus on generating bbox and classification results. 
It loads Tensorflow frozen model (.pb) and perform inference with OpenCV dnn module. Following features will be included:

*  Configurable quantization of layers (different bit depth per layer) 
*  Configurable backbone architecture (ResNet, MobileNet, etc)
*  Configurable FPN (Number of layers)

# Preparing Model
RunDetectNet is the application that reads in video or image files and pass the input to DetectNet. Before you can run the application, you need to generate a model file that can
be supported by OpenCV. Here is the procedure:

1. DeepPhoton's version of RetinaNet training code is in "mvpvas/algo_core/dptf/pvc". Note: PVC(Primary Visual Cortex) pvcnet is an ensemble of neural networks to mimic low level 
   visual functions of human. Several characteristics of this implementation:
     * It is implemented in Keras (DeepPhoton will use Keras as our standard training framework). We currently use Tensorflow backend, we may use other depending on the original source of model)
     * We use fixed network input (640x640 by default). The configuration is in mvpvas/algo_core/dptf/pvc/pvcnet/pvcnet/bin/config.py
     * Floating point backbone network is used by default. To train quantized backbone, training code need to specify --backbone qresnet50
     * The script for running the training code is in mvpvas/algo_core/dptf/pvc/pvcnet/script
  
2. The training in step 1 generates keras .h5 format, we have a keras_to_tf.py script which performs following function:
    * Freeze the graph
    * Optimize the graph
    * Remove layers not necessary for inference 
    * Note: Current keras_to_tf.py has some Retinanet specific trimming which may not work for other networks. We will need to develop a more generic graph transform and adaptation pipeline.
  
# Installing OpenVino 
On Intel platform, OpenVino is the latest SDK that includes Intel Inference Engine. OpenCV inference engine has very good performance. It uses Intel MKL in core compute functions
Before building opencv to use OpenVino, you need to install OpenVino on your Ubuntu (See https://software.intel.com/en-us/articles/OpenVINO-Install-Linux)

Note: I have also evaluated Halide on x86. It seems like the current performance is no better than pure C++ OpenCV implementation. We will not use it for now.

# Building OpenCV
The build_opencv_linux.sh already includes option to enable MKL (i.e. OpenVino/Intel Inference Engine), you set enable_mkl to true after OpenVino is installed.

# Building Running DetectNet
Once OpenCV is built, you can run build and DetectNet using following command:
* goto mvpvas/algo_core/dpocv/apps/DetectNet folder
* mkdir build
* cd build
* cmake ..
* make
* ./RunDetector --m=<.pb model file> --i=<input video or image file>

