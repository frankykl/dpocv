set(the_description "DeepPhoton BG Model module")
include_directories(${OPENCV_EXTRA_MODULES_PATH}/bioinspired/include)
ocv_define_module(dp_bgmodel opencv_bioinspired opencv_core opencv_imgproc opencv_video opencv_highgui WRAP java python)
