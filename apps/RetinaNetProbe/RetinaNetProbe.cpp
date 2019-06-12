#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
    "{ help  h     | | Print help message. }"
    "{ device      |  0 | camera device number. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ model m     | | Path to a binary file of model contains trained weights. "
    "It could be a file with extensions .caffemodel (Caffe), "
    ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet).}"
    "{ config c    | | Path to a text file of model contains network configuration. "
    "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet).}"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
    "{ mean        | | Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces. }"
    "{ scale       |  1 | Preprocess input image by multiplying on a scale factor. }"
    "{ width       | -1 | Preprocess input image by resizing to a specific width. }"
    "{ height      | -1 | Preprocess input image by resizing to a specific height. }"
    "{ rgb         |    | Indicate that model works with RGB input images instead BGR ones. }"
    "{ thr         | .5 | Confidence threshold. }"
    "{ nms         | .4 | Non-maximum suppression threshold. }"
    "{ layer       | | Name of the layer to be probed. }"
    "{ backend     |  0 | Choose one of computation backends: "
    "0: automatically (by default), "
    "1: Halide language (http://halide-lang.org/), "
    "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
    "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
    "0: CPU target (by default), "
    "1: OpenCL, "
    "2: OpenCL fp16 (half-float precision), "
    "3: VPU }";

using namespace std;
using namespace cv;
using namespace dnn;

float confThreshold, nmsThreshold;
std::vector<std::string> classes;

static void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
            Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

static void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector < Rect > boxes;
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
            {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() == 1);
        float* data = (float*) outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7) {
            float confidence = data[i + 2];
            if (confidence > confThreshold) {
                int left = (int) data[i + 3];
                int top = (int) data[i + 4];
                int right = (int) data[i + 5];
                int bottom = (int) data[i + 6];
                int width = right - left + 1;
                int height = bottom - top + 1;
                classIds.push_back((int) (data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    } else if (outLayerType == "DetectionOutput") {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() == 1);
        float* data = (float*) outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7) {
            float confidence = data[i + 2];
            if (confidence > confThreshold) {
                int left = (int) (data[i + 3] * frame.cols);
                int top = (int) (data[i + 4] * frame.rows);
                int right = (int) (data[i + 5] * frame.cols);
                int bottom = (int) (data[i + 6] * frame.rows);
                int width = right - left + 1;
                int height = bottom - top + 1;
                classIds.push_back((int) (data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    } else if (outLayerType == "Region") {
        for (size_t i = 0; i < outs.size(); ++i) {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*) outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold) {
                    int centerX = (int) (data[0] * frame.cols);
                    int centerY = (int) (data[1] * frame.rows);
                    int width = (int) (data[2] * frame.cols);
                    int height = (int) (data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float) confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    } else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width,
                box.y + box.height, frame);
    }
}

static std::vector<String> getOutputsNames(const Net& net)
{
    static std::vector<String> names;
    if (names.empty()) {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector < String > layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    confThreshold = parser.get<float>("thr");
    nmsThreshold = parser.get<float>("nms");
    float scale = parser.get<float>("scale");
    Scalar mean = parser.get < Scalar > ("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    String layerName = parser.get<String>("layer");

    // Open file with classes names.
    if (parser.has("classes")) {
        std::string file = parser.get < String > ("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line)) {
            classes.push_back(line);
        }
    }

    // Load a model.
    CV_Assert(parser.has("model"));
    printf("Loading model...\n");
    Net net = readNet(parser.get<String>("model"), parser.get<String>("config"),
            parser.get < String > ("framework")); 
    net.setPreferableBackend(parser.get<int>("backend"));
    net.setPreferableTarget(parser.get<int>("target"));

    // Get layer
    printf("Getting layers...\n");
    vector < String > layerNames = net.getLayerNames();
    for (int i = 0; i < layerNames.size(); i++) {
        cout << i << ", " << layerNames[i] << endl;
    }
    printf("Get layerId for %s\n", layerName.c_str());
    int layerId = net.getLayerId(layerName);
    cout << "layerId = " << layerId << endl;
    Ptr<Layer> layer = net.getLayer(layerId);

    // Create a window
    printf("Create output window\n");
    static const std::string kWinName = "Source";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(parser.get<int>("device"));

    printf("Start processing frames...\n");
    // Process frames.
    Mat frame, blob;
    int channel = 0, maxChannel = 0;
    while (1) {
        char ch = waitKey(100);
        if (ch == 'q') {
            break;
        } else if (ch == 'a') {
            if (channel < maxChannel) {
                channel++;
                cout << "channel = " << channel << endl;
            }
        } else if (ch == 'd') {
            if (channel > 0) {
                channel--;
                cout << "channel = " << channel << endl;
            }
        }

        // Read input
        cap >> frame;
        if (frame.empty())
            break;

        // Create a 4D blob from a frame.
        Size inpSize(inpWidth > 0 ? inpWidth : frame.cols, inpHeight > 0 ? inpHeight : frame.rows);
        blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);

        // Run network to the layer
        net.setInput(blob);
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1) {
            resize(frame, frame, inpSize);
            Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
            net.setInput(imInfo, "im_info");
        }
        std::vector<Mat> outs;
        //net.forward(outs, getOutputsNames(net));
        net.forward(outs, layerName);

        // Get layer output
        std::vector<Mat> outputBlobs;
        net.getOutputBlobs(layerId, outputBlobs);
        //cout << "blobs = " << outputBlobs.size() << ", " << outputBlobs[0].dims
        //        << outputBlobs[0].size[0] << ", " << outputBlobs[0].size[1] << ", "
        //        << outputBlobs[0].size[2] << ", " << outputBlobs[0].size[3] << endl;
        maxChannel = outputBlobs[0].size[1] - 1;

        // Display layer output by channel
        Mat slice(outputBlobs[0].size[3], outputBlobs[0].size[2], CV_32F,
                outputBlobs[0].ptr<float>(0, channel));
        normalize(slice, slice, 1, 0, NORM_L2);
        slice = slice * 255;
        resize(slice, slice, frame.size());
        imshow("map", slice);

        //postprocess(frame, outs, net);

        imshow(kWinName, frame);
    }
    return 0;
}

