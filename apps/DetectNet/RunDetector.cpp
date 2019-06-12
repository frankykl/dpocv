#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "DpDetector.hpp"

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dpdetector;

const char* keys =
    "{ help  h     | | Print help message. }"
    "{ device      |  0 | camera device number. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ model m     | | Path to a binary file of model contains trained weights. "
    "It could be a file with extensions .caffemodel (Caffe), "
    ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet).}"
    "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
    "{ width       | | Preprocess input image by resizing to a specific width. }"
    "{ height      | | Preprocess input image by resizing to a specific height. }";

std::vector<std::string> classes;

#define MAX_NUM_CLASS_COLORS 20

int main(int argc, char** argv)
{
    Scalar classColorMap[MAX_NUM_CLASS_COLORS] = {
        Scalar(255, 255, 255),  /* White */
        Scalar(255, 0, 0),      /* Red */
        Scalar(0, 255, 0),      /* Lime */
        Scalar(0,0, 255),       /* Blue */
        Scalar(255,255, 0),     /* Yellow */
        Scalar(0, 255, 255),    /* Cyan */
        Scalar(255, 0, 255),    /* Magenta */
        Scalar(192, 192, 192),  /* Silver */
        Scalar(128, 128, 128),   /* Gray */
        Scalar(128, 0, 0),      /* Maroon */
        Scalar(128, 128, 0),    /* Olive */
        Scalar(0, 128, 0),      /* Green */
        Scalar(128, 0, 128),    /* Purple */
        Scalar(0, 128, 128),    /* Teal */
        Scalar(0, 0, 128),      /* Navy */
        Scalar(184, 134, 11),   /* Dark golden rod */
        Scalar(255, 69, 0),     /* Orange red */
        Scalar(255, 192, 203),  /* Pink */
        Scalar(245, 245, 220),  /* Beige */
        Scalar(139, 69, 19)     /* Saddle brown */         
    };

    CommandLineParser parser(argc, argv, keys);
    parser.about("DeepPhoton Detector.");
    if (argc == 1 || parser.has("help")) {
        parser.printMessage();
        return 0;
    }

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

    DpDetectorConfig config;
    Size inpSize(DPDETECTOR_DEFAULT_INPUT_WIDTH, DPDETECTOR_DEFAULT_INPUT_HEIGHT);

    if (parser.has("width")) {
        inpSize.width = parser.get<int>("width");
    }

    if (parser.has("height")) {
        inpSize.height = parser.get<int>("height");
    }

    config.type = DPDETECTOR_RETINANET;
    config.model = parser.get < String > ("model");
    config.netInputSize = inpSize;
    config.nmsThreshold = DPDETECTOR_DEFAULT_NMS_THRESHOLD;
    config.scoreThreshold = 0.5; //DPDETECTOR_DEFAULT_SCORE_THRESHOLD;
    config.maxDetections = DPDETECTOR_DEFAULT_MAX_DETECTIONS;

    Ptr<DpDetector> pDetector = DpDetector::Create(config);
    if (pDetector == NULL) { 
        printf("Fail to create DpDetector \n");
        exit(-1);
    }

    pDetector->SetLogLevel(LOG_INFO);

    // Create a window
    printf("Create source window\n");
    static const std::string srcWinName = "Source";
    namedWindow(srcWinName, WINDOW_NORMAL);

    printf("Create annotated window\n");
    static const std::string annotatedWinName = "Annotated";
    namedWindow(annotatedWinName, WINDOW_NORMAL);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(parser.get<int>("device"));


    printf("Start processing frames...\n");
    // Process frames.
    Mat frame;
    Mat resizedFrame;
    int channel = 0, maxChannel = 0;
    while (1) {
        char ch = waitKey(100);
        if (ch == 'q') {
            break;
        }

        // Read input
        cap >> frame;
        if (frame.empty()) {
            printf("End of video/image\n");
            break;
        }

        int maxDetections = pDetector->GetMaxDetectionsPerFrame();

        // Run network to the layer
        std::vector<vector<float>> bboxes;
        bboxes.reserve(maxDetections);

        std::vector<int> labels;
        labels.reserve(maxDetections);

        std::vector<float> scores;
        scores.reserve(maxDetections);

        auto begin = std::chrono::high_resolution_clock::now();
        pDetector->ProcessFrame(frame, &bboxes, &labels, &scores);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - begin;
        printf("Frame processing time %lf \n", elapsed.count());

        imshow(srcWinName, frame);

        // For debugging
        resize(frame, resizedFrame, inpSize);
        for (int i = 0; i < bboxes.size(); i++) {
            Point pt1(bboxes[i][0], bboxes[i][1]);
            Point pt2(bboxes[i][2], bboxes[i][3]);
            rectangle(resizedFrame, pt1, pt2, classColorMap[labels[i] % MAX_NUM_CLASS_COLORS]);
//            printf("box[%f %f %f %f] label %d score %f\n", bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], labels[i], scores[i]);
        }
        imshow(annotatedWinName, resizedFrame);

    }
    waitKey(0);

    return 0;
}


