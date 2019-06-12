#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <queue>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "DpDetector.hpp"

using namespace cv;
using namespace dpdetector;
using namespace dnn;
using namespace std;

DpDetector::~DpDetector()
{

}

Ptr<DpDetector> DpDetector::Create(const DpDetectorConfig& config)
{
    switch(config.type) 
    {
        case DPDETECTOR_RETINANET:
            return Ptr<RetinaNetDetector> (new RetinaNetDetector(config));
        default:
            CV_Error(-1, "Detector type not supported");
    }
}

//////////////////////////// RetinaNet ////////////////////////////////////////

static vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty()) {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector < String > layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}


RetinaNetDetector::RetinaNetDetector(const DpDetectorConfig& config)
{
    Log(LOG_INFO, "=== Instantiating RetinaNetDetector ===\n");

    netInputSize        = config.netInputSize;
    scoreThreshold      = config.scoreThreshold;
    maxDetections       = config.maxDetections;
    nmsThreshold        = config.nmsThreshold;

    Log(LOG_INFO, "Model: %s\n", config.model.c_str());
    Log(LOG_INFO, "Input size: %d x %d\n", config.netInputSize.width, config.netInputSize.height);
    Log(LOG_INFO, "Score threshold: %f\n", config.scoreThreshold);
    Log(LOG_INFO, "NMS threshold: %f\n", config.nmsThreshold);
    Log(LOG_INFO, "Max detections: %d\n", config.maxDetections);

    className = "RetinaNetDetector";

    featureShapes.empty();
    Size levelFeatureShape = Size(netInputSize.width / 8, netInputSize.height / 8);
    for (int i = 0; i <= 5; i++) {
        featureShapes.push_back(levelFeatureShape);
        levelFeatureShape.width = (int)ceil((float)levelFeatureShape.width / 2);
        levelFeatureShape.height = (int)ceil((float)levelFeatureShape.height / 2);
    }

    net = dnn::readNet(config.model, "", ""); 
//    net.setPreferableBackend(DNN_BACKEND_HALIDE);
    net.setPreferableBackend(dnn::DNN_BACKEND_DEFAULT);      /* Default to OPENCV for now */
//    net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    netInitialized = true;

    Log(LOG_DEBUG, "=== All layers ===\n");
    layerNames = net.getLayerNames();
    for (int i = 0; i < layerNames.size(); ++i) {
        Log(LOG_DEBUG, "%d, %s\n", i, layerNames[i].c_str());
    }

    Log(LOG_DEBUG, "=== Outputs ===\n");
    vector<String> outputNames = getOutputsNames(net);
    for (int i = 0; i < outputNames.size(); ++i) {
        Log(LOG_DEBUG, "%d, %s\n", i, outputNames[i].c_str());
    }

    Log(LOG_DEBUG, "=== Generate Anchors ===\n");
    vector<float> ratios;
    ratios.push_back(0.5);
    ratios.push_back(1);
    ratios.push_back(2);

    vector<float> scales;
    scales.push_back(pow(2.0, 0.0));
    scales.push_back(pow(2.0, 1.0 / 3.0));
    scales.push_back(pow(2.0, 2.0 / 3.0));

    // Compute anchors over all pyramid layers
    for (int i = 0; i < 5; ++i) {
        Log(LOG_DEBUG, "--- Generating base anchors for layer %d size %d ---\n", i, anchorSizes[i]);
        vector<float> anchors = GenerateAnchors(anchorSizes[i], ratios, scales);

        for (int j = 0; j < anchors.size() / 4; j++) {
            Log(LOG_DEBUG, "Final anchor[%d]: %f %f %f %f\n", j, anchors[4 * j], anchors[4 * j + 1], anchors[4 * j + 2], anchors[4 * j + 3]);
        }

        Log(LOG_DEBUG, "Number of base anchors = %d\n", anchors.size() / 4);
        Log(LOG_DEBUG, "Shifting anchors for layer %d: featureShape %dx%d, stride %d \n", i, featureShapes[i].width, featureShapes[i].height, featureStrides[i]);
        vector<float> shiftedAnchors = Shift(featureShapes[i], featureStrides[i], anchors);

        for (int j = 0; j < 16; j++) {
            Log(LOG_DEBUG, "Final shifedAnchor[%d]: %f %f %f %f\n", j, shiftedAnchors[4 * j], shiftedAnchors[4 * j + 1], shiftedAnchors[4 * j + 2], shiftedAnchors[4 * j + 3]);
        }

        Log(LOG_DEBUG, "Number of anchors generated %d\n", (int)shiftedAnchors.size() / 4);
        levelAnchors.push_back(shiftedAnchors);
    }

    Log(LOG_DEBUG, "=== Get Output LayerId ===\n");
    for (int i = 0; i < sizeof(regressionLayerNames)/sizeof(String); ++i) {
        regressionLayerIds.push_back(net.getLayerId(regressionLayerNames[i]));
        Log(LOG_DEBUG, "%s -> %d \n", regressionLayerNames[i].c_str(), regressionLayerIds[i]);
    }

    for (int i = 0; i < sizeof(classificationLayerNames)/sizeof(String); ++i) {
        classificationLayerIds.push_back(net.getLayerId(classificationLayerNames[i]));
        Log(LOG_DEBUG, "%s -> %d \n", classificationLayerNames[i].c_str(), classificationLayerIds[i]);
    }
}

RetinaNetDetector::~RetinaNetDetector()
{

}

int RetinaNetDetector::GetInputSize(Size* size)
{
    *size = netInputSize;
    return 0;
}

int RetinaNetDetector::ProcessFrame(Mat frame, vector<vector<float>>* bboxesOut, vector<int>* labelsOut, vector<float>* scoresOut) 
{ 
    Mat blob;

    Log(LOG_DEBUG, "Process frame\n");

    // Create a 4D blob from a frame.
    dnn::blobFromImage(frame, blob, 1, netInputSize, Scalar(123, 116, 103), false, false);
    //subtract(blob, Scalar(123, 116, 103), blob);

    net.setInput(blob);

    if (net.getLayer(0)->outputNameToIndex("im_info") != -1) {
        Mat imInfo = (Mat_<float>(1, 3) << netInputSize.height, netInputSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }

    Log(LOG_DEBUG, "net.forward()\n");
    net.forward();

    vector<vector<float>> regressionBboxes;     
    for (int i = 0; i < regressionLayerIds.size(); ++i) {
        std::vector<Mat> outputBlobs;
        vector<vector<float>> levelBboxes;
        net.getOutputBlobs(regressionLayerIds[i], outputBlobs);
        Log(LOG_DEBUG, "RegressionLayer %d dimensions: %d, %d, %d\n", i, outputBlobs[0].size[0], outputBlobs[0].size[1], outputBlobs[0].size[2]);
        RegressBoxes(outputBlobs[0], levelAnchors[i], levelBboxes);
        Log(LOG_DEBUG, "Number of regressed boxes %d \n", levelBboxes.size());
        ClipBoxes(levelBboxes);
        regressionBboxes.insert(regressionBboxes.end(), levelBboxes.begin(), levelBboxes.end());
    }
    Log(LOG_DEBUG, "Regression output: num boxes %d\n", regressionBboxes.size() / 4);

    vector<float> classification; 
    int numClasses = 0;

    for (int i = 0; i < classificationLayerIds.size(); ++i) {
        std::vector<Mat> outputBlobs;
        net.getOutputBlobs(classificationLayerIds[i], outputBlobs);
        Log(LOG_DEBUG, "ClassificationLayer %d dimensions: %d, %d, %d\n", i, outputBlobs[0].size[0], outputBlobs[0].size[1], outputBlobs[0].size[2]);
        if (outputBlobs[0].isContinuous()) {
            int size = outputBlobs[0].size[0] * outputBlobs[0].size[1] * outputBlobs[0].size[2];
            classification.insert(classification.end(), outputBlobs[0].ptr<float>(0), outputBlobs[0].ptr<float>(0) + size);
            if (numClasses == 0) {
                numClasses = outputBlobs[0].size[2];
            } else {
                assert(numClasses == outputBlobs[0].size[2]);
            }
        } else {
            Log(LOG_ERROR, "Not Continuous!!\n");
            assert(false);
        }
    }

    FilterDetections(regressionBboxes, classification, numClasses, bboxesOut, labelsOut, scoresOut);

    return DPDETECTOR_ERROR_NONE;
}

int RetinaNetDetector::ProcessFrameToLayer(Mat frame, String layerName, vector<Mat>* outs)
{ 
    return DPDETECTOR_ERROR_UNSUPPORTED; 
}

int RetinaNetDetector::RegressBoxes(const Mat& regression, const vector<float>& anchors, vector<vector<float>>& boxes)
{
    /* bbox_transform_inv */
    Log(LOG_DEBUG, "regression size %d anchors size %d\n", regression.size[1], anchors.size() / 4);
    assert(regression.size[1] == anchors.size() / 4);
    assert(regression.isContinuous());

    for (int k = 0; k < anchors.size() / 4; ++k) {
        float width = anchors[4 * k + 2]  - anchors[4 * k];
        float height = anchors[4 * k + 3] - anchors[4 * k + 1];

        float x1 = anchors[4 * k]    + (regression.at<float>(0, k, 0) * std[0] + mean[0]) * width; 
        float y1 = anchors[4 * k + 1] + (regression.at<float>(0, k, 1) * std[1] + mean[1]) * height; 
        float x2 = anchors[4 * k + 2] + (regression.at<float>(0, k, 2) * std[2] + mean[2]) * width;
        float y2 = anchors[4 * k + 3] + (regression.at<float>(0, k, 3) * std[3] + mean[3]) * height;

        vector<float> box;
        box.push_back(x1);
        box.push_back(y1);
        box.push_back(x2);
        box.push_back(y2);
        boxes.push_back(box);
    }
    return DPDETECTOR_ERROR_NONE;
}


int RetinaNetDetector::ClipBoxes(vector<vector<float>>& boxes)
{
    for (int i = 0; i < boxes.size(); ++i) {
        vector<float> box = (boxes)[i];
        box[0] = max((float)0, min(box[0], (float)netInputSize.width));
        box[1] = max((float)0, min(box[1], (float)netInputSize.height));
        box[2] = max((float)0, min(box[2], (float)netInputSize.width));
        box[3] = max((float)0, min(box[3], (float)netInputSize.height));
        (boxes)[i] = box;
    }
}

// Perform per class filtering
int RetinaNetDetector::FilterDetections(const vector<vector<float>>& regressionBboxes, const vector<float>& classification, const int numClasses, vector<vector<float>>* bboxesOut, vector<int>* labelsOut, vector<float>* scoresOut)
{
    vector<float> filteredScores;
    vector<int> labels;
    vector<vector<float>> filteredBoxes;

    int numBoxes = regressionBboxes.size();
    assert((classification.size() / numClasses) == numBoxes);        // Sanity check

    Log(LOG_DEBUG, "FilterDetections: numClasses %d, numBoxes %d\n", numClasses, numBoxes);

    // For each class, perform thresholding and nms
    for (int i = 0; i < numClasses; ++i) { 
        vector<float> classFilteredScores;
        vector<vector<float>> classFilteredBoxes;
        for (int j = 0; j < numBoxes; ++j) {
            float score = classification[j * numClasses + i];
            if (score > scoreThreshold) {
                classFilteredScores.push_back(score);
                classFilteredBoxes.push_back(regressionBboxes[j]);
            }
        }
        Log(LOG_DEBUG,"Class[%d]: size %d \n", i, classFilteredScores.size());
        vector<int> classLabels(classFilteredScores.size(), i);

        if (classFilteredBoxes.size() > 0) {
            /* Perform NMS on filtered boxes */
            NMS(classFilteredBoxes, classFilteredScores, classLabels);
            Log(LOG_DEBUG,"Class[%d]: After NMS size %d \n", i, classFilteredScores.size());
            filteredScores.insert(filteredScores.end(), classFilteredScores.begin(), classFilteredScores.end());
            filteredBoxes.insert(filteredBoxes.end(), classFilteredBoxes.begin(), classFilteredBoxes.end());
            labels.insert(labels.end(), classLabels.begin(), classLabels.end());
        }
    }


    // Pick top k scores
    vector<int> indices;
    int k = min(maxDetections, (int)filteredScores.size());
    Log(LOG_DEBUG, "Pick top %d scores \n", k);
    priority_queue<pair<float, int>> q;
    for (int i = 0; i < filteredScores.size(); ++i) {
        q.push(pair<float, int>(filteredScores[i], i));
    }
    for (int i = 0; i < k; ++i) {
        int index = q.top().second;
        indices.push_back(index);
        q.pop();
    }

    (*bboxesOut).empty();
    (*labelsOut).empty();
    (*scoresOut).empty();

    for (int i = 0; i < indices.size(); ++i) {
        (*bboxesOut).push_back(filteredBoxes[i]);
        (*labelsOut).push_back(labels[i]);
        (*scoresOut).push_back(filteredScores[i]);
    }

    return DPDETECTOR_ERROR_NONE;
}
