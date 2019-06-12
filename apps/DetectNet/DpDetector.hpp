/*
 * Copyright Deep Photon Inc.
 */

#ifndef __DPDETECTOR_HPP__
#define __DPDETECTOR_HPP__

#ifdef __cplusplus

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <stdarg.h>
#include <stdio.h>

using namespace cv;
using namespace std;

// DeepPhoton
namespace dpdetector  {

#define DPDETECTOR_DEFAULT_INPUT_WIDTH             640
#define DPDETECTOR_DEFAULT_INPUT_HEIGHT            640
#define DPDETECTOR_DEFAULT_SCORE_THRESHOLD         0.05
#define DPDETECTOR_DEFAULT_MAX_DETECTIONS          300
#define DPDETECTOR_DEFAULT_NMS_THRESHOLD           0.5

typedef enum {
    LOG_DEBUG = 1,
    LOG_INFO = 2,
    LOG_WARNING = 3,
    LOG_ERROR = 4
} DpDetectorLogLevel;

typedef enum {
    DPDETECTOR_RETINANET = 0,
    DPDETECTOR_MTCNN = 1
} DpDetectorType;

typedef enum {
    DPDETECTOR_ERROR_NONE = 0,
    DPDETECTOR_ERROR_UNKNOWN = -1,
    DPDETECTOR_ERROR_UNSUPPORTED = -2
} DpDetectorErr;

typedef enum {
    QUANT_TYPE_FLOAT32 = 0,
    QUANT_TYPE_8B = 0,
    QUANT_TYPE_4B = 0,
    QUANT_TYPE_2B = 0,
    QUANT_TYPE_1B = 0
} DpDetectorQuantType;

typedef struct {
    DpDetectorType                      type;    
    String                              model;
    map<String, DpDetectorQuantType>    layersQuant;
    bool                                useMotionMask;
    Size                                netInputSize;
    float                               scoreThreshold;
    float                               nmsThreshold;
    int                                 maxDetections;
} DpDetectorConfig;

class CV_EXPORTS DpDetector
{
public: 
    static Ptr<DpDetector> Create(const DpDetectorConfig& config);

    virtual ~DpDetector();
    virtual int DetectorSummary() { return DPDETECTOR_ERROR_UNKNOWN; }
    virtual int GetInputSize(Size* size) { return DPDETECTOR_ERROR_UNKNOWN; }
    virtual int ProcessFrame(Mat frame, vector<vector<float>>* bboxes, vector<int>* labels, vector<float>* scores) 
    { 
        return DPDETECTOR_ERROR_UNSUPPORTED; 
    }

    virtual int ProcessFrameToLayer(Mat frame, String layerName, vector<Mat>* outs)
    {
        return DPDETECTOR_ERROR_UNSUPPORTED;
    }

    int SetLogLevel(DpDetectorLogLevel level)
    {
        printf("Set log level to %u \n", level);
        logLevel = level;
        return DPDETECTOR_ERROR_NONE;
    }

    void Log(DpDetectorLogLevel level, const char *pFormat, ...)
    {
        if (level >= logLevel) {
            va_list args;

            va_start(args, pFormat);
            vprintf(pFormat, args);
            va_end(args);
        }
    }

    int GetMaxDetectionsPerFrame() {
        return maxDetections;
    }

    int NMS(vector<vector<float>>& bboxes, vector<float>& confidence, vector<int>& labels) {
        for (int i = 0; i < bboxes.size()-1; i++) {
            float s1 = (bboxes[i][2] - bboxes[i][0] + 1) *(bboxes[i][3] - bboxes[i][1] + 1);
            for (int j = i + 1; j < bboxes.size(); j++)
            {
                float s2 = (bboxes[j][2] - bboxes[j][0] + 1) *(bboxes[j][3] - bboxes[j][1] + 1);

                float x1 = max(bboxes[i][0], bboxes[j][0]);
                float y1 = max(bboxes[i][1], bboxes[j][1]);
                float x2 = min(bboxes[i][2], bboxes[j][2]);
                float y2 = min(bboxes[i][3], bboxes[j][3]);

                float width = x2 - x1;
                float height = y2 - y1;
                if (width > 0 && height > 0) {
                    float IOU = width * height / (s1 + s2 - width * height);
                    if (IOU > nmsThreshold) {
                        if (confidence[i] >= confidence[j]) {
                            bboxes.erase(bboxes.begin() + j);
                            confidence.erase(confidence.begin() + j);
                            labels.erase(labels.begin() + j);
                            j--;
                        } else {
                            bboxes.erase(bboxes.begin() + i);
                            confidence.erase(confidence.begin() + i);
                            labels.erase(labels.begin() + i);
                            i--;
                            break;
                        }
                    }
                }
            }
        }
    }


protected:
    bool                netInitialized = false;        
    // Caller needs to specify dimension (see https://github.com/opencv/opencv/issues/10210)
    Size                netInputSize = Size(DPDETECTOR_DEFAULT_INPUT_WIDTH, DPDETECTOR_DEFAULT_INPUT_HEIGHT); 
    int                 maxDetections = DPDETECTOR_DEFAULT_MAX_DETECTIONS;
    float               scoreThreshold = DPDETECTOR_DEFAULT_SCORE_THRESHOLD;
    float               nmsThreshold = DPDETECTOR_DEFAULT_NMS_THRESHOLD;
    String              className;
    vector<String>      layerNames;

     vector<float> GenerateAnchors(int baseSize, vector<float> ratios, vector<float> scales)
     {
         vector<float>anchors;
         for (int i = 0; i < ratios.size(); ++i) {
             for (int j = 0; j < scales.size(); ++j) {
                 float scaledBaseSize = baseSize * scales[j];
                 float area = scaledBaseSize * scaledBaseSize; 
                 float ws = sqrt(area / ratios[i]);
                 float hs = ws * ratios[i];
                 Log(LOG_DEBUG, "Ratio = %lf / %lf\n", ws, hs);

                 anchors.push_back(-ws * 0.5);
                 anchors.push_back(-hs * 0.5);
                 anchors.push_back(ws * 0.5);
                 anchors.push_back(hs * 0.5);
             }
         }
         return anchors;
     }

     vector<float> Shift(Size featureShape, float stride, vector<float> anchors)
     {
         vector<float> shiftedAnchors;
         vector<float> shiftX, shiftY;     

         for (int i = 0; i < featureShape.width; ++i)   shiftX.push_back(((float)i + 0.5) * stride);
         for (int i = 0; i < featureShape.height; ++i)  shiftY.push_back(((float)i + 0.5) * stride);

         for (int i = 0; i < shiftY.size(); ++i) {
             for (int j = 0; j < shiftX.size(); ++j) {
                 //Log(LOG_DEBUG, "Pos %f, %f\n", shiftX[j], shiftY[i]);
                 for (int k = 0; k < anchors.size() / 4; ++k) {
                     shiftedAnchors.push_back(anchors[4 * k] + shiftX[j]);
                     shiftedAnchors.push_back(anchors[4 * k + 1] + shiftY[i]);
                     shiftedAnchors.push_back(anchors[4 * k + 2] + shiftX[j]);
                     shiftedAnchors.push_back(anchors[4 * k + 3] + shiftY[i]);
                 }
             }
         }
         return shiftedAnchors;
     }



private:
    DpDetectorLogLevel  logLevel = LOG_INFO;
    int                 regressionOutLayerId[5];
    int                 classificationOutLayerId[5];

};

class CV_EXPORTS RetinaNetDetector : public DpDetector
{
public:
    RetinaNetDetector(const DpDetectorConfig& config);
    ~RetinaNetDetector();
    int GetInputSize(Size* size) CV_OVERRIDE;
    int ProcessFrame(Mat frame, vector<vector<float>>* bboxesOut, vector<int>* labelsOut, vector<float>* scoresOut) CV_OVERRIDE;
    int ProcessFrameToLayer(Mat frame, String layerName, vector<Mat>* outs) CV_OVERRIDE;

private:
    int RegressBoxes(const Mat& regression, const vector<float>& anchors, vector<vector<float>>& boxes);  /* Applies delta to default anchor boxes */
    int ClipBoxes(vector<vector<float>>& boxes);
    int FilterDetections(const vector<vector<float>>& bboxes, const vector<float>& classificaton, const int numClasses, vector<vector<float>>* bboxesOut, vector<int>* labelsOut, vector<float>* scoresOut);

private:
    dnn::Net net;
    vector<int> featureStrides{8, 16, 32, 64, 128};
    vector<Size> featureShapes;
    vector<int> anchorSizes{32, 64, 128, 256, 512};
    vector<vector<float>> levelAnchors;
    float mean[4] = {0, 0, 0, 0};
    float std[4] = {0.2, 0.2, 0.2, 0.2};
    const String regressionLayerNames[5] = { 
        "regression_submodel/pyramid_regression_reshape/Reshape",               /* P3 (1 x 64 x 64 x 36) */
        "regression_submodel_1/pyramid_regression_reshape/Reshape",             /* P4 (1 x 32 x 32 x 36) */
        "regression_submodel_2/pyramid_regression_reshape/Reshape",             /* P5 (1 x 16 x 16 x 36) */
        "regression_submodel_3/pyramid_regression_reshape/Reshape",             /* P6 (1 x 8 x 8 x 36) */
        "regression_submodel_4/pyramid_regression_reshape/Reshape"              /* P7 (1 x 4 x 4 x 36) */
    };
    const String classificationLayerNames[5] = { 
        "classification_submodel/pyramid_classification_sigmoid/Sigmoid",       /* P3 (1 x 36864 x num_classes) */
        "classification_submodel_1/pyramid_classification_sigmoid/Sigmoid",     /* P4 (1 x 9216 x num_classes) */
        "classification_submodel_2/pyramid_classification_sigmoid/Sigmoid",     /* P5 (1 x 2304 x num_classes) */
        "classification_submodel_3/pyramid_classification_sigmoid/Sigmoid",     /* P6 (1 x 576 x num_classes) */
        "classification_submodel_4/pyramid_classification_sigmoid/Sigmoid"      /* P7 (1 x 144 x num_classes) */
    };

    vector<int> regressionLayerIds;
    vector<int> classificationLayerIds;
};

class CV_EXPORTS MTCNNDetector : public DpDetector
{
public:
    MTCNNDetector(const DpDetectorConfig& config);
    ~MTCNNDetector();
    int GetInputSize(Size* size) CV_OVERRIDE;
    int ProcessFrame(Mat frame, vector<vector<float>>* bboxesOut, vector<int>* labelsOut, vector<float>* scoresOut) CV_OVERRIDE;
    int ProcessFrameToLayer(Mat frame, String layerName, vector<Mat>* outs) CV_OVERRIDE;

private:
    dnn::Net net;
    const String bboxRegressLayerNames[1] = { 
        "bbox_regress/Reshape",     
    };
    const String classifierLayerNames[1] = { 
        "classifier/Reshape",
    };

    vector<int> bboxRegressLayerIds;
    vector<int> classifierLayerIds;
};

} // namespace dpdetector

#endif // #ifdef __cplusplus
#endif // #ifndef __DPDETECTOR_HPP__
