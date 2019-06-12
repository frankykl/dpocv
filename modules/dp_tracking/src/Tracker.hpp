#ifndef __TRACKER_HPP__
#define __TRACKER_HPP__

#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "MOSSETracker.hpp"
#include "CSRTracker.hpp"
#include "ORF.h"

using namespace cv;
using namespace std;

namespace cv {
namespace dp_tracking {

/*
 * Detection info.
 */
typedef struct {
	Rect bbox;     // bounding box
	float conf;    // confidence
	vector<int> trackerId;  // tracker id
} DetectionInfo;

/*
 * Tracker state.
 */
typedef enum {
	IDLE,
	ACTIVE,
	OCCLUDED,
} TrackerState;

typedef struct {
	MatND hist;
} KeyPointStat;

/*
 * Tracker class.
 * 
 * A tracker is responsible to track a single object. To perform tracking task, a tracker
 * needs to have the following capability.
 * 
 * - Maintain a state-space model for the object dynamics, a by-product will be the motion
 *   trajectory of the object. The state-space model returns the following information to
 *   upper layer: object position, object bounding box (and mask), and object motion
 *   trajectory, and the probability of the object location. The state-space model is
 *   achieved by a Kalman filter or a particle filter.
 * 
 * - An observation model is required to build the state-space model. The observation model
 *   requires to maintain an effective appearance model to identify the object. The observation
 *   model is achieved by an online learning mechanism, using effective image features, such
 *   as HOG+SVM, or HOG+ORF.
 */
class CTracker {
public:
	static const int _historySize = 15;
	static constexpr double _trjWeightInit = 0.3;

	// Constructor/destructor
	~CTracker();
	CTracker(int id);

	// Get/Set
	int GetId() {return _id;}
	TrackerState GetState() {return _state;}
	Rect GetBoundingBox() {return _boundingBox;}
	vector<Rect>& GetBBoxHistory() {return _bboxHistory;}

	// Operations
	void Initialize(Mat& frame, Rect& boundingBox);
	void Stop();
	void Track(Mat& frame, vector<DetectionInfo>& detections);
	void Recover(Mat& frame, vector<DetectionInfo>& detections);
	void SetObject(Mat& frame, Rect bbox, double confTracker);

protected:

private:
	int _id;               // tracker id, used to identify tracked object
	TrackerState _state;   // tracker state
	Size _imageSize;       // input image size
	Rect _boundingBox;     // bounding box of tracked object
	Mat _mask;             // mask of tracked object
	Mat _patch;            // image patch of tracked object
	Mat _curFrame;
	double _trjWeight;
    int _lostCount;
    int _missCount;
    Ptr<DPTrackerCSRT> _pCSRTTracker;
	vector<Rect> _bboxHistory;
	vector<Mat> _patchHistory;
    vector<MatND> _featureHistory; // patch feature history
    CFeatureSpace* _pFeatureSpace;

    // tracking confidence
    int _trackingCount;
    double _confMean;
    double _confStd;

    // ORF
    CORFParameters _orfParams;
    OnlineRF* _pORF;
	SampleSet _trainingSet;

    // private functions
	double ComputeTrajectoryCost(Rect& target);
	int CheckMatch(double thr);
	int CheckRecover(Mat& frame, vector<DetectionInfo>& detections);
	void UpdateConfidence(double conf);
	void DisplayHistory();
	void TrainORF();
	bool DetectOcclusionORF(Mat& patch);
	bool DetectOcclusion(Mat& patch, double conf);
	bool DetectMistrack(Rect bbox, vector<DetectionInfo>& detections);
	void GenerateNegativeSample();
}; // class CTracker

} // namespace dp_tracking
} // namespace cv

#endif /* __TRACKER_HPP__ */
