#ifndef __MTTRACKER_HPP__
#define __MTTRACKER_HPP__

#include "opencv2/core.hpp"
#include <vector>

#include "Tracker.hpp"

using namespace cv;
using namespace std;

namespace cv {
namespace dp_tracking {

/*
 * Multi-target tracker class.
 * 
 * The purpose of DPMTTracker class is to track multiple objects simultaneously using information
 * from multiple single-object trackers.
 * 
 * - MTTracker receives a new image frame from the caller, with optional detected objects' info.
 * - MTTracker creates a new tracker when the detected object does not match any of the existing
 *   tracked objects.
 * - MTTracker deletes an existing tracker when its object disappears from the image.
 * - MTTracker receives info from individual trackers and optimizes globally the trajectories of
 *   all tracked objects.
 */
class DPMTTrackerImpl: public DPMTTracker {
public:
	static const int _maxNumberTrackers = 10;

	DPMTTrackerImpl(int numberTrackers, double sigmaGaussian);
	~DPMTTrackerImpl();

	int GetNumberTrackers() {return _trackers.size();}
	int GetTrackerState(int id);
	Rect GetTrackerBBox(int id);
	bool StartTracker(int trackerId, Mat& frame, Mat& boundingBox);
	void StopTracker(int trackerId);
	void ProcessFrame(Mat& frame, Mat& boundingBox);
	void GlobalOptimization();
	void UpdateTrackers();

protected:
	void SetDetection(Mat& boundingBox);
	void SetDetectionAndMask(Mat& boundingBox, Mat& mask);
	void ComputeOpticalFlow();

private:
	String _name;       // class name
	Mat _prevFrame;     // previous frame
	Mat _curFrame;      // current frame
	int _procFrames;    // number of processed frames
	int _numTrackers;   // number of trackers
	double _sigmaGaussian;
	vector<DetectionInfo> _detections;   // detected objects
	vector<Mat> _masks;            // masks of objects
	vector<CTracker*> _trackers;        // list of all single-target trackers
	vector<CTracker*> _freeTrackers;    // list of free trackers
	vector<CTracker*> _activeTrackers;  // list of active trackers
}; // class DPTrackerImpl


} // namespace dp_tracking
} // namespace cv

#endif /* __MTTRACKER_HPP__ */
