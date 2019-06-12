/*
 * Copyright Deep Photon.
 */

#include "precomp.hpp"
#include "opencv2/core/utility.hpp"
#include <limits>
#include <iostream>
#include <string>
#include <vector>

#include "MTTracker.hpp"

using namespace std;

/*
 * Convert an integer value to a string.
 */
static string ToString(int val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}

namespace cv {
namespace dp_tracking {

/*
 * Convert data type to string.
 */
static string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += ToString(chans);

  return r;
}

/*
 * Draw optical flow.
 */
void DrawOpticalFlow(const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
	for (int y = 0; y < cflowmap.rows; y += step) {
		for (int x = 0; x < cflowmap.cols; x += step) {
			const Point2f& fxy = flow.at < Point2f > (y, x);
			line(cflowmap, Point(x, y),
					Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
			circle(cflowmap, Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), 1,
					color, -1);
		}
	}
}

/*
 * Constructor.
 */
DPMTTrackerImpl::DPMTTrackerImpl(int numberTrackers, double sigmaGaussian) {
	// Default Parameter Values. Override with algorithm "set" method.
	_name = "DPTracker";
	_numTrackers = numberTrackers;
	_procFrames = 0;
	_sigmaGaussian = sigmaGaussian;

	for (int i = 0; i < numberTrackers; i++) {
		CTracker *tracker = new CTracker(i);
		_trackers.push_back(tracker);
		_freeTrackers.push_back(tracker);
	}
}

/*
 * Destructor.
 */
DPMTTrackerImpl::~DPMTTrackerImpl() {
}

/*
 * Update trackers.
 */
void DPMTTrackerImpl::UpdateTrackers()
{
    //printf("  trackers = %d, %d\n", _activeTrackers.size(), _freeTrackers.size());

    // check idle trackers
    for (int i = _activeTrackers.size() - 1; i >= 0; i--) {
        if (_activeTrackers[i]->GetState() == IDLE) {
            _freeTrackers.push_back(_activeTrackers[i]);
            _activeTrackers.erase(_activeTrackers.begin() + i);
        }
    }

#if 0
    // search untracked detections
    vector<Rect> untrackedBoxes;
    for (int i = 0; i < _boundingBoxes.size(); i++) {
        Rect r = _boundingBoxes[i];
        bool bTracked = false;
        for (int j = 0; j < _activeTrackers.size(); j++) {
            Rect r1 = _activeTrackers[j]->GetBoundingBox();
            double iou = (r & r1).area() / (r | r1).area();
            if (iou > 0.8) {
                bTracked = true;
                break;
            }
        }
        if (!bTracked) {
            untrackedBoxes.push_back(r);
        }
    }
    //printf("  Untracked = %d\n", untrackedBoxes.size());

	for (int i = 0; i < untrackedBoxes.size(); i++) {
        Rect r = untrackedBoxes[i];
        if (_freeTrackers.size() > 0) {
            CTracker *tracker = _freeTrackers.front();
            _freeTrackers.erase(_freeTrackers.begin());
            tracker->Initialize(_curFrame, r);
            _activeTrackers.push_back(tracker);
        } else
            break;
	}
#endif
}

/*
 * Run global optimization.
 */
void DPMTTrackerImpl::GlobalOptimization()
{
}

/*
 * Get the tracker state.
 */
int DPMTTrackerImpl::GetTrackerState(int id)
{
	if (id >= 0 && id < _trackers.size())
		return _trackers[id]->GetState();

	return -1;
}

/*
 * Get the object bounding box of a tracker.
 */
Rect DPMTTrackerImpl::GetTrackerBBox(int id)
{
	if (id >= 0 && id < _trackers.size())
		return _trackers[id]->GetBoundingBox();
	
	return Rect(0, 0, 0, 0);
}

/*
 * Start a tracker on an object.
 */
bool DPMTTrackerImpl::StartTracker(int trackerId, Mat& frame, Mat& boundingBox)
{
	// check tracker id range
	if (trackerId < 0 || trackerId >= _numTrackers)
		return false;

	if (boundingBox.rows < 1)
		return false;

	GaussianBlur(frame, _curFrame, Size(5, 5), _sigmaGaussian, _sigmaGaussian);

	float *p = boundingBox.ptr<float>(0);
	printf("StartTracker %d, ", trackerId);
	std::cout << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << std::endl;
	Rect rect(p[0], p[1], p[2] - p[0] + 1, p[3] - p[1] + 1);

	CTracker* tracker = _trackers[trackerId];
	TrackerState state = tracker->GetState();
	tracker->Initialize(frame, rect);
	if (state == IDLE)
        _activeTrackers.push_back(tracker);

	return true;
}

void DPMTTrackerImpl::StopTracker(int trackerId)
{
	// check tracker id range
	if (trackerId < 0 || trackerId >= _numTrackers)
		return;

	CTracker* tracker = _trackers[trackerId];
	tracker->Stop();
	UpdateTrackers();
}

/*
 * Process a new frame.
 * Arguments:
 *   frame[in]: new frame
 *   boundingBox[in]: bounding boxes of detected objects
 */
void DPMTTrackerImpl::ProcessFrame(Mat& frame, Mat& boundingBox)
{
	//printf("frame %d, %f\n", _procFrames, _sigmaGaussian);
	GaussianBlur(frame, _curFrame, Size(5, 5), _sigmaGaussian, _sigmaGaussian);

	// set detection
	SetDetection(boundingBox);

	// run trackers in active mode
	for (int i = 0; i < _activeTrackers.size(); i++) {
		if (_activeTrackers[i]->GetState() == ACTIVE)
			_activeTrackers[i]->Track(_curFrame, _detections);
	}

	// run re-detection
	for (int i = 0; i < _activeTrackers.size(); i++) {
		if (_activeTrackers[i]->GetState() == OCCLUDED) {
			_activeTrackers[i]->Recover(_curFrame, _detections);
		}
	}

	// global optimization
	GlobalOptimization();

	// update individual trackers
	UpdateTrackers();

	// update previous frame
	_curFrame.copyTo(_prevFrame);
	_procFrames++;
}

/*
 * Set detection result.
 */
void DPMTTrackerImpl::SetDetection(Mat& boundingBox)
{
	//std::cout << boundingBox.dims << " ## " << boundingBox.size() << type2str(boundingBox.type()) << std::endl;
	_detections.clear();

	// create display image
	Mat display = Mat::zeros(_curFrame.size(), _curFrame.type());

	// loop for detected objects
	//std::cout << "#objects = " << boundingBox.rows << std::endl;
	for (int i = 0; i < boundingBox.rows; i++) {
		DetectionInfo detection;

		// get detection bounding box
		float *p = boundingBox.ptr<float>(i);
		//std::cout << i << ", " << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << std::endl;
		detection.bbox.x = p[0];
		detection.bbox.y = p[1];
		detection.bbox.width = p[2] - p[0] + 1;
		detection.bbox.height = p[3] - p[1] + 1;
		detection.conf = p[4];
		_detections.push_back(detection);

#ifdef DEBUG_DISPLAY
		// display detected object
		//Rect rect(p[0], p[1], p[2] - p[0] + 1, p[3] - p[1] + 1);
		_curFrame(rect).copyTo(display(rect));
		// draw bounding-box
		Point p0 = Point(p[0], p[1]);
		Point p1 = Point(p[2], p[3]);
		rectangle(display, p0, p1, Scalar(255, 0, 0), 1);
#endif
	}
#ifdef DEBUG_DISPLAY
	imshow("Detection", display);
#endif
}

/*
 * Set detection and mask result.
 */
void DPMTTrackerImpl::SetDetectionAndMask(Mat& boundingBox, Mat& mask)
{
#if 0
	//std::cout << boundingBox.dims << " ## " << boundingBox.size() << type2str(boundingBox.type()) << std::endl;
	//std::cout << mask.dims << ", " << mask.size() << ", " << type2str(mask.type()) << std::endl;
	//std::cout << mask.elemSize() << std::endl;
	_detections.clear();
	_masks.clear();

	Mat mask1;
	int size[3] = {boundingBox.rows, 21, 21};
	mask1 = mask.reshape(1, 3, size);

	// create display image
	Mat display = Mat::zeros(_curFrame.size(), _curFrame.type());

	// loop for detected objects
	for (int i = 0; i < boundingBox.rows; i++) {
		// get detection bounding box
		double *p = boundingBox.ptr<double>(i);

		// copy object mask
		Mat mask2(21, 21, CV_32FC1);
		for (int y = 0; y < 21; y++) {
			for (int x = 0; x < 21; x++) {
				mask2.at<float>(y, x) = mask1.at<float>(i, y, x);
			}
		}

		// resize object mask to bounding-box
		Mat mask3(p[3] - p[1] + 1, p[2] - p[0] + 1, CV_32FC1);
		resize(mask2, mask3, mask3.size());

		// thresholding
		Mat mask4, mask5;
		threshold(mask3, mask4, 0.5, 1.0, 0);
		mask4.convertTo(mask5, CV_8UC1);

		// display masked object
		Rect rect(p[0], p[1], p[2] - p[0] + 1, p[3] - p[1] + 1);
		_curFrame(rect).copyTo(display(rect), mask5);

		// draw bounding-box
		Point p0 = Point(p[0], p[1]);
		Point p1 = Point(p[2], p[3]);
		rectangle(display, p0, p1, Scalar(255, 0, 0), 1);

		_boundingBoxes.push_back(rect);
		_masks.push_back(mask5);
	}
	imshow("Detection", display);
#endif
}

/*
 * Compute whole frame optical flow.
 */
void DPMTTrackerImpl::ComputeOpticalFlow()
{
	Mat prevGray, curGray;
	cvtColor(_prevFrame, prevGray, COLOR_RGB2GRAY);
	cvtColor(_curFrame, curGray, COLOR_RGB2GRAY);

	Mat flow, display = _prevFrame;
	calcOpticalFlowFarneback(prevGray, curGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	DrawOpticalFlow(flow, display, 10, Scalar(128, 0, 0));
	//imshow("flow", display);
}

/*
 * Function to create an multi-target tracker.
 */
Ptr<DPMTTracker> CreateDPMTTracker(int numberTrackers, double sigmaGaussian) {
	Ptr < DPMTTracker > tracker = makePtr<DPMTTrackerImpl>(numberTrackers, sigmaGaussian);
	return tracker;
}

} // namespace dp_tracking
} // namespace cv

