/*
 * Copyright Deep Photon.
 */

#include "precomp.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include <limits>
#include <iostream>
#include <string>
#include "Tracker.hpp"

using namespace std;

namespace cv {
namespace dp_tracking {

static Point GetRectCenter(Rect& rect)
{
	return Point((double)rect.x + (double)rect.width / 2, (double)rect.y + (double)rect.height / 2);
}

/*
 * Compute patch color histgram.
 */
static void ComputeHistogram(Mat& patch, MatND& hist)
{
	// histogram settings
	int hueBins = 90, satBins = 64;
	int histSize[] = {hueBins, satBins};
	float hueRanges[] = {0, 180};
	float satRanges[] = {0, 256};
	const float *ranges[] = {hueRanges, satRanges};
	int channels[] = {0, 1};

	Mat hsvPatch, scaledPatch;
	resize(patch, scaledPatch, Size(64, 128));
	Mat crop = scaledPatch(Rect(4,8,56,112));
	cvtColor(scaledPatch, hsvPatch, COLOR_BGR2HSV);
	calcHist(&hsvPatch, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
}

/*
 * Compute patch matching cost using NCC.
 */
static float ComputeMatchingCost(vector<Mat>& patchHistory, Mat& target)
{
	float cost = 0.0f;
	int count = patchHistory.size();
	for (int i = 0; i < count; i++) {
		Mat patch = patchHistory[i];
		Mat resizedTarget;
		resize(target, resizedTarget, patch.size());
		Mat result;
		result.create(1, 1, CV_32FC1);
		matchTemplate(resizedTarget, patch, result, cv::TM_CCORR_NORMED);

		double minVal; double maxVal; Point minLoc; Point maxLoc;
  		Point matchLoc;
  		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		cost += minVal;
  	}
	
	if (count > 0)
		cost /= count;

	return cost;
}

/*
 * Compute patch color matching cost.
 */
static double ComputeColorCost(vector<Mat>& patchHistory, Mat& target)
{
	//imshow("target", target);
	MatND histTarget;
	ComputeHistogram(target, histTarget);

	double cost = 0.0;
	int count = patchHistory.size();
	for (int i = 0; i < count; i++) {
		Mat patch = patchHistory[i];
		//imshow("patch", patch);
		//waitKey(0);
		MatND histPatch;
		ComputeHistogram(patch, histPatch);

		// match histogram between object and candidates
		double match = compareHist(histPatch, histTarget, HISTCMP_BHATTACHARYYA);
		//printf("match = %f\n", match);
		cost += match;
  	}
	
	if (count > 0)
		cost /= count;

	//printf("cost = %f\n", cost);
	return cost;
}


/*
 * Class member functions.
 */

/*
 * Destructor.
 */
CTracker::~CTracker()
{
    if (_pORF != NULL)
    	delete _pORF;

    if (_pFeatureSpace != NULL)
    	delete _pFeatureSpace;
}

/*
 * Constructor.
 */
CTracker::CTracker(int id)
{
	_id = id;
    _state = IDLE;

    _pCSRTTracker = DPTrackerCSRT::create();
    _pFeatureSpace = new CHOGSpace;

    // initialize ORF
	_orfParams.numClasses = 2;
	_orfParams.numTrees = 16;
	_orfParams.maxDepth = 7;
	_orfParams.useSoftVoting = 1;
	_orfParams.numEpochs = 1;
	_orfParams.numFeatures = 1024;
	_orfParams.numRandomTests = 16;
	_orfParams.numProjFeatures = 1;
	_orfParams.counterThreshold = 100;
	_orfParams.verbose = 0;

	_pORF = NULL;

	for (int i = 0; i < _historySize * 2; i++) {
		Sample *sample = new Sample(_pFeatureSpace->GetFeatureDimension());
		_trainingSet.push_back(sample);
	}
}

/*
 * Initialize the tracker.
 */
void CTracker::Initialize(Mat& frame, Rect& boundingBox)
{
    //printf("#Init\n");
	_curFrame = frame;
	_imageSize = frame.size();
	_boundingBox = boundingBox;
	_mask = Mat();
	_patch = frame(boundingBox);
	_state = ACTIVE;
	_lostCount = 0;
	_missCount = 0;
    _trjWeight = _trjWeightInit;
    _trackingCount = 0;
    _confMean = 0.0;
    _confStd = 0.0;

    // initialize CSR tracker
    _pCSRTTracker->init(frame, boundingBox);

	// initialize bbox history
	_bboxHistory.clear();
	for (int i = 0; i < _historySize; i++)
		_bboxHistory.push_back(_boundingBox);

	// initialize patch history
	_patchHistory.clear();
	Mat patch;
	resize(_patch, patch, Size(64, 128));
	for (int i = 0; i < _historySize; i++)
		_patchHistory.push_back(patch);

	_featureHistory.clear();

#if 0
	FILE *pf = fopen("/tmp/orf.bin", "rb");
	if (pf != NULL) {
		_orfParams.Load(pf);
		_orfParams.Print();

		if (_pORF != NULL)
			delete _pORF;
		_pORF = new OnlineRF(_orfParams);

		_pORF->Load(pf);
		_pORF->Analyse();
		fclose(pf);
	} else {
		printf("Failed to open ORF file\n");
	}
#else
	if (_pORF != NULL)
		delete _pORF;
	memset(&_orfParams, 0, sizeof(_orfParams));
	_orfParams.numClasses = 2;
	_orfParams.numFeatures = 1024;
	_orfParams.numTrees = 20;
	_orfParams.maxDepth = 7;
	_orfParams.numRandomTests = 16;
	_orfParams.numProjFeatures = 1;
	_orfParams.numEpochs = 1;
	_orfParams.useSoftVoting = 1;
	_pORF = new OnlineRF(_orfParams);
	for (int i = 0; i < 10; i++)
		TrainORF();
#endif
}

void CTracker::Stop()
{
	_state = IDLE;
}

/*
 * Compute mean and variance of a list of values.
 */
static void ComputeMeanVar(vector<double>& list, double& mean, double& var)
{
	int size = list.size();
	mean = 0.0;
	for (int i = 0; i < size; i++) {
		//printf("[%d]: %f\n", i, list[i]);
		mean += list[i];
	}
	mean /= size;

	var = 0.0;
	for (int i = 0; i < size; i++) {
		var += (list[i] - mean) * (list[i] - mean);
	}
	var /= size;
}

/*
 * Update tracker confidence mean and standard deviation.
 */
void CTracker::UpdateConfidence(double conf)
{
	if (_trackingCount == 0) {
		_confMean = conf;
		_confStd = 0.0;
	} else {
		_confMean = (_confMean * _trackingCount + conf) / (_trackingCount + 1);
		_confStd = std::sqrt(((_confStd * _confStd) * (_trackingCount - 1)
				+ (_confMean - conf) * (_confMean - conf) * (_trackingCount + 1) / _trackingCount) / _trackingCount);
	}

	//printf("  conf: %d, %f, %f, %f\n", _trackingCount, conf, _confMean, _confStd);
	_trackingCount++;
}

bool CTracker::DetectOcclusionORF(Mat& patch)
{
	Sample sample;
	Result result;
	resize(patch, sample.imagePatch, Size(64,128));

	sample.x = new int[1024];
	_pFeatureSpace->ComputeSampleFeatures(sample);

	_pORF->Evaluate(sample, result);
	//printf("orf: %d, %f, %f\n", result.prediction, result.confidence[0], result.confidence[1]);
	if (result.prediction == 1 && result.confidence[1] > 0.5)
		return false;
	else
		return true;
}

bool CTracker::DetectOcclusion(Mat& patch, double conf)
{
	// good match
	double delta = max(_confStd, _confMean * 0.05);
	//if (conf > _confMean + delta * 2.0)
	//	return false;

	// bad match
	double thr = max(10.0, _confMean - delta * 3.0);
	if (conf < thr)
		return true;

	return false;

	// check ORF
	return DetectOcclusionORF(patch);
}

bool CTracker::DetectMistrack(Rect bbox, vector<DetectionInfo>& detections)
{
	if (detections.size() > 0) {
		// find best IOU
		double maxIOU = -1.0;
		for (int i = 0; i < detections.size(); i++) {
			Rect2d r = detections[i].bbox;
			Rect2d b = bbox;
			double iou = (b & r).area() / (b | r).area();
			if (iou > maxIOU)
				maxIOU = iou;
		}
		if (maxIOU < 0.2) {
			_missCount++;
			if (_missCount > 3) {
				return true;
			}
		} else {
			_missCount = 0;
		}
	}

	return false;
}

/*
 * Tracking a object.
 * To do:
 *   - Update state-space model
 *   - Run searching based on state-space and appearance model
 *   - Update final prediction
 */
void CTracker::Track(Mat& frame, vector<DetectionInfo>& detections)
{
	if (_state != ACTIVE)
		return;

	_curFrame = frame;
	_imageSize = frame.size();

	//printf("  tracker %d, state = %d\n", _id, _state);
	//printf("  bbox = [%d, %d, %d, %d], %d\n", _boundingBox.x, _boundingBox.y,
	//		_boundingBox.width, _boundingBox.height, _featureHistory.size());

	// run base tracker
	Rect2d bbox;
	double confTracker = _pCSRTTracker->update(frame, bbox);
	Rect bboxTracker;
	bboxTracker.x = (int)floor(bbox.x + 0.5);
	bboxTracker.y = (int)floor(bbox.y + 0.5);
	bboxTracker.width = (int)floor(bbox.x + bbox.width + 0.5) - bboxTracker.x;
	bboxTracker.height = (int)floor(bbox.y + bbox.height + 0.5) - bboxTracker.y;

	// clip tracker bbox within image boundary
	Rect bboxClip;
	bboxClip.x = max(0, bboxTracker.x);
	bboxClip.y = max(0, bboxTracker.y);
	int x1 = min(frame.cols, (bboxTracker.x + bboxTracker.width));
	int y1 = min(frame.rows, (bboxTracker.y + bboxTracker.height));
	bboxClip.width = x1 - bboxClip.x;
	bboxClip.height = y1 - bboxClip.y;

	// check if object touches image boundary
	bool touchBoundary = false;
	if (bboxTracker.x < bboxClip.x || bboxTracker.y < bboxClip.y
			|| bboxTracker.x + bboxTracker.width > bboxClip.x + bboxClip.width
			|| bboxTracker.y + bboxTracker.height > bboxClip.y + bboxClip.height)
		touchBoundary = true;

	// run ORF to check occlusion
	Mat patch = frame(bboxClip);
	if (DetectOcclusion(patch, confTracker)) {
		// stop tracker if object touch image boundary
		if (_featureHistory.size() > 0 && !touchBoundary)
			_state = OCCLUDED;
		else
			_state = IDLE;
		return;
	}

	if (DetectMistrack(bboxClip, detections)) {
		_state = IDLE;
		return;
	}

	// update tracking confidence statistics
	UpdateConfidence(confTracker);

#if 0
	// display tracker result
	Mat display;
	frame.copyTo(display);
	rectangle(display, bboxClip, Scalar(0,0,255), 2);
	int centerX = (int)floor(bboxClip.x + bboxClip.width / 2 + 0.5);
	int centerY = (int)floor(bboxClip.y + bboxClip.height / 2 + 0.5);
	circle(display, Point(centerX, centerY), 2, Scalar(0,0,255), 2);
	imshow("Base Tracker", display);
#endif

	// update bounding box with detections
	Rect2d bestBox = bboxClip;
	double lrBox = 0.5;
#if 1
	if (detections.size() > 0) {
		// find best matching cost from all detections
		double minCost = 999;
		int minIndex = -1;
		for (int i = 0; i < detections.size(); i++) {
			Mat target = frame(detections[i].bbox);
			double appCost = ComputeColorCost(_patchHistory, target);
			double trjCost = ComputeTrajectoryCost(detections[i].bbox);
			double cost = appCost + trjCost * _trjWeight;
			//printf("  target %d, %f, %f, %f\n", i, appCost, trjCost, cost);
			if (cost < minCost) {
				minCost = cost;
				minIndex = i;
			}
		}
		//printf("    min = %d, %f\n", minIndex, _appCost);

		// cross checking between tracker result and the best detection
		if (minIndex >= 0) {
			Rect2d r = detections[minIndex].bbox;
			Rect2d bboxClip1 = bboxClip;
			double iou = (bboxClip1 & r).area() / (bboxClip1 | r).area();
			if (iou > 0.5 && iou < 0.9) {
				double x = (bestBox.x + r.x) / 2 + (bestBox.width + r.width) / 4;
				double y = (bestBox.y + r.y) / 2 + (bestBox.height + r.height) / 4;
				double width = (bestBox.width + r.width) / 2;
				double height = (bestBox.height + r.height) / 2;
				bestBox.x = max((int)(x - width / 2), 0);
				bestBox.y = max((int)(y - height / 2), 0);
				double x1 = min((int)(x + width / 2), _imageSize.width);
				double y1 = min((int)(y + height / 2), _imageSize.height);
				bestBox.width = x1 - bestBox.x;
				bestBox.height = y1 - bestBox.y;
				detections[minIndex].trackerId.push_back(_id);
				_pCSRTTracker->init(frame, bestBox);
			}
		}
		//printf("  bestBox = [%d, %d, %d, %d]\n", bestBox.x, bestBox.y, bestBox.width, bestBox.height);
	} else {
		double x = bestBox.x + bestBox.width / 2;
		double y = bestBox.y + bestBox.height / 2;
		double width = bestBox.width * lrBox + _boundingBox.width * (1.0 - lrBox);
		double height = bestBox.height * lrBox + _boundingBox.height * (1.0 - lrBox);
		bestBox.x = max((int)(x - width / 2), 0);
		bestBox.y = max((int)(y - height / 2), 0);
		double x1 = min((int)(x + width / 2), _imageSize.width);
		double y1 = min((int)(y + height / 2), _imageSize.height);
		bestBox.width = x1 - bestBox.x;
		bestBox.height = y1 - bestBox.y;
	}
#endif

	// set tracked bbox
	SetObject(frame, bestBox, confTracker);
}

/*
 * Set final object decision to the tracker.
 */
void CTracker::SetObject(Mat& frame, Rect bbox, double confTracker)
{
	_boundingBox = bbox;
	_patch = frame(bbox);

	// update bounding box history
	_bboxHistory.push_back(bbox);
	if (_bboxHistory.size() > _historySize)
		_bboxHistory.erase(_bboxHistory.begin());

	// update long-term history
	double ltThr = _confMean + _confStd * 0.5;
	if (_trackingCount > 3 && confTracker > ltThr) {
		Mat patch;
		resize(_patch, patch, Size(64, 128));

		// update patch history
		_patchHistory.push_back(patch);
		if (_patchHistory.size() > _historySize)
			_patchHistory.erase(_patchHistory.begin());

		MatND hist;
		ComputeHistogram(patch, hist);
		//printf("    #kp = %d\n", stat.keyPoints.size());
		_featureHistory.push_back(hist);
		if (_featureHistory.size() > _historySize) {
			_featureHistory.erase(_featureHistory.begin());
		}
		//printf("  lt history: %d\n", _featureHistory.size());
		//DisplayHistory();

		TrainORF();
	}
}

void CTracker::GenerateNegativeSample()
{
	int offset[16][2] = {
		{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
		{-2, -1}, {2, -1},
		{-2, 0}, {2, 0},
		{-2, 1}, {2, 1},
		{-2, 2}, {-1, 2}, {0, 2}, {1, 2}, {2, 2},
	};
	int w2 = _boundingBox.width / 2, h2 = _boundingBox.height / 2;
	int index = _historySize;

	for (int i = 0; i < 16; i++) {
		int x = _boundingBox.x - offset[i][0] * w2;
		int y = _boundingBox.y - offset[i][0] * h2;
		if (x >= 0 && x < _imageSize.width - _boundingBox.width
				&& y >= 0 && y < _imageSize.height - _boundingBox.height) {
			Sample* sample = _trainingSet[index];
			sample->w = 1.0;
			sample->y = 0;
			resize(_curFrame(Rect(x, y, _boundingBox.width, _boundingBox.height)), sample->imagePatch, Size(64, 128));
			_pFeatureSpace->ComputeSampleFeatures(*sample);
			index++;
		}
		if (index == (int)_trainingSet.size() - 1)
			break;
	}

	for (int i = index, j = 0; i < (int)_trainingSet.size(); i++, j++) {
		Sample* sample = _trainingSet[i];
		sample->w = 1.0;
		sample->y = 0;
		sample->imagePatch = _trainingSet[j]->imagePatch;
		_pFeatureSpace->ComputeSampleFeatures(*sample);
	}
}

void CTracker::TrainORF()
{
	// use long-term history for positive samples
	for (int i = 0; i < _historySize; i++) {
		Sample* sample = _trainingSet[i];
		sample->w = 10.0;
		sample->y = 1;
		sample->imagePatch = _patchHistory[i];
		_pFeatureSpace->ComputeSampleFeatures(*sample);
	}

	// generate negative samples
	GenerateNegativeSample();

	// train ORF
	_pORF->Train(_trainingSet);
}

/*
 * Check object recovery.
 */
int CTracker::CheckRecover(Mat& frame, vector<DetectionInfo>& detections)
{
	int speed = 3, range = speed * _lostCount;
	int bestIndex = -1;
	double bestMatch = 100;
	for (int i = 0; i < detections.size(); i++) {
		// check if detection is being tracked
		if (detections[i].trackerId.size() > 0)
			continue;

		// check range
		Rect bboxDet = detections[i].bbox;
		int dist = (abs(_boundingBox.x - bboxDet.x + (_boundingBox.width - bboxDet.width) / 2)
				+ abs(_boundingBox.y - bboxDet.y + (_boundingBox.height - bboxDet.height) / 2)) / 2;
		if (dist > range)
			continue;

		Mat patch = frame(detections[i].bbox);

		// check with ORF
		Sample sample;
		Result result;
		resize(patch, sample.imagePatch, Size(64,128));
		sample.x = new int[1024];
		_pFeatureSpace->ComputeSampleFeatures(sample);
		_pORF->Evaluate(sample, result);
		if (result.prediction == 0 || result.confidence[1] < 0.7)
			continue;

		// check color
		MatND histTarget;
		ComputeHistogram(patch, histTarget);

		vector<double> costs;
		for (int j = 0; j < _featureHistory.size(); j++) {
			MatND hist = _featureHistory[j];
			double match = compareHist(hist, histTarget, HISTCMP_BHATTACHARYYA);
			costs.push_back(match);
			//printf("  match = %f, %f\n", match, cost);
		}

		sort(costs.begin(), costs.end());
		double cost = 0;
		int count = 0;
		for (int j = 0; j < costs.size(); j++) {
			//printf("    cost %d, %f\n", j, costs[j]);
			cost += costs[j];
			count++;
			if (count >= 5)
				break;
		}
		if (count > 0)
			cost /= count;

		if (cost < bestMatch) {
			bestMatch = cost;
			bestIndex = i;
		}
	}
	//printf("  Best match %d, %f\n", bestIndex, bestMatch);

#if 0
	Mat display;
	frame.copyTo(display);
	rectangle(display, boundingBoxes[bestIndex], Scalar(0,0,255), 2);
	imshow("Recover", display);
#endif

	double thr = 0.6;
	if (bestIndex >= 0 && bestMatch < thr)
		return bestIndex;
	else
		return -1;
}

void CTracker::Recover(Mat& frame, vector<DetectionInfo>& detections)
{
	if (_state != OCCLUDED)
		return;

	_curFrame = frame;
	_imageSize = frame.size();

	printf("  tracker %d, state = %d\n", _id, _state);

	// check if object is recovered
	int index = CheckRecover(frame, detections);
	if (index >= 0 && detections[index].trackerId.size() == 0) {
		Rect bbox = detections[index].bbox;
		printf("  #Recovered [%d, %d, %d, %d]\n", bbox.x, bbox.y, bbox.width, bbox.height);
		SetObject(frame, bbox, 0.0);
		_pCSRTTracker->init(frame, bbox);
		_trjWeight = _trjWeightInit;
		_state = ACTIVE;
		_lostCount = 0;
		detections[index].trackerId.push_back(_id);
	} else {
		// decrease trajectory weight
		_trjWeight = max(_trjWeight - _trjWeightInit * 0.01, 0.0);
		// release tracker if object is lost for too long
		_lostCount++;
		if (_lostCount > 150) {
			_state = IDLE;
		}
	}
}

/*
 * Compute trajectory cost.
 */
double CTracker::ComputeTrajectoryCost(Rect& target)
{
	Point p0 = GetRectCenter(_bboxHistory[_bboxHistory.size() - 2]);
	Point p1 = GetRectCenter(_bboxHistory[_bboxHistory.size() - 1]);
	Point p2 = GetRectCenter(target);
	double dist02 = norm(p0 - p2);
	if (dist02 < 1.0)
		return dist02;
	double distance = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
	return abs(distance);
}

void CTracker::DisplayHistory()
{
	Mat display = Mat::zeros(Size(_historySize * 64, 128), CV_8UC3);
	for (int i = 0; i < _patchHistory.size(); i++) {
		Rect r(i*64, 0, 64, 128);
		_patchHistory[i].copyTo(display(r));
	}
	imshow("patch", display);
}

} // namespace dp_tracking
} // namespace cv

