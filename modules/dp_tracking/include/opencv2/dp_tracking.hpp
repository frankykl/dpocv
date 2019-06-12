/*
 * Copyright Deep Photon Inc.
 */

#ifndef __OPENCV_DP_TRACKING_HPP__
#define __OPENCV_DP_TRACKING_HPP__

#include "opencv2/video.hpp"

#ifdef __cplusplus

/** @defgroup dp_tracking Deep Photon algorithms
 */

namespace cv {
namespace dp_tracking {

//! @addtogroup dp_tracking
//! @{

/*
 * Tracker state.
 */
#define DP_TRACKER_STATE_INVALID     (-1)
#define DP_TRACKER_STATE_IDLE          0
#define DP_TRACKER_STATE_ACTIVE        1
#define DP_TRACKER_STATE_OCCLUDED      2


/** @brief Deep Photon multi-target tracker.

 The class implements Deep Photon multi-target tracker.
 */
class CV_EXPORTS_W DPMTTracker : public Algorithm
{
public:
	CV_WRAP virtual int GetNumberTrackers() = 0;
	CV_WRAP virtual int GetTrackerState(int id) = 0;
	CV_WRAP virtual Rect GetTrackerBBox(int id) = 0;
	CV_WRAP virtual bool StartTracker(int trackerId, Mat& frame, Mat& boundingBox) = 0;
	CV_WRAP virtual void StopTracker(int trackerId) = 0;
	CV_WRAP virtual void ProcessFrame(Mat& frame, Mat& boundingBox) = 0;
};

/** @brief Creates an multi-target tracker
 */
CV_EXPORTS_W Ptr<DPMTTracker>
CreateDPMTTracker(int numberTrackers, double sigmaGaussian = 1.0);

} // namespace dp_tracking
} // namespace cv

#endif // #ifdef __cplusplus
#endif // #ifndef __OPENCV_DP_TRACKING_HPP__

