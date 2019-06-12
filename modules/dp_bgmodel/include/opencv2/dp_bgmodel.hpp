/*
 * Copyright Deep Photon Inc.
 */

#ifndef __OPENCV_DP_BGMODEL_HPP__
#define __OPENCV_DP_BGMODEL_HPP__

#include "opencv2/video.hpp"

#ifdef __cplusplus

/** @defgroup dp_bgmodel Deep Photon algorithms
 */

namespace cv {
namespace dp_bgmodel {

//! @addtogroup dp_bgmodel
//! @{

//! @addtogroup dp_bgmodel
//! @{

/** @brief Deep Photon background modeler.

 The class implements Deep Photon background modeler.
 */
class CV_EXPORTS_W DPBackgroundModeler : public Algorithm
{
public:
	CV_WRAP virtual void ProcessFrame(Mat& in) = 0;
	CV_WRAP virtual Mat& GetParvoOutput() = 0;
	CV_WRAP virtual Mat& GetMotionMap() = 0;
	CV_WRAP virtual Mat& GetMotionMask() = 0;
	CV_WRAP virtual Mat& GetBackgroundFrame() = 0;
	CV_WRAP virtual Mat& GetForegroundFrame() = 0;
};


/** @brief Creates an background modeler.
 */
CV_EXPORTS_W Ptr<DPBackgroundModeler>
CreateDPBackgroundModeler();


} // namespace dp_bgmodel
} // namespace cv

#endif // #ifdef __cplusplus
#endif // #ifndef __OPENCV_DP_BGMODEL_HPP__

