/*
 * Copyright Deep Photon.
 */

#ifndef __OPENCV_DP_BGMODEL_PRECOMP_HPP__
#define __OPENCV_DP_BGMODEL_PRECOMP_HPP__

#include <opencv2/dp_bgmodel.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <cmath>
#include <valarray>

template<typename T> inline const T* get_data(const std::valarray<T>& arr)
{
	return &((std::valarray<T>&)arr)[0];
}

#include "Retina.hpp"
#include "FastToneMapping.hpp"
#include "TransientSegmentation.hpp"

#endif
