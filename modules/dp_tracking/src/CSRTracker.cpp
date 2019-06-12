// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "CSRTracker.hpp"

namespace cv {
namespace dp_tracking {

/*
 * CSRT model.
 */
class DPTrackerCSRTModel: public TrackerModel {
public:
	DPTrackerCSRTModel(DPTrackerCSRT::Params /*params*/) {}
	~DPTrackerCSRTModel() {}

protected:
	void modelEstimationImpl(const std::vector<Mat>& /*responses*/) CV_OVERRIDE {}
	void modelUpdateImpl() CV_OVERRIDE {}
};

/*
 * CSRT implementation class.
 */
class DPTrackerCSRTImpl: public DPTrackerCSRT {
public:
	DPTrackerCSRTImpl(const DPTrackerCSRT::Params &parameters = DPTrackerCSRT::Params());
	void read(const FileNode& fn);
	void write(FileStorage& fs) const;

protected:
	DPTrackerCSRT::Params params;

	// API functions
	bool initImpl(const Mat& image, const Rect2d& boundingBox);
	void setInitialMask(const Mat mask);
	bool updateImpl(const Mat& image, Rect2d& boundingBox);
	double update(const Mat& image, Rect2d& boundingBox);

	// internal functions
	void UpdateCSRFilter(const Mat &image, const Mat &my_mask);
	void UpdateHistograms(const Mat &image, const Rect &region);
	void ExtractHistograms(const Mat &image, cv::Rect region, Histogram &hf, Histogram &hb);
	std::vector<Mat> CreateCSRFilter(const std::vector<cv::Mat> img_features, const cv::Mat Y,
			const cv::Mat P);
	Mat CalculateResponse(const Mat &image, const std::vector<Mat> filter);
	Mat GetLocationPrior(const Rect roi, const Size2f target_size, const Size img_sz);
	Mat SegmentRegion(const Mat &image, const Point2f &objectCenter, const Size2f &templateSize,
			const Size &target_size, float scale_factor);
	Point2f EstimateNewPosition(const Mat &image);
	std::vector<Mat> GetFeatures(const Mat &patch, const Size2i &feature_size);

private:
	bool CheckMaskArea(const Mat &mat, const double obj_area);
	void UpdateBBox();

	float currentScaleFactor;
	Mat window;
	Mat yf;
	Rect2f _boundingBox;
	std::vector<Mat> csrFilter;
	std::vector<Mat> csrFilterLT;
	std::vector<float> filterWeights;
	Size2f originalTargetSize;
	Size2i imageSize;
	Size2f templateSize;
	Size2i rescaledTemplateSize;
	float rescaleRatio;
	Point2f objectCenter;
	DSST dsst;
	Histogram histForeground;
	Histogram histBackground;
	double p_b;
	Mat erodeElement;
	Mat filterMask;
	Mat presetMask;
	Mat defaultMask;
	float defaultMaskArea;
	int cellSize;
	Mat _objectMask;
};

/*
 * Create CSRT tracker with parameters.
 */
Ptr<DPTrackerCSRT> DPTrackerCSRT::create(const DPTrackerCSRT::Params &parameters)
{
	return Ptr<DPTrackerCSRTImpl>(new DPTrackerCSRTImpl(parameters));
}

/*
 * Create CSRT tracker without parameters.
 */
Ptr<DPTrackerCSRT> DPTrackerCSRT::create()
{
	return Ptr<DPTrackerCSRTImpl>(new DPTrackerCSRTImpl());
}

/*
 * Constructor.
 */
DPTrackerCSRTImpl::DPTrackerCSRTImpl(const DPTrackerCSRT::Params &parameters) :
		params(parameters)
{
	isInit = false;
}

/*
 * Read parameters.
 */
void DPTrackerCSRTImpl::read(const cv::FileNode& fn)
{
	params.read(fn);
}

/*
 * Write parameters.
 */
void DPTrackerCSRTImpl::write(cv::FileStorage& fs) const
{
	params.write(fs);
}

void DPTrackerCSRTImpl::setInitialMask(const Mat mask)
{
	presetMask = mask;
}

/*
 * Check mask area against object area.
 */
bool DPTrackerCSRTImpl::CheckMaskArea(const Mat &mat, const double objArea)
{
	double threshold = 0.05;
	double maskArea = sum(mat)[0];
	if (maskArea < threshold * objArea) {
		return false;
	}
	return true;
}

/*
 * Calculate response.
 */
Mat DPTrackerCSRTImpl::CalculateResponse(const Mat &image, const std::vector<Mat> filter)
{
	//printf("CalculateResponse\n");
	Mat patch = GetSubwindow(image, objectCenter,
			cvFloor(currentScaleFactor * templateSize.width),
			cvFloor(currentScaleFactor * templateSize.height));
	resize(patch, patch, rescaledTemplateSize, 0, 0, INTER_CUBIC);

	std::vector<Mat> ftrs = GetFeatures(patch, yf.size());
	std::vector<Mat> Ffeatures = FourierTransformFeatures(ftrs);
	Mat resp, res;
	if (params.use_channel_weights) {
		res = Mat::zeros(Ffeatures[0].size(), CV_32FC2);
		Mat resp_ch;
		Mat mul_mat;
		for (size_t i = 0; i < Ffeatures.size(); ++i) {
			mulSpectrums(Ffeatures[i], filter[i], resp_ch, 0, true);
			res += (resp_ch * filterWeights[i]);
		}
		idft(res, res, DFT_SCALE | DFT_REAL_OUTPUT);
	} else {
		res = Mat::zeros(Ffeatures[0].size(), CV_32FC2);
		Mat resp_ch;
		for (size_t i = 0; i < Ffeatures.size(); ++i) {
			mulSpectrums(Ffeatures[i], filter[i], resp_ch, 0, true);
			res = res + resp_ch;
		}
		idft(res, res, DFT_SCALE | DFT_REAL_OUTPUT);
	}
	return res;
}

/*
 * Update CSR filter.
 */
void DPTrackerCSRTImpl::UpdateCSRFilter(const Mat &image, const Mat &mask)
{
	//printf("UpdateCSRFilter\n");
	Mat patch = GetSubwindow(image, objectCenter,
			cvFloor(currentScaleFactor * templateSize.width),
			cvFloor(currentScaleFactor * templateSize.height));
	resize(patch, patch, rescaledTemplateSize, 0, 0, INTER_CUBIC);

	std::vector<Mat> features = GetFeatures(patch, yf.size());
	std::vector<Mat> filters = FourierTransformFeatures(features);
	std::vector<Mat> newFilter = CreateCSRFilter(filters, yf, mask);
	// calculate per channel weights
	if (params.use_channel_weights) {
		Mat currentResp;
		double maxVal;
		float sumWeights = 0;
		std::vector<float> new_filter_weights = std::vector<float>(newFilter.size());
		for (size_t i = 0; i < newFilter.size(); ++i) {
			mulSpectrums(filters[i], newFilter[i], currentResp, 0, true);
			idft(currentResp, currentResp, DFT_SCALE | DFT_REAL_OUTPUT);
			minMaxLoc(currentResp, NULL, &maxVal, NULL, NULL);
			sumWeights += static_cast<float>(maxVal);
			new_filter_weights[i] = static_cast<float>(maxVal);
		}

		// update filter weights with new values using learning rate
		float updated_sum = 0;
		for (size_t i = 0; i < filterWeights.size(); ++i) {
			filterWeights[i] = filterWeights[i] * (1.0f - params.weights_lr)
					+ params.weights_lr * (new_filter_weights[i] / sumWeights);
			updated_sum += filterWeights[i];
		}
		// normalize weights
		for (size_t i = 0; i < filterWeights.size(); ++i) {
			filterWeights[i] /= updated_sum;
		}
	}

	// update by learning rate
	for (size_t i = 0; i < csrFilter.size(); ++i) {
		csrFilter[i] = (1.0f - params.filter_lr) * csrFilter[i] + params.filter_lr * newFilter[i];
		// long-term feature learning
		double lr = params.filter_lr / 5;
		csrFilterLT[i] = (1.0f - lr) * csrFilterLT[i] + lr * newFilter[i];
	}
	std::vector<Mat>().swap(features);
	std::vector<Mat>().swap(filters);
}

/*
 * Get features.
 */
std::vector<Mat> DPTrackerCSRTImpl::GetFeatures(const Mat &patch, const Size2i &feature_size)
{
	//printf("GetFeatures\n");
	std::vector<Mat> features;

	// HOG feature
	if (params.use_hog) {
		std::vector<Mat> hog = GetFeaturesHOG(patch, cellSize);
		//printf("hog = %d\n", hog.size());
		features.insert(features.end(), hog.begin(), hog.begin() + params.num_hog_channels_used);
	}

	// ColorNames feature
	if (params.use_color_names) {
		std::vector<Mat> cn;
		cn = GetFeaturesCN(patch, feature_size);
		features.insert(features.end(), cn.begin(), cn.end());
	}

	// gray scale feature
	if (params.use_gray) {
		Mat gray_m;
		cvtColor(patch, gray_m, CV_BGR2GRAY);
		resize(gray_m, gray_m, feature_size, 0, 0, INTER_CUBIC);
		gray_m.convertTo(gray_m, CV_32FC1, 1.0 / 255.0, -0.5);
		features.push_back(gray_m);
	}

	// RGB feature
	if (params.use_rgb) {
		std::vector<Mat> rgb_features = GetFeaturesRGB(patch, feature_size);
		features.insert(features.end(), rgb_features.begin(), rgb_features.end());
	}

	// apply window function
	for (size_t i = 0; i < features.size(); ++i) {
		features.at(i) = features.at(i).mul(window);
	}
	return features;
}

/*
 * Parallel implementation.
 */
class ParallelCreateCSRFilter: public ParallelLoopBody {
public:
	// constructor
	ParallelCreateCSRFilter(const std::vector<cv::Mat> img_features, const cv::Mat Y,
			const cv::Mat P, int admm_iterations, std::vector<Mat> &result_filter_) :
			result_filter(result_filter_)
	{
		this->img_features = img_features;
		this->Y = Y;
		this->P = P;
		this->admm_iterations = admm_iterations;
	}

	// operator
	virtual void operator ()(const Range& range) const CV_OVERRIDE
	{
		for (int i = range.start; i < range.end; i++) {
			float mu = 5.0f;
			float beta = 3.0f;
			float mu_max = 20.0f;
			float lambda = mu / 100.0f;

			Mat F = img_features[i];

			Mat Sxy, Sxx;
			mulSpectrums(F, Y, Sxy, 0, true);
			mulSpectrums(F, F, Sxx, 0, true);

			Mat H;
			H = DivideComplexMatrices(Sxy, (Sxx + lambda));
			idft(H, H, DFT_SCALE|DFT_REAL_OUTPUT);
			H = H.mul(P);
			dft(H, H, DFT_COMPLEX_OUTPUT);
			Mat L = Mat::zeros(H.size(), H.type()); // Lagrangian multiplier
			Mat G;
			for(int iteration = 0; iteration < admm_iterations; ++iteration) {
				G = DivideComplexMatrices((Sxy + (mu * H) - L) , (Sxx + mu));
				idft((mu * G) + L, H, DFT_SCALE | DFT_REAL_OUTPUT);
				float lm = 1.0f / (lambda+mu);
				H = H.mul(P*lm);
				dft(H, H, DFT_COMPLEX_OUTPUT);

				// Update variables for next iteration
				L = L + mu * (G - H);
				mu = min(mu_max, beta * mu);
			}
			result_filter[i] = H;
		}
	}

	// assignment operator
	ParallelCreateCSRFilter& operator=(const ParallelCreateCSRFilter &)
	{
		return *this;
	}

private:
	int admm_iterations;
	Mat Y;
	Mat P;
	std::vector<Mat> img_features;
	std::vector<Mat> &result_filter;
};

/*
 * Create CSR filter.
 */
std::vector<Mat> DPTrackerCSRTImpl::CreateCSRFilter(const std::vector<cv::Mat> img_features,
		const cv::Mat Y, const cv::Mat P)
{
	std::vector<Mat> result_filter;
	result_filter.resize(img_features.size());
	ParallelCreateCSRFilter parallelCreateCSRFilter(img_features, Y, P, params.admm_iterations,
			result_filter);
	parallel_for_(Range(0, static_cast<int>(result_filter.size())), parallelCreateCSRFilter);

	return result_filter;
}

/*
 * Get location prior.
 */
Mat DPTrackerCSRTImpl::GetLocationPrior(const Rect roi, const Size2f target_size,
		const Size img_sz)
{
	int x1 = cvRound(max(min(roi.x - 1, img_sz.width - 1), 0));
	int y1 = cvRound(max(min(roi.y - 1, img_sz.height - 1), 0));

	int x2 = cvRound(min(max(roi.width - 1, 0), img_sz.width - 1));
	int y2 = cvRound(min(max(roi.height - 1, 0), img_sz.height - 1));

	Size target_sz;
	target_sz.width = target_sz.height = cvFloor(min(target_size.width, target_size.height));

	double cx = x1 + (x2 - x1) / 2.;
	double cy = y1 + (y2 - y1) / 2.;
	double kernel_size_width = 1.0 / (0.5 * static_cast<double>(target_sz.width) * 1.4142 + 1);
	double kernel_size_height = 1.0 / (0.5 * static_cast<double>(target_sz.height) * 1.4142 + 1);

	cv::Mat kernel_weight = Mat::zeros(1 + cvFloor(y2 - y1), 1 + cvFloor(-(x1 - cx) + (x2 - cx)),
			CV_64FC1);
	for (int y = y1; y < y2 + 1; ++y) {
		double * weightPtr = kernel_weight.ptr<double>(y);
		double tmp_y = std::pow((cy - y) * kernel_size_height, 2);
		for (int x = x1; x < x2 + 1; ++x) {
			weightPtr[x] = KernelEpan(std::pow((cx - x) * kernel_size_width, 2) + tmp_y);
		}
	}

	double max_val;
	cv::minMaxLoc(kernel_weight, NULL, &max_val, NULL, NULL);
	Mat fg_prior = kernel_weight / max_val;
	fg_prior.setTo(0.5, fg_prior < 0.5);
	fg_prior.setTo(0.9, fg_prior > 0.9);
	return fg_prior;
}

/*
 * Compute object segmentation mask.
 */
Mat DPTrackerCSRTImpl::SegmentRegion(const Mat &image, const Point2f &objectCenter,
		const Size2f &templateSize, const Size &targetSize, float scaleFactor)
{
	//printf("templateSize = %f, %f, targetSize = %d, %d, scaleFactor = %f\n",
	//		templateSize.width, templateSize.height, targetSize.width, targetSize.height, scaleFactor);
	Rect validPixels;
	Mat patch = GetSubwindow(image, objectCenter, cvFloor(scaleFactor * templateSize.width),
			cvFloor(scaleFactor * templateSize.height), &validPixels);
	Size2f scaledTarget = Size2f(targetSize.width * scaleFactor,
			targetSize.height * scaleFactor);
	Mat fgPrior = GetLocationPrior(Rect(0, 0, patch.size().width, patch.size().height),
			scaledTarget, patch.size());

	std::vector<Mat> img_channels;
	split(patch, img_channels);
	std::pair<Mat, Mat> probs = Segment::ComputePosteriors2(img_channels, 0, 0, patch.cols,
			patch.rows, p_b, fgPrior, 1.0 - fgPrior, histForeground, histBackground);

	Mat mask = Mat::zeros(probs.first.size(), probs.first.type());
	probs.first(validPixels).copyTo(mask(validPixels));
	double max_resp = GetMax(mask);
	threshold(mask, mask, max_resp / 2.0, 1, THRESH_BINARY);
	mask.convertTo(mask, CV_32FC1, 1.0);
	return mask;
}

/*
 * Extract histogram for foreground/background segmentation.
 */
void DPTrackerCSRTImpl::ExtractHistograms(const Mat &image, cv::Rect region, Histogram &hf,
		Histogram &hb)
{
	// get coordinates of the region
	int x1 = std::min(std::max(0, region.x), image.cols - 1);
	int y1 = std::min(std::max(0, region.y), image.rows - 1);
	int x2 = std::min(std::max(0, region.x + region.width), image.cols - 1);
	int y2 = std::min(std::max(0, region.y + region.height), image.rows - 1);

	// calculate coordinates of the background region
	int offsetX = (x2 - x1 + 1) / params.background_ratio;
	int offsetY = (y2 - y1 + 1) / params.background_ratio;
	int outer_y1 = std::max(0, (int) (y1 - offsetY));
	int outer_y2 = std::min(image.rows, (int) (y2 + offsetY + 1));
	int outer_x1 = std::max(0, (int) (x1 - offsetX));
	int outer_x2 = std::min(image.cols, (int) (x2 + offsetX + 1));

	// calculate probability for the background
	p_b = 1.0 - ((x2 - x1 + 1) * (y2 - y1 + 1))
			/ ((double) (outer_x2 - outer_x1 + 1) * (outer_y2 - outer_y1 + 1));

	// split multi-channel image into the std::vector of matrices
	std::vector<Mat> img_channels(image.channels());
	split(image, img_channels);
	for (size_t k = 0; k < img_channels.size(); k++) {
		img_channels.at(k).convertTo(img_channels.at(k), CV_8UC1);
	}

	hf.ExtractForegroundHistogram(img_channels, Mat(), false, x1, y1, x2, y2);
	hb.ExtractBackGroundHistogram(img_channels, x1, y1, x2, y2, outer_x1, outer_y1, outer_x2,
			outer_y2);
	std::vector<Mat>().swap(img_channels);
}

/*
 * Update object histogram.
 */
void DPTrackerCSRTImpl::UpdateHistograms(const Mat &image, const Rect &region)
{
	// create temporary histograms
	Histogram hf(image.channels(), params.histogram_bins);
	Histogram hb(image.channels(), params.histogram_bins);
	ExtractHistograms(image, region, hf, hb);

	// get histogram vectors from temporary histograms
	std::vector<double> hf_vect_new = hf.GetHistogramVector();
	std::vector<double> hb_vect_new = hb.GetHistogramVector();
	// get histogram vectors from learned histograms
	std::vector<double> hf_vect = histForeground.GetHistogramVector();
	std::vector<double> hb_vect = histBackground.GetHistogramVector();

	// update histograms using learning rate
	for (size_t i = 0; i < hf_vect.size(); i++) {
		hf_vect_new[i] = (1 - params.histogram_lr) * hf_vect[i]
				+ params.histogram_lr * hf_vect_new[i];
		hb_vect_new[i] = (1 - params.histogram_lr) * hb_vect[i]
				+ params.histogram_lr * hb_vect_new[i];
	}

	// set learned histograms
	histForeground.SetHistogramVector(&hf_vect_new[0]);
	histBackground.SetHistogramVector(&hb_vect_new[0]);

	std::vector<double>().swap(hf_vect);
	std::vector<double>().swap(hb_vect);
}

/*
 * Estimate new object position (center).
 */
Point2f DPTrackerCSRTImpl::EstimateNewPosition(const Mat &image)
{
	// compute CSR filter response
	std::vector<Mat> filter;
	filter.resize(csrFilter.size());
	for (size_t i = 0; i < csrFilter.size(); ++i) {
		double rate = 0.6;
		filter[i] = rate * csrFilter[i] + (1.0 - rate) * csrFilterLT[i];
	}
	Mat resp = CalculateResponse(image, filter);

	Point max_loc;
	double maxVal;
	minMaxLoc(resp, NULL, &maxVal, NULL, &max_loc);
	// take into account also subpixel accuracy
	float col = ((float) max_loc.x) + SubpixelPeak(resp, "horizontal", max_loc);
	float row = ((float) max_loc.y) + SubpixelPeak(resp, "vertical", max_loc);
	if (row + 1 > (float) resp.rows / 2.0f) {
		row = row - resp.rows;
	}
	if (col + 1 > (float) resp.cols / 2.0f) {
		col = col - resp.cols;
	}
	// calculate x and y displacements
	Point2f new_center = objectCenter
			+ Point2f(currentScaleFactor * (1.0f / rescaleRatio) * cellSize * (col),
					currentScaleFactor * (1.0f / rescaleRatio) * cellSize * (row));

	// clip new position within image boundary
	if (new_center.x < 0)
		new_center.x = 0;
	if (new_center.x >= imageSize.width)
		new_center.x = static_cast<float>(imageSize.width - 1);
	if (new_center.y < 0)
		new_center.y = 0;
	if (new_center.y >= imageSize.height)
		new_center.y = static_cast<float>(imageSize.height - 1);

    // normalize response
    Scalar mean, std;
    const double eps = 0.00001;
    meanStdDev(resp, mean, std);
    confidence = (maxVal - mean[0]) / (std[0] + eps); // PSR

	return new_center;
}

/*
 * CSRT tracker initialization function.
 */
bool DPTrackerCSRTImpl::initImpl(const Mat& image_, const Rect2d& boundingBox)
{
	cv::setNumThreads(getNumThreads());

	// treat gray image as color image
	Mat image;
	if (image_.channels() == 1) {
		std::vector<Mat> channels(3);
		channels[0] = channels[1] = channels[2] = image_;
		merge(channels, image);
	} else {
		image = image_;
	}

	// initialize parameters
	currentScaleFactor = 1.0;
	imageSize = image.size();
	_boundingBox = boundingBox;
	double bboxArea = _boundingBox.width * _boundingBox.height;
	// compute cell size
	cellSize = cvFloor(std::min(4.0, std::max(1.0, static_cast<double>(cvCeil(bboxArea / 400.0)))));
	// original target size and area
	originalTargetSize = Size(_boundingBox.size());
	double orgArea = originalTargetSize.width * originalTargetSize.height;
	// compute template size
	templateSize.width = static_cast<float>(cvFloor(originalTargetSize.width + params.padding * sqrt(orgArea)));
	templateSize.height = static_cast<float>(cvFloor(originalTargetSize.height + params.padding * sqrt(orgArea)));
	templateSize.width = templateSize.height = (templateSize.width + templateSize.height) / 2.0f;
	//printf("templateSize = %f, %f\n", templateSize.width, templateSize.height);

	// compute rescale ratio
	rescaleRatio = sqrt(pow(params.template_size, 2) / (templateSize.width * templateSize.height));
	if (rescaleRatio > 1) {
		rescaleRatio = 1;
	}
	rescaledTemplateSize = Size2i(cvFloor(templateSize.width * rescaleRatio),
			cvFloor(templateSize.height * rescaleRatio));
	objectCenter = Point2f(static_cast<float>(boundingBox.x) + originalTargetSize.width / 2.0f,
			static_cast<float>(boundingBox.y) + originalTargetSize.height / 2.0f);

	// compute window function
	yf = GaussianShapedLabels(params.gsl_sigma, rescaledTemplateSize.width / cellSize,
			rescaledTemplateSize.height / cellSize);
	if (params.window_function.compare("hann") == 0) {
		window = GetHanningWindow(Size(yf.cols, yf.rows));
	} else if (params.window_function.compare("cheb") == 0) {
		window = GetChebyshevWindow(Size(yf.cols, yf.rows), params.cheb_attenuation);
	} else if (params.window_function.compare("kaiser") == 0) {
		window = GetKaiserWindow(Size(yf.cols, yf.rows), params.kaiser_alpha);
	} else {
		std::cout << "Not a valid window function" << std::endl;
		return false;
	}

	Size2i scaled_obj_size = Size2i(cvFloor(originalTargetSize.width * rescaleRatio / cellSize),
			cvFloor(originalTargetSize.height * rescaleRatio / cellSize));
	// set dummy mask and area;
	int x0 = std::max((yf.size().width - scaled_obj_size.width) / 2 - 1, 0);
	int y0 = std::max((yf.size().height - scaled_obj_size.height) / 2 - 1, 0);
	defaultMask = Mat::zeros(yf.size(), CV_32FC1);
	defaultMask(Rect(x0, y0, scaled_obj_size.width, scaled_obj_size.height)) = 1.0f;
	defaultMaskArea = static_cast<float>(sum(defaultMask)[0]);

	// initialize segmentation
	if (params.use_segmentation) {
		// convert input frame to HSV
		Mat hsv_img = BGR2HSV(image);
		// compute foreground/background color histogram
		histForeground = Histogram(hsv_img.channels(), params.histogram_bins);
		histBackground = Histogram(hsv_img.channels(), params.histogram_bins);
		ExtractHistograms(hsv_img, _boundingBox, histForeground, histBackground);
		// run segmentation
		filterMask = SegmentRegion(hsv_img, objectCenter, templateSize, originalTargetSize,
				currentScaleFactor);

		// update calculated mask with preset mask
		if (presetMask.data) {
			Mat preset_mask_padded = Mat::zeros(filterMask.size(), filterMask.type());
			int sx = std::max((int) cvFloor(preset_mask_padded.cols / 2.0f - presetMask.cols / 2.0f) - 1, 0);
			int sy = std::max((int) cvFloor(preset_mask_padded.rows / 2.0f - presetMask.rows / 2.0f) - 1, 0);
			presetMask.copyTo(preset_mask_padded(Rect(sx, sy, presetMask.cols, presetMask.rows)));
			filterMask = filterMask.mul(preset_mask_padded);
		}
		erodeElement = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(1, 1));
		resize(filterMask, filterMask, yf.size(), 0, 0, INTER_NEAREST);
		if (CheckMaskArea(filterMask, defaultMaskArea)) {
			dilate(filterMask, filterMask, erodeElement);
		} else {
			filterMask = defaultMask;
		}
	} else {
		filterMask = defaultMask;
	}

	// resize patch
	Mat patch = GetSubwindow(image, objectCenter,
			cvFloor(currentScaleFactor * templateSize.width),
			cvFloor(currentScaleFactor * templateSize.height));
	resize(patch, patch, rescaledTemplateSize, 0, 0, INTER_CUBIC);
	// compute patch features and run FFT
	std::vector<Mat> patchFilters = GetFeatures(patch, yf.size());
	std::vector<Mat> filters = FourierTransformFeatures(patchFilters);
	// create CSR filter
	csrFilter = CreateCSRFilter(filters, yf, filterMask);
	// copy to long-term filters
	csrFilterLT.resize(csrFilter.size());
	for (size_t i = 0; i < csrFilter.size(); ++i) {
		csrFilter[i].copyTo(csrFilterLT[i]);
	}

	// apply channel weights
	if (params.use_channel_weights) {
		Mat currentResp;
		filterWeights = std::vector<float>(csrFilter.size());
		float weightSum = 0;
		for (size_t i = 0; i < csrFilter.size(); ++i) {
			mulSpectrums(filters[i], csrFilter[i], currentResp, 0, true);
			idft(currentResp, currentResp, DFT_SCALE | DFT_REAL_OUTPUT);
			double maxVal;
			minMaxLoc(currentResp, NULL, &maxVal, NULL, NULL);
			weightSum += static_cast<float>(maxVal);
			filterWeights[i] = static_cast<float>(maxVal);
		}
		for (size_t i = 0; i < filterWeights.size(); ++i) {
			filterWeights[i] /= weightSum;
		}
	}

	// initialize scale search
	dsst = DSST(image, _boundingBox, templateSize, params.number_of_scales, params.scale_step,
			params.scale_model_max_area, params.scale_sigma_factor, params.scale_lr);

	// create tracker model
	model = Ptr<DPTrackerCSRTModel>(new DPTrackerCSRTModel(params));

	isInit = true;
	return true;
}

/*
 * CSRT tracker update function.
 * This is the original function, and is not used currently.
 */
bool DPTrackerCSRTImpl::updateImpl(const Mat& image_, Rect2d& boundingBox)
{
	// treat gray image as color image
	Mat image;
	if (image_.channels() == 1) {
		std::vector<Mat> channels(3);
		channels[0] = channels[1] = channels[2] = image_;
		merge(channels, image);
	} else {
		image = image_;
	}

	// estimate new object position
	objectCenter = EstimateNewPosition(image);

	// estimate new object scale
	currentScaleFactor = dsst.GetScale(image, objectCenter);
	//printf("scale = %f\n", currentScaleFactor);
	//update bouding_box according to new scale and location
	_boundingBox.x = objectCenter.x - currentScaleFactor * originalTargetSize.width / 2.0f;
	_boundingBox.y = objectCenter.y - currentScaleFactor * originalTargetSize.height / 2.0f;
	_boundingBox.width = currentScaleFactor * originalTargetSize.width;
	_boundingBox.height = currentScaleFactor * originalTargetSize.height;

	// update object mask
	if (params.use_segmentation) {
		Mat hsv_img = BGR2HSV(image);
		UpdateHistograms(hsv_img, _boundingBox);
		filterMask = SegmentRegion(hsv_img, objectCenter, templateSize, originalTargetSize,
				currentScaleFactor);
		resize(filterMask, filterMask, yf.size(), 0, 0, INTER_NEAREST);
		if (CheckMaskArea(filterMask, defaultMaskArea)) {
			dilate(filterMask, filterMask, erodeElement);
		} else {
			filterMask = defaultMask;
		}
	} else {
		filterMask = defaultMask;
	}

	// update tracker
	UpdateCSRFilter(image, filterMask);
	dsst.Update(image, objectCenter);
	boundingBox = _boundingBox;
	return true;
}

void DPTrackerCSRTImpl::UpdateBBox()
{
	Mat display;
	cvtColor(_objectMask, display, COLOR_GRAY2RGB);
	imshow("mask", display);
}

/*
 * CSRT tracker update function.
 * This is the function being used currently.
 */
double DPTrackerCSRTImpl::update(const Mat& image_, Rect2d& boundingBox)
{
	// treat gray image as color image
	Mat image;
	if (image_.channels() == 1) {
		std::vector<Mat> channels(3);
		channels[0] = channels[1] = channels[2] = image_;
		merge(channels, image);
	} else {
		image = image_;
	}

	// estimate new object position
	objectCenter = EstimateNewPosition(image);

	// estimate new object scale
	currentScaleFactor = dsst.GetScale(image, objectCenter);
	//printf("scale = %f\n", currentScaleFactor);

	// update bouding_box according to new scale and location
	_boundingBox.x = objectCenter.x - currentScaleFactor * originalTargetSize.width / 2.0f;
	_boundingBox.y = objectCenter.y - currentScaleFactor * originalTargetSize.height / 2.0f;
	_boundingBox.width = currentScaleFactor * originalTargetSize.width;
	_boundingBox.height = currentScaleFactor * originalTargetSize.height;

	// update object mask
	if (params.use_segmentation) {
		Mat hsvImg = BGR2HSV(image);
		UpdateHistograms(hsvImg, _boundingBox);
		filterMask = SegmentRegion(hsvImg, objectCenter, templateSize, originalTargetSize,
				currentScaleFactor);
		//filterMask.copyTo(_objectMask);
		resize(filterMask, filterMask, yf.size(), 0, 0, INTER_NEAREST);
		if (CheckMaskArea(filterMask, defaultMaskArea)) {
			dilate(filterMask, filterMask, erodeElement);
		} else {
			filterMask = defaultMask;
		}
	} else {
		filterMask = defaultMask;
	}

	// update bounding box based on mask
	//UpdateBBox();

	// update tracker
	UpdateCSRFilter(image, filterMask);
	dsst.Update(image, objectCenter);
	boundingBox = _boundingBox;
	return confidence;
}

/*
 * Parameter constructor.
 */
DPTrackerCSRT::Params::Params()
{
	use_channel_weights = true;
	use_segmentation = true;
	use_hog = true;
	use_color_names = false;
	use_gray = false;
	use_rgb = true;
	window_function = "hann";
	kaiser_alpha = 3.75f;
	cheb_attenuation = 45;
	padding = 3.0f;
	template_size = 200;
	gsl_sigma = 1.0f;
	hog_orientations = 9;
	hog_clip = 0.2f;
	num_hog_channels_used = 18;
	filter_lr = 0.02f;//0.02f;
	weights_lr = 0.02f;
	admm_iterations = 4;
	number_of_scales = 33;
	scale_sigma_factor = 0.250f;
	scale_model_max_area = 512.0f;
	scale_lr = 0.025f;
	scale_step = 1.020f;
	histogram_bins = 16;
	background_ratio = 2;
	histogram_lr = 0.04f;
}

/*
 * Parameter read.
 */
void DPTrackerCSRT::Params::read(const FileNode& fn)
{
	*this = DPTrackerCSRT::Params();
	if (!fn["padding"].empty())
		fn["padding"] >> padding;
	if (!fn["templateSize"].empty())
		fn["templateSize"] >> template_size;
	if (!fn["gsl_sigma"].empty())
		fn["gsl_sigma"] >> gsl_sigma;
	if (!fn["hog_orientations"].empty())
		fn["hog_orientations"] >> hog_orientations;
	if (!fn["num_hog_channels_used"].empty())
		fn["num_hog_channels_used"] >> num_hog_channels_used;
	if (!fn["hog_clip"].empty())
		fn["hog_clip"] >> hog_clip;
	if (!fn["use_hog"].empty())
		fn["use_hog"] >> use_hog;
	if (!fn["use_color_names"].empty())
		fn["use_color_names"] >> use_color_names;
	if (!fn["use_gray"].empty())
		fn["use_gray"] >> use_gray;
	if (!fn["use_rgb"].empty())
		fn["use_rgb"] >> use_rgb;
	if (!fn["window_function"].empty())
		fn["window_function"] >> window_function;
	if (!fn["kaiser_alpha"].empty())
		fn["kaiser_alpha"] >> kaiser_alpha;
	if (!fn["cheb_attenuation"].empty())
		fn["cheb_attenuation"] >> cheb_attenuation;
	if (!fn["filter_lr"].empty())
		fn["filter_lr"] >> filter_lr;
	if (!fn["admm_iterations"].empty())
		fn["admm_iterations"] >> admm_iterations;
	if (!fn["number_of_scales"].empty())
		fn["number_of_scales"] >> number_of_scales;
	if (!fn["scale_sigma_factor"].empty())
		fn["scale_sigma_factor"] >> scale_sigma_factor;
	if (!fn["scale_model_max_area"].empty())
		fn["scale_model_max_area"] >> scale_model_max_area;
	if (!fn["scale_lr"].empty())
		fn["scale_lr"] >> scale_lr;
	if (!fn["scale_step"].empty())
		fn["scale_step"] >> scale_step;
	if (!fn["use_channel_weights"].empty())
		fn["use_channel_weights"] >> use_channel_weights;
	if (!fn["weights_lr"].empty())
		fn["weights_lr"] >> weights_lr;
	if (!fn["use_segmentation"].empty())
		fn["use_segmentation"] >> use_segmentation;
	if (!fn["histogram_bins"].empty())
		fn["histogram_bins"] >> histogram_bins;
	if (!fn["background_ratio"].empty())
		fn["background_ratio"] >> background_ratio;
	if (!fn["histogram_lr"].empty())
		fn["histogram_lr"] >> histogram_lr;
	CV_Assert(number_of_scales % 2 == 1);
	CV_Assert(use_gray || use_color_names || use_hog || use_rgb);
}

/*
 * Parameter write.
 */
void DPTrackerCSRT::Params::write(FileStorage& fs) const
{
	fs << "padding" << padding;
	fs << "templateSize" << template_size;
	fs << "gsl_sigma" << gsl_sigma;
	fs << "hog_orientations" << hog_orientations;
	fs << "num_hog_channels_used" << num_hog_channels_used;
	fs << "hog_clip" << hog_clip;
	fs << "use_hog" << use_hog;
	fs << "use_color_names" << use_color_names;
	fs << "use_gray" << use_gray;
	fs << "use_rgb" << use_rgb;
	fs << "window_function" << window_function;
	fs << "kaiser_alpha" << kaiser_alpha;
	fs << "cheb_attenuation" << cheb_attenuation;
	fs << "filter_lr" << filter_lr;
	fs << "admm_iterations" << admm_iterations;
	fs << "number_of_scales" << number_of_scales;
	fs << "scale_sigma_factor" << scale_sigma_factor;
	fs << "scale_model_max_area" << scale_model_max_area;
	fs << "scale_lr" << scale_lr;
	fs << "scale_step" << scale_step;
	fs << "use_channel_weights" << use_channel_weights;
	fs << "weights_lr" << weights_lr;
	fs << "use_segmentation" << use_segmentation;
	fs << "histogram_bins" << histogram_bins;
	fs << "background_ratio" << background_ratio;
	fs << "histogram_lr" << histogram_lr;
}

} /* namespace dp_tracking */
} /* namespace cv */
