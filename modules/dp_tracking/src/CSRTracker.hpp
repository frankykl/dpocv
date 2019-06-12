#ifndef __CSRTRACKER_HPP__
#define __CSRTRACKER_HPP__

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/tracking.hpp"
#include <vector>

using namespace std;

namespace cv {
namespace dp_tracking {

// Color names array.
extern const float DPColorNames[][10];

// Inline functions.
inline int Modulo(int a, int b)
{
	// function calculates the module of two numbers and it takes into account also negative numbers
	return ((a % b) + b) % b;
}

inline double KernelEpan(double x)
{
	return (x <= 1) ? (2.0 / 3.14) * (1 - x) : 0;
}

// Function prototypes
Mat CircularShift(Mat matrix, int dx, int dy);
Mat GaussianShapedLabels(const float sigma, const int w, const int h);
std::vector<Mat> FourierTransformFeatures(const std::vector<Mat> &M);
Mat DivideComplexMatrices(const Mat &A, const Mat &B);
Mat GetSubwindow(const Mat &image, const Point2f center, const int w, const int h,
		Rect *valid_pixels = NULL);

float SubpixelPeak(const Mat &response, const std::string &s, const Point2f &p);
double GetMax(const Mat &m);
double GetMin(const Mat &m);

Mat GetHanningWindow(Size sz);
Mat GetKaiserWindow(Size sz, float alpha);
Mat GetChebyshevWindow(Size sz, float attenuation);

std::vector<Mat> GetFeaturesRGB(const Mat &patch, const Size &output_size);
std::vector<Mat> GetFeaturesHOG(const Mat &im, const int bin_size);
std::vector<Mat> GetFeaturesCN(const Mat &im, const Size &output_size);

Mat BGR2HSV(const Mat &img);

/*
 * Histogram class.
 */
class Histogram {
public:
	int m_numBinsPerDim;
	int m_numDim;

	Histogram() : m_numBinsPerDim(0), m_numDim(0){}
	Histogram(int numDimensions, int numBinsPerDimension = 8);
	void ExtractForegroundHistogram(std::vector<cv::Mat> & imgChannels, cv::Mat weights,
			bool useMatWeights, int x1, int y1, int x2, int y2);
	void ExtractBackGroundHistogram(std::vector<cv::Mat> & imgChannels, int x1, int y1, int x2,
			int y2, int outer_x1, int outer_y1, int outer_x2, int outer_y2);
	cv::Mat BackProject(std::vector<cv::Mat> & imgChannels);
	std::vector<double> GetHistogramVector();
	void SetHistogramVector(double *vector);

private:
	int p_size;
	std::vector<double> p_bins;
	std::vector<int> p_dimIdCoef;

	inline double KernelProfileEpanechnikov(double x)
	{
		return (x <= 1) ? (2.0 / CV_PI) * (1 - x) : 0;
	}
};

/*
 * Segment class.
 */
class Segment {
public:
	static std::pair<cv::Mat, cv::Mat> ComputePosteriors(std::vector<cv::Mat> & imgChannels, int x1,
			int y1, int x2, int y2, cv::Mat weights, cv::Mat fgPrior, cv::Mat bgPrior,
			const Histogram &fgHistPrior, int numBinsPerChannel = 16);
	static std::pair<cv::Mat, cv::Mat> ComputePosteriors2(std::vector<cv::Mat> & imgChannels,
			int x1, int y1, int x2, int y2, double p_b, cv::Mat fgPrior, cv::Mat bgPrior,
			Histogram hist_target, Histogram hist_background);
	static std::pair<cv::Mat, cv::Mat> ComputePosteriors2(std::vector<cv::Mat> &imgChannels,
			cv::Mat fgPrior, cv::Mat bgPrior, Histogram hist_target, Histogram hist_background);

private:
	static std::pair<cv::Mat, cv::Mat> GetRegularizedSegmentation(cv::Mat & prob_o,
			cv::Mat & prob_b, cv::Mat &prior_o, cv::Mat &prior_b);

	inline static double Gaussian(double x2, double y2, double std2)
	{
		return exp(-(x2 + y2) / (2 * std2)) / (2 * CV_PI * std2);
	}
};

/*
 * DSST class.
 */
class DSST {
public:
	DSST() {};
	DSST(const Mat &image, Rect2f bounding_box, Size2f template_size, int numberOfScales,
			float scaleStep, float maxModelArea, float sigmaFactor, float scaleLearnRate);
	~DSST();
	void Update(const Mat &image, const Point2f objectCenter);
	float GetScale(const Mat &image, const Point2f objecCenter);

private:
	Size scale_model_sz;
	Mat ys;
	Mat ysf;
	Mat scale_window;
	std::vector<float> scale_factors;
	Mat sf_num;
	Mat sf_den;
	float scale_sigma;
	float min_scale_factor;
	float max_scale_factor;
	float current_scale_factor;
	int scales_count;
	float scale_step;
	float max_model_area;
	float sigma_factor;
	float learn_rate;
	Size original_targ_sz;

	Mat GetScaleFeatures(Mat img, Point2f pos, Size2f base_target_sz, float current_scale,
			std::vector<float> &scale_factors, Mat scale_window, Size scale_model_sz);
};

/*
 * CSRT tracker.
 */
class DPTrackerCSRT : public Tracker {
public:
	struct Params
	{
		Params();
		void read(const FileNode& /*fn*/);
		void write(cv::FileStorage& fs) const;

		bool use_hog;
		bool use_color_names;
		bool use_gray;
		bool use_rgb;
		bool use_channel_weights;
		bool use_segmentation;

		// window functions
		std::string window_function;
		float kaiser_alpha;
		float cheb_attenuation;

		float template_size;
		float gsl_sigma;
		float hog_orientations;
		float hog_clip;
		float padding;
		float filter_lr;
		float weights_lr;
		int num_hog_channels_used;
		int admm_iterations;
		int histogram_bins;
		float histogram_lr;
		int background_ratio;
		int number_of_scales;
		float scale_sigma_factor;
		float scale_model_max_area;
		float scale_lr;
		float scale_step;
	};

	virtual ~DPTrackerCSRT() {}

	static Ptr<DPTrackerCSRT> create(const DPTrackerCSRT::Params &parameters);
	static Ptr<DPTrackerCSRT> create();

	virtual void setInitialMask(const Mat mask) = 0;
	virtual bool initImpl(const Mat& image, const Rect2d& boundingBox) = 0;
	virtual bool updateImpl(const Mat& image, Rect2d& boundingBox) = 0;
	virtual double update(const Mat& image, Rect2d& boundingBox) = 0;

	/*
	 * Override init function of parent class.
	 */
	bool init(InputArray image, const Rect2d& boundingBox)
	{
		if (image.empty())
			return false;

		sampler = Ptr<TrackerSampler>(new TrackerSampler());
		featureSet = Ptr<TrackerFeatureSet>(new TrackerFeatureSet());
		model = Ptr<TrackerModel>();

		isInit = initImpl(image.getMat(), boundingBox);
		if (model == 0) {
			CV_Error(-1, "The model is not initialized");
			return false;
		}

		return isInit;
	}

	double confidence;
};

} // namespace dp_tracking
} // namespace cv

#endif /* __TRACKER_HPP__ */
