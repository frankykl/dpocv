#include "MOSSETracker.hpp"

namespace cv {
namespace dp_tracking {


const double eps = 0.00001;      // for normalization
const double rate = 0.2;         // learning rate
const double psrThreshold = 1.0;//5.7; // no detection, if PSR is smaller than this

#if 0

class DPTrackerMOSSEImpl: public CMOSSETracker {
public:
	DPTrackerMOSSEImpl();
	void read(const FileNode& fn) {};
	void write(FileStorage& fs) const {};

	bool initImpl(const Mat& image, const Rect2d& boundingBox);
	bool updateImpl(const Mat& image, Rect2d& boundingBox);
    void SetBoundingBox(const Mat& image, Rect2d boundingBox);

private:
    Mat DivDFTs(const Mat &src1, const Mat &src2) const;
    void Preprocess( Mat &window ) const;
    double Correlate( const Mat &image_sub, Point &delta_xy ) const;
    Mat RandWarp( const Mat& a ) const;

    Point2d _center; // center of the bounding box
    Size _size;      // size of the bounding box
    Mat _hanWin;     // Hanning window
    Mat _G;          // goal
    Mat _H, _A, _B;    // state
};

/*
 * Create MOSSE tracker without parameters.
 */
Ptr<CMOSSETracker> CMOSSETracker::create()
{
	return Ptr<DPTrackerMOSSEImpl>(new DPTrackerMOSSEImpl());
}


//  Element-wise division of complex numbers in src1 and src2
Mat DPTrackerMOSSEImpl::DivDFTs(const Mat& src1, const Mat& src2) const
{
    Mat c1[2], c2[2], a1, a2, s1, s2, denom, re, im;

    // split into re and im per src
    cv::split(src1, c1);
    cv::split(src2, c2);

    // (Re2*Re2 + Im2*Im2) = denom
    //   denom is same for both channels
    cv::multiply(c2[0], c2[0], s1);
    cv::multiply(c2[1], c2[1], s2);
    cv::add(s1, s2, denom);

    // (Re1*Re2 + Im1*Im1)/(Re2*Re2 + Im2*Im2) = Re
    cv::multiply(c1[0], c2[0], a1);
    cv::multiply(c1[1], c2[1], a2);
    cv::divide(a1 + a2, denom, re, 1.0 );

    // (Im1*Re2 - Re1*Im2)/(Re2*Re2 + Im2*Im2) = Im
    cv::multiply(c1[1], c2[0], a1);
    cv::multiply(c1[0], c2[1], a2);
    cv::divide(a1 + a2, denom, im, -1.0);

    // Merge Re and Im back into a complex matrix
    Mat dst, chn[] = {re, im};
    cv::merge(chn, 2, dst);
    return dst;
}

/*
 * Preprocessing.
 */
void DPTrackerMOSSEImpl::Preprocess(Mat& window) const
{
    window.convertTo(window, CV_32F);
    log(window + 1.0f, window);

    // normalize
    Scalar mean, StdDev;
    meanStdDev(window, mean, StdDev);
    window = (window - mean[0]) / (StdDev[0] + eps);

    // Gaussain weighting
    window = window.mul(_hanWin);
}

/*
 * Run correlation filter.
 */
double DPTrackerMOSSEImpl::Correlate(const Mat& image_sub, Point& delta_xy) const
{
    Mat IMAGE_SUB, RESPONSE, response;
    // filter in dft space
    dft(image_sub, IMAGE_SUB, DFT_COMPLEX_OUTPUT);
    mulSpectrums(IMAGE_SUB, _H, RESPONSE, 0, true);
    idft(RESPONSE, response, DFT_SCALE | DFT_REAL_OUTPUT);

    // update center position
    double maxVal; Point maxLoc;
    minMaxLoc(response, 0, &maxVal, 0, &maxLoc);
    delta_xy.x = maxLoc.x - int(response.size().width / 2);
    delta_xy.y = maxLoc.y - int(response.size().height / 2);

    // normalize response
    Scalar mean, std;
    meanStdDev(response, mean, std);
    return (maxVal - mean[0]) / (std[0] + eps); // PSR
}

/*
 * Random warping.
 */
Mat DPTrackerMOSSEImpl::RandWarp(const Mat& a) const
{
    cv::RNG rng(8031965);

    // random rotation
    double C = 0.1;
    double ang = rng.uniform(-C, C);
    double c = cos(ang), s = sin(ang);

    // affine warp matrix
    Mat_<float> W(2, 3);
    W << c + rng.uniform(-C, C), -s + rng.uniform(-C, C), 0,
            s + rng.uniform(-C, C),  c + rng.uniform(-C, C), 0;

    // random translation
    Mat_<float> center_warp(2, 1);
    center_warp << a.cols / 2, a.rows / 2;
    W.col(2) = center_warp - (W.colRange(0, 2)) * center_warp;

    // apply affine transform
    Mat warped;
    warpAffine(a, warped, W, a.size(), BORDER_REFLECT);
    return warped;
}

void DPTrackerMOSSEImpl::SetBoundingBox(const Mat& image, Rect2d boundingBox)
{
    if (_H.empty()) // not initialized
    return;

    Mat img;
    if (image.channels() == 1)
        img = image;
    else
        cvtColor(image, img, COLOR_BGR2GRAY);

    int w = getOptimalDFTSize(int(boundingBox.width));
    int h = getOptimalDFTSize(int(boundingBox.height));

    // Get the center position
    int x1 = int(floor((2 * boundingBox.x + boundingBox.width-w) / 2));
    int y1 = int(floor((2 * boundingBox.y + boundingBox.height-h) / 2));
    _center.x = x1 + (w) / 2;
    _center.y = y1 + (h) / 2;
    _size.width = w;
    _size.height = h;

    // create Hanning window
    Mat window;
    getRectSubPix(img, _size, _center, window);
    createHanningWindow(_hanWin, _size, CV_32F);

    // crop new image patch
    if (window.channels() != 1)
        cvtColor(window, window, COLOR_BGR2GRAY);
    Preprocess(window);

    // new state for A and B
    Mat F, A_new, B_new;
    dft(window, F, DFT_COMPLEX_OUTPUT);
    mulSpectrums(_G, F, A_new, 0, true);
    mulSpectrums(F, F, B_new, 0, true);

    // update A ,B, and H with learning rate
    _A = _A * (1 - rate) + A_new * rate;
    _B = _B * (1 - rate) + B_new * rate;
    _H = DivDFTs(_A, _B);
}

bool DPTrackerMOSSEImpl::initImpl(const Mat& image, const Rect2d& boundingBox)
{
    Mat img;
    if (image.channels() == 1)
        img = image;
    else
        cvtColor(image, img, COLOR_BGR2GRAY);

    int w = getOptimalDFTSize(int(boundingBox.width));
    int h = getOptimalDFTSize(int(boundingBox.height));

    // Get the center position
    int x1 = int(floor((2 * boundingBox.x + boundingBox.width-w) / 2));
    int y1 = int(floor((2 * boundingBox.y + boundingBox.height-h) / 2));
    _center.x = x1 + (w) / 2;
    _center.y = y1 + (h) / 2;
    _size.width = w;
    _size.height = h;

    // create Hanning window
    Mat window;
    getRectSubPix(img, _size, _center, window);
    createHanningWindow(_hanWin, _size, CV_32F);

    // initialize goal
    Mat g = Mat::zeros(_size, CV_32F);
    g.at<float>(h / 2, w / 2) = 1;
    GaussianBlur(g, g, Size(-1, -1), 2.0);
    double maxVal;
    minMaxLoc(g, 0, &maxVal);
    g = g / maxVal;
    dft(g, _G, DFT_COMPLEX_OUTPUT);

    // initial A, B and H
    _A = Mat::zeros(_G.size(), _G.type());
    _B = Mat::zeros(_G.size(), _G.type());
    for (int i = 0; i < 8; i++) {
        Mat window_warp = RandWarp(window);
        Preprocess(window_warp);

        Mat WINDOW_WARP, A_i, B_i;
        dft(window_warp, WINDOW_WARP, DFT_COMPLEX_OUTPUT);
        mulSpectrums(_G, WINDOW_WARP, A_i, 0, true);
        mulSpectrums(WINDOW_WARP, WINDOW_WARP, B_i, 0, true);
        _A += A_i;
        _B += B_i;
    }
    _H = DivDFTs(_A, _B);
    return true;
}

/*
 * Tracking in a frame.
 */
bool DPTrackerMOSSEImpl::updateImpl(const Mat& image, Rect2d& boundingBox)
{
    if (_H.empty()) // not initialized
        return false;

    // crop image patch with current center
    Mat image_sub;
    getRectSubPix(image, _size, _center, image_sub);

    // preprocessing
    if (image_sub.channels() != 1)
        cvtColor(image_sub, image_sub, COLOR_BGR2GRAY);
    Preprocess(image_sub);

    // run correlation filter
    Point delta_xy;
    double PSR = Correlate(image_sub, delta_xy);

    if (PSR > psrThreshold) {
		// update location
		_center.x += delta_xy.x;
		_center.y += delta_xy.y;

		// crop new image patch
		Mat img_sub_new;
		getRectSubPix(image, _size, _center, img_sub_new);
		if (img_sub_new.channels() != 1)
			cvtColor(img_sub_new, img_sub_new, COLOR_BGR2GRAY);
		Preprocess(img_sub_new);

		// new state for A and B
		Mat F, A_new, B_new;
		dft(img_sub_new, F, DFT_COMPLEX_OUTPUT);
		mulSpectrums(_G, F, A_new, 0, true);
		mulSpectrums(F, F, B_new, 0, true);

		// update A ,B, and H with learning rate
		_A = _A * (1 - rate) + A_new * rate;
		_B = _B * (1 - rate) + B_new * rate;
		_H = DivDFTs(_A, _B);

		// return tracked rect
		double x = _center.x, y = _center.y;
		int w = _size.width, h = _size.height;
		boundingBox = Rect2d(Point2d(x - 0.5 * w, y - 0.5 * h), Point2d(x + 0.5 * w, y + 0.5 * h));
		return true;
    } else
    	return false;
}

#endif

} // dp_tracking
} // cv
