#include "BGModel.hpp"
#include <cstdio>
#include <sstream>
#include <valarray>

using namespace cv;

namespace cv {
namespace dp_bgmodel {

// default parameters of gaussian background detection algorithm
static const int defaultInitFrames = 0; // Initialization frames
static const int defaultHistory2 = 500; // Learning rate; alpha = 1/defaultHistory2
static const float defaultVarThreshold2 = 4.0f * 4.0f;
static const int defaultNMixtures2 = 3; // maximal number of Gaussians in mixture
static const float defaultBackgroundRatio2 = 0.9f; // threshold sum of weights for background test
static const float defaultVarThresholdGen2 = 3.0f * 3.0f;
static const float defaultVarInit2 = 15.0f; // initial variance for new components
static const float defaultVarMax2 = 5 * defaultVarInit2;
static const float defaultVarMin2 = 4.0f;

// additional parameters
static const float defaultfCT2 = 0.05f; // complexity reduction prior constant 0 - no reduction of number of components
static const unsigned char defaultnShadowDetection2 = (unsigned char) 127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
static const float defaultfTau = 0.5f; // Tau - shadow threshold, see the paper for explanation

BGModelMOG::~BGModelMOG()
{
}

BGModelMOG::BGModelMOG()
{
	frameSize = Size(0, 0);
	frameType = 0;

	initFrames = defaultInitFrames;
	nframes = 0;
	history = defaultHistory2;
	varThreshold = defaultVarThreshold2;
	bShadowDetection = 1;

	nmixtures = defaultNMixtures2;
	backgroundRatio = defaultBackgroundRatio2;
	fVarInit = defaultVarInit2;
	fVarMax = defaultVarMax2;
	fVarMin = defaultVarMin2;

	varThresholdGen = defaultVarThresholdGen2;
	fCT = defaultfCT2;
	nShadowDetection = defaultnShadowDetection2;
	fTau = defaultfTau;
}

BGModelMOG::BGModelMOG(int _history, float _varThreshold, bool _bShadowDetection)
{
	frameSize = Size(0, 0);
	frameType = 0;

	nframes = 0;
	history = _history > 0 ? _history : defaultHistory2;
	varThreshold = (_varThreshold > 0) ? _varThreshold : defaultVarThreshold2;
	bShadowDetection = _bShadowDetection;

	nmixtures = defaultNMixtures2;
	backgroundRatio = defaultBackgroundRatio2;
	fVarInit = defaultVarInit2;
	fVarMax = defaultVarMax2;
	fVarMin = defaultVarMin2;

	varThresholdGen = defaultVarThresholdGen2;
	fCT = defaultfCT2;
	nShadowDetection = defaultnShadowDetection2;
	fTau = defaultfTau;
	name_ = "BackgroundSubtractor.MOG2";
}

/*
 * Internal template function to get background image.
 */
template<typename T, int CN>
void BGModelMOG::getBackgroundImage_intern(OutputArray backgroundImage) const
{
	Mat meanBackground(frameSize, frameType, Scalar::all(0));
	int firstGaussianIdx = 0;
	const GMM* gmm = bgmodel.ptr<GMM>();
	const float* mean = reinterpret_cast<const float*>(gmm
			+ frameSize.width * frameSize.height * nmixtures);
	Vec<float, CN> meanVal(0.f);
	// loop for rows
	for (int row = 0; row < meanBackground.rows; row++) {
		// loop for columns
		for (int col = 0; col < meanBackground.cols; col++) {
			int nmodes = bgmodelUsedModes.at<uchar>(row, col);
			float totalWeight = 0.f;
			// loop for mixture modes
			// the output background value is the weighted sum of the means of all modes
			for (int gaussianIdx = firstGaussianIdx; gaussianIdx < firstGaussianIdx + nmodes;
					gaussianIdx++) {
				GMM gaussian = gmm[gaussianIdx];
				size_t meanPosition = gaussianIdx * CN;
				for (int chn = 0; chn < CN; chn++) {
					meanVal(chn) += gaussian.weight * mean[meanPosition + chn];
				}
				totalWeight += gaussian.weight;

				if (totalWeight > backgroundRatio)
					break;
			}
			float invWeight = 1.f / totalWeight;

			meanBackground.at<Vec<T, CN> >(row, col) = Vec<T, CN>(meanVal * invWeight);
			meanVal = 0.f;

			firstGaussianIdx += nmixtures;
		}
	}
	meanBackground.copyTo(backgroundImage);
}

/*
 * Get background image.
 */
void BGModelMOG::getBackgroundImage(OutputArray backgroundImage) const
{
	CV_Assert(frameType == CV_8UC1 || frameType == CV_8UC3
			|| frameType == CV_32FC1 || frameType == CV_32FC3);

	switch (frameType) {
	case CV_8UC1:
		getBackgroundImage_intern<uchar, 1>(backgroundImage);
		break;
	case CV_8UC3:
		getBackgroundImage_intern<uchar, 3>(backgroundImage);
		break;
	case CV_32FC1:
		getBackgroundImage_intern<float, 1>(backgroundImage);
		break;
	case CV_32FC3:
		getBackgroundImage_intern<float, 3>(backgroundImage);
		break;
	}
}

/*
 * Detect shadow based on GMM.
 */
inline bool detectShadowGMM(const float* data, int nchannels, int nmodes, const GMM* gmm,
		const float* mean, float Tb, float TB, float tau)
{
	float tWeight = 0;

	// check all the components  marked as background:
	for (int mode = 0; mode < nmodes; mode++, mean += nchannels) {
		GMM g = gmm[mode];

		float numerator = 0.0f;
		float denominator = 0.0f;
		for (int c = 0; c < nchannels; c++) {
			numerator += data[c] * mean[c];
			denominator += mean[c] * mean[c];
		}

		// no division by zero allowed
		if (denominator == 0)
			return false;

		// if tau < a < 1 then also check the color distortion
		if (numerator <= denominator && numerator >= tau * denominator) {
			float a = numerator / denominator;
			float dist2a = 0.0f;

			for (int c = 0; c < nchannels; c++) {
				float dD = a * mean[c] - data[c];
				dist2a += dD * dD;
			}

			if (dist2a < Tb * g.variance * a * a)
				return true;
		};

		tWeight += g.weight;
		if (tWeight > TB)
			return false;
	};
	return false;
}

/*
 * MOG implementation class.
 */
class MOG2Invoker: public ParallelLoopBody {
public:
	// member variables
	const Mat* src;
	Mat* dst;
	GMM* gmm0;
	float* mean0;
	uchar* modesUsed0;
	int nmixtures;
	float alphaT, alphaT1, Tb, TB, Tg;
	float varInit, varMin, varMax, prune, tau;
	bool detectShadows;
	uchar shadowVal;
	Mat *motionMask;

	// constructor
	MOG2Invoker(const Mat& _src, Mat& _dst, GMM* _gmm, float* _mean, uchar* _modesUsed,
			int _nmixtures, float _alphaT, float _Tb, float _TB, float _Tg, float _varInit,
			float _varMin, float _varMax, float _prune, float _tau, bool _detectShadows,
			uchar _shadowVal, Mat *mask)
	{
		src = &_src;
		dst = &_dst;
		gmm0 = _gmm;
		mean0 = _mean;
		modesUsed0 = _modesUsed;
		nmixtures = _nmixtures;
		alphaT = _alphaT;
		alphaT1 = alphaT / 10.0f;  // moving pixel learning rate
		Tb = _Tb;
		TB = _TB;
		Tg = _Tg;
		varInit = _varInit;
		varMin = MIN(_varMin, _varMax);
		varMax = MAX(_varMin, _varMax);
		prune = _prune;
		tau = _tau;
		detectShadows = _detectShadows;
		shadowVal = _shadowVal;
		motionMask = mask;
	}

	// operation
	void operator()(const Range& range) const
	{
		int y0 = range.start, y1 = range.end;
		int ncols = src->cols, nchannels = src->channels();
		AutoBuffer<float> buf(src->cols * nchannels);
		float alpha1 = 1.f - alphaT;
		float dData[CV_CN_MAX];

		// loop for image height
		for (int y = y0; y < y1; y++) {
			// convert source into float
			const float* data = buf.data();
			if (src->depth() != CV_32F)
				src->row(y).convertTo(Mat(1, ncols, CV_32FC(nchannels), (void*) data), CV_32F);
			else
				data = src->ptr<float>(y);

			// motion mask
			const unsigned char *pMask = NULL;
			if (motionMask != NULL) {
				pMask = motionMask->ptr<unsigned char>(y);
			}

			// buffer pointers
			float* mean = mean0 + ncols * nmixtures * nchannels * y;
			GMM* gmm = gmm0 + ncols * nmixtures * y;
			uchar* modesUsed = modesUsed0 + ncols * y;
			uchar* mask = dst->ptr(y);

			// loop for pixels
			for (int x = 0; x < ncols; x++, data += nchannels, gmm += nmixtures, mean += nmixtures * nchannels) {
				// calculate distances to the modes (+ sort)
				// here we need to go in descending order!!!
				bool background = false; //return true if the pixel is classified as background

				// internal
				bool fitsPDF = false;      // if it remains zero a new GMM mode will be added
				int nmodes = modesUsed[x]; // number of modes in GMM for current pixel
				float totalWeight = 0.f;
				float* mean_m = mean;
				bool movingPixel = pMask && pMask[x] > 0;
				float alpha = movingPixel ? alphaT1 : alphaT;
				alpha1 = 1.f - alpha;

				// go through all modes
				for (int mode = 0; mode < nmodes; mode++, mean_m += nchannels) {
					float weight = alpha1 * gmm[mode].weight + prune; //need only weight if fit is found
					int swap_count = 0;

					// fit not found yet
					if (!fitsPDF) {
						// check if it belongs to some of the remaining modes
						float var = gmm[mode].variance;

						// calculate difference and distance
						float dist2;
						if (nchannels == 3) {
							dData[0] = mean_m[0] - data[0];
							dData[1] = mean_m[1] - data[1];
							dData[2] = mean_m[2] - data[2];
							dist2 = dData[0] * dData[0] + dData[1] * dData[1] + dData[2] * dData[2];
						} else {
							dist2 = 0.f;
							for (int c = 0; c < nchannels; c++) {
								dData[c] = mean_m[c] - data[c];
								dist2 += dData[c] * dData[c];
							}
						}

						// check distance with variance threshold
						// why check totalWeight ?
						if (totalWeight < TB && dist2 < Tb * var)
							background = true;

						// update the Gaussian component if it fits
						if (dist2 < Tg * var) {
							fitsPDF = true;

							// update the distribution
							// update weight
							weight += alpha;
							float k = alpha / weight;

							// update mean
							for (int c = 0; c < nchannels; c++)
								mean_m[c] -= k * dData[c];

							// update variance
							float varnew = var + k * (dist2 - var);
							// limit the variance
							varnew = MAX(varnew, varMin);
							varnew = MIN(varnew, varMax);
							gmm[mode].variance = varnew;

							// sort all Gaussian components by weight
							// all other weights are at the same place and
							// only the matched (iModes) is higher -> just find the new place for it
							for (int i = mode; i > 0; i--) {
								// terminate condition
								if (weight < gmm[i - 1].weight)
									break;

								// swap one up
								swap_count++;
								// swap GMM
								std::swap(gmm[i], gmm[i - 1]);
								// swap mean value
								for (int c = 0; c < nchannels; c++)
									std::swap(mean[i * nchannels + c], mean[(i - 1) * nchannels + c]);
							}
						}
					} // if (!fitsPDF)

					// prune this component if its weight is too small
					if (weight < -prune) {
						weight = 0.0;
						nmodes--;
					}

					gmm[mode - swap_count].weight = weight;  //update weight by the calculated value
					totalWeight += weight;
				} // end of mode loop

				// normalize weights
				totalWeight = 1.f / totalWeight;
				for (int mode = 0; mode < nmodes; mode++) {
					gmm[mode].weight *= totalWeight;
				}

				// make new mode if needed and exit
				if (!fitsPDF && alpha > 0.f) {
					// replace the weakest or add a new one
					int mode = nmodes == nmixtures ? nmixtures - 1 : nmodes++;

					if (nmodes == 1)
						gmm[mode].weight = 1.f;
					else {
						gmm[mode].weight = alpha;
						// re-normalize all other weights
						for (int i = 0; i < nmodes - 1; i++)
							gmm[i].weight *= alpha1;
					}

					// initialize Gaussian component
					for (int c = 0; c < nchannels; c++)
						mean[mode * nchannels + c] = data[c];
					gmm[mode].variance = varInit;

					// sort to find the new place for it
					for (int i = nmodes - 1; i > 0; i--) {
						// terminate condition
						if (alpha < gmm[i - 1].weight)
							break;

						// swap one up
						std::swap(gmm[i], gmm[i - 1]);
						for (int c = 0; c < nchannels; c++)
							std::swap(mean[i * nchannels + c], mean[(i - 1) * nchannels + c]);
					}
				}

				// set the number of modes
				modesUsed[x] = uchar(nmodes);
				mask[x] = background ? 0 : detectShadows
						&& detectShadowGMM(data, nchannels, nmodes, gmm, mean, Tb, TB, tau) ? shadowVal : 255;
			} // end of x loop
		} // end of y loop
	}
};

/*
 * Run BG model for one frame.
 */
void BGModelMOG::apply(InputArray _image, OutputArray _fgmask, Mat *mask)
{

	bool needToInitialize = nframes == 0 || _image.size() != frameSize
			|| _image.type() != frameType;

	if (needToInitialize)
		initialize(_image.size(), _image.type());

	Mat image = _image.getMat();

	// create mask image
	_fgmask.create(image.size(), CV_8U);
	Mat fgmask = _fgmask.getMat();

	++nframes;

	double learningRate = 0.02;//1. / std::min(2 * nframes, history);
	CV_Assert(learningRate >= 0);
	//printf("%d, %f\n", nframes, learningRate);

	/*
	 * bgmodel is a big array contains the GMM models and means of each pixel
	 * bgmodel's type is float, its size is: frameSize.height * frameSize.width * nmixtures * (2 + nchannels)
	 * bgmode's layout:
	 *   - GMM: frameHeight * frameWidth * numMixtures * 2 (mean + variance)
	 *   - Pixel means: frameHeight * frameWidth * numMixtures * numChannels
	 */
	parallel_for_(Range(0, image.rows),
			MOG2Invoker(image, fgmask, bgmodel.ptr<GMM>(), // pointer of GMM models
					(float*)(bgmodel.ptr() + sizeof(GMM) * nmixtures * image.rows * image.cols), // pointer of pixel means
					bgmodelUsedModes.ptr(), nmixtures, (float)learningRate, (float)varThreshold,
					backgroundRatio, varThresholdGen, fVarInit, fVarMin, fVarMax,
					float(-learningRate * fCT), fTau, bShadowDetection, nShadowDetection,
					nframes < initFrames ? NULL : mask),
			image.total() / (double) (1 << 16));
}

} // end of namespace
}
