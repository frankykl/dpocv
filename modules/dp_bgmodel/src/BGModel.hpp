#ifndef __BGMODEL_HPP__
#define __BGMODEL_HPP__

#include "opencv2/core.hpp"

using namespace cv;

namespace cv {
namespace dp_bgmodel {

struct GaussBGStatModel2Params {
	// image info
	int nWidth;   // image width
	int nHeight;  // image height
	int nND;      // number of data dimensions (image channels)

	bool bPostFiltering; // defult 1 - do postfiltering - will make shadow detection results also give value 255
	double minArea;      // for postfiltering

	bool bInit;   // default 1, faster updates at start

	/*
	 * very important parameters - things you will change
	 */

	// alpha - speed of update - if the time interval you want to average over is T
	// set alpha = 1/T. It is also useful at start to make T slowly increase
	// from 1 until the desired T
	float fAlphaT;

	// Tb - threshold on the squared Mahalan. dist. to decide if it is well described
	// by the background model or not. Related to Cthr from the paper.
	// This does not influence the update of the background. A typical value could be
	// 4 times of sigma, result as Tb = 4 * 4 = 16;
	float fTb;

	/*
	 * less important parameters - things you might change but be careful
	 */

	// Tg - threshold on the squared Mahalan. dist. to decide when a sample is close
	// to the existing components. If it is not close to any a new component will be
	// generated. I use 3 sigma => Tg = 3 * 3 = 9.
	// Smaller Tg leads to more generated components and bigger Tg might lead to small
	// number of components but they can grow too large
	float fTg;

	// TB - threshold when the component becomes significant enough to be included into
	// the background model. It is the TB = 1 - cf from the paper. So I use cf = 0.1 => TB = 0.
	// For alpha = 0.001 it means that the mode should exist for approximately 105 frames
	// before it is considered foreground
	float fTB;  // 1 - cf from the paper

	// Initial standard deviation for the newly generated components.
	// It will will influence the speed of adaptation. A good guess should be made.
	// A simple way is to estimate the typical standard deviation from the images.
	// I used here 10 as a reasonable value
	float fVarInit;
	float fVarMax;
	float fVarMin;

	// This is related to the number of samples needed to accept that a component
	// actually exists. We use CT=0.05 of all the samples. By setting CT=0 you get
	// the standard Stauffer & Grimson algorithm (maybe not exact but very similar)
	float fCT;  // CT - complexity reduction prior

	//even less important parameters
	int nM;    // max number of modes (constant), 4 is usually enough

	// shadow detection parameters
	bool bShadowDetection;    // default 1 - do shadow detection
	unsigned char nShadowDetection; // do shadow detection - insert this value as the detection result

	// Tau - shadow threshold. The shadow is detected if the pixel is darker
	// version of the background. Tau is a threshold on how much darker the shadow can be.
	// Tau = 0.5 means that if pixel is more than 2 times darker then it is not shadow
	// See: Prati, Mikic, Trivedi, Cucchiarra, "Detecting Moving Shadows...", IEEE PAMI, 2003.
	float fTau;
};

struct GMM {
	float weight;
	float variance;
};

class BGModelMOG {
public:
	//! the default constructor
	BGModelMOG();

	//! the full constructor that takes the length of the history,
	// the number of gaussian mixtures, the background ratio parameter and the noise strength
	BGModelMOG(int _history, float _varThreshold, bool _bShadowDetection = true);

	//! the destructor
	~BGModelMOG();

	//! the update operator
	void apply(InputArray image, OutputArray fgmask, Mat *mask = NULL);

	//! computes a background image which are the mean of all background gaussians
	void getBackgroundImage(OutputArray backgroundImage) const;

	//! re-initiaization method
	void initialize(Size _frameSize, int _frameType)
	{
		frameSize = _frameSize;
		frameType = _frameType;
		nframes = 0;

		int nchannels = CV_MAT_CN(frameType);
		CV_Assert(nchannels <= CV_CN_MAX);
		CV_Assert(nmixtures <= 255);

		// for each Gaussian mixture of each pixel bg model we store:
		// the mixture weight (w), the mean (nchannels values) and the covariance
		bgmodel.create(1, frameSize.height * frameSize.width * nmixtures * (2 + nchannels), CV_32F);

		// make the array for keeping track of the used modes per pixel - all zeros at start
		bgmodelUsedModes.create(frameSize, CV_8U);
		bgmodelUsedModes = Scalar::all(0);
	}

	int getHistory() const
	{
		return history;
	}
	void setHistory(int _nframes)
	{
		history = _nframes;
	}

	int getNMixtures() const
	{
		return nmixtures;
	}
	void setNMixtures(int nmix)
	{
		nmixtures = nmix;
	}

	double getBackgroundRatio() const
	{
		return backgroundRatio;
	}
	void setBackgroundRatio(double _backgroundRatio)
	{
		backgroundRatio = (float) _backgroundRatio;
	}

	double getVarThreshold() const
	{
		return varThreshold;
	}
	void setVarThreshold(double _varThreshold)
	{
		varThreshold = _varThreshold;
	}

	double getVarThresholdGen() const
	{
		return varThresholdGen;
	}
	void setVarThresholdGen(double _varThresholdGen)
	{
		varThresholdGen = (float) _varThresholdGen;
	}

	double getVarInit() const
	{
		return fVarInit;
	}
	void setVarInit(double varInit)
	{
		fVarInit = (float) varInit;
	}

	double getVarMin() const
	{
		return fVarMin;
	}
	void setVarMin(double varMin)
	{
		fVarMin = (float) varMin;
	}

	double getVarMax() const
	{
		return fVarMax;
	}
	void setVarMax(double varMax)
	{
		fVarMax = (float) varMax;
	}

	double getComplexityReductionThreshold() const
	{
		return fCT;
	}
	void setComplexityReductionThreshold(double ct)
	{
		fCT = (float) ct;
	}

	bool getDetectShadows() const
	{
		return bShadowDetection;
	}
	void setDetectShadows(bool detectshadows)
	{
		if ((bShadowDetection && detectshadows) || (!bShadowDetection && !detectshadows))
			return;
		bShadowDetection = detectshadows;
	}

	int getShadowValue() const
	{
		return nShadowDetection;
	}
	void setShadowValue(int value)
	{
		nShadowDetection = (uchar) value;
	}

	double getShadowThreshold() const
	{
		return fTau;
	}
	void setShadowThreshold(double value)
	{
		fTau = (float) value;
	}

#if 0
	void write(FileStorage& fs) const
	{
		writeFormat(fs);
		fs << "name" << name_
		<< "history" << history
		<< "nmixtures" << nmixtures
		<< "backgroundRatio" << backgroundRatio
		<< "varThreshold" << varThreshold
		<< "varThresholdGen" << varThresholdGen
		<< "varInit" << fVarInit
		<< "varMin" << fVarMin
		<< "varMax" << fVarMax
		<< "complexityReductionThreshold" << fCT
		<< "detectShadows" << (int)bShadowDetection
		<< "shadowValue" << (int)nShadowDetection
		<< "shadowThreshold" << fTau;
	}

	virtual void read(const FileNode& fn)
	{
		CV_Assert( (String)fn["name"] == name_ );
		history = (int)fn["history"];
		nmixtures = (int)fn["nmixtures"];
		backgroundRatio = (float)fn["backgroundRatio"];
		varThreshold = (double)fn["varThreshold"];
		varThresholdGen = (float)fn["varThresholdGen"];
		fVarInit = (float)fn["varInit"];
		fVarMin = (float)fn["varMin"];
		fVarMax = (float)fn["varMax"];
		fCT = (float)fn["complexityReductionThreshold"];
		bShadowDetection = (int)fn["detectShadows"] != 0;
		nShadowDetection = saturate_cast<uchar>((int)fn["shadowValue"]);
		fTau = (float)fn["shadowThreshold"];
	}
#endif

protected:
	Size frameSize;
	int frameType;
	Mat bgmodel;
	Mat bgmodelUsedModes;   // keep track of number of modes per pixel

	 // Disable motion mask for certain number of frames to learn background
	int initFrames;

	int nframes;
	int history;
	//! here it is the maximum allowed number of mixture components.
	//! Actual number is determined dynamically per pixel
	int nmixtures;

	// Threshold on the squared Mahalanobis distance to decide if it is well described
	// by the background model or not. Related to Cthr from the paper.
	// This does not influence the update of the background. A typical value could be
	// 4 time of sigma, and that is varThreshold=4*4=16; Corresponds to Tb in the paper.
	double varThreshold;

	/*
	 * less important parameters - things you might change but be carefull
	 */

	// Corresponds to fTB=1-cf from the paper
	// TB - threshold when the component becomes significant enough to be included into
	// the background model. It is the TB=1-cf from the paper. So I use cf=0.1 => TB=0.
	// For alpha=0.001 it means that the mode should exist for approximately 105 frames before
	// it is considered foreground
	float backgroundRatio;

	// Correspondts to Tg - threshold on the squared Mahalan. dist. to decide
	// when a sample is close to the existing components. If it is not close
	// to any a new component will be generated. I use 3 sigma => Tg=3*3=9.
	// Smaller Tg leads to more generated components and higher Tg might make
	// lead to small number of components but they can grow too large
	float varThresholdGen;

	// Initial variance  for the newly generated components.
	// It will will influence the speed of adaptation. A good guess should be made.
	// A simple way is to estimate the typical standard deviation from the images.
	// I used here 10 as a reasonable value
	// min and max can be used to further control the variance
	float fVarInit;
	float fVarMin;
	float fVarMax;

	// This is related to the number of samples needed to accept that a component
	// actually exists. We use CT=0.05 of all the samples. By setting CT=0 you get
	// the standard Stauffer&Grimson algorithm (maybe not exact but very similar)
	float fCT;    //CT - complexity reduction prior

	// shadow detection parameters
	bool bShadowDetection;    // default 1 - do shadow detection
	 // do shadow detection - insert this value as the detection result - 127 default value
	unsigned char nShadowDetection;
	// Tau - shadow threshold. The shadow is detected if the pixel is darker version
	// of the background. Tau is a threshold on how much darker the shadow can be.
	// Tau = 0.5 means that if pixel is more than 2 times darker then it is not shadow
	// See: Prati, Mikic, Trivedi, Cucchiarra, "Detecting Moving Shadows...", IEEE PAMI, 2003.
	float fTau;

	String name_;

	template<typename T, int CN>
	void getBackgroundImage_intern(OutputArray backgroundImage) const;
};

}
}

#endif /* __BGMODEL_HPP__ */
