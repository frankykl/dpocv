#ifndef _FEATURE_SPACE_H_
#define _FEATURE_SPACE_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*
 * Sample class.
 */
class Sample 
{
public:
	// constructor
	Sample(int size = 0) 
	{
		x = NULL;
		y = 0;
		w = 0.0;
		locX = -1;
		locY = -1;
		vecSize = size;
		if (vecSize > 0) {
			x = new int[vecSize];
		}
	}

	// destructor
	~Sample()
	{
		if (x != NULL) {
			delete[] x;
		}
	}

	// members
	Mat imagePatch; // sample image patch
	int* x;			// sample feature vector
	int y;			// sample label
	double w;		// sample weight
	int locX;		// sample location x
	int locY;		// sample location y
	int vecSize;	// x vector size
};

/*
 * Sample data set.
 */
typedef vector<Sample*> SampleSet;


/*
 * Feature space class.
 */
class CFeatureSpace
{
public:
	virtual ~CFeatureSpace() {};
	CFeatureSpace() {};

	virtual void ComputeSampleFeatures(Sample& sample) = 0;
	virtual int GetFeatureDimension() = 0;
};


/*
 * Haar feature.
 */
const int HAAR_SIZE = 24;
const int HAAR_DC = 128;

typedef struct HaarFeature {
	Rect		rect[3];	// rectangles
	int			w[3];		// weights
	const int*	p[3][4];	// pointers
	int			thr;		// threshold

	HaarFeature();
	~HaarFeature();
	void Load(FILE* pf);
	void Store(FILE* pf);
} HaarFeature;

/*
 * Haar feature space.
 */
class CHaarSpace : public CFeatureSpace
{
public:
	~CHaarSpace();
	CHaarSpace(FILE* pf);

	bool Load(FILE* pf);
	void Store(FILE* pf);
	int InitFeatures(int step);
	void SetImage(const Mat &image, Size origWinSize);
	void ComputeFeatures(Point& point, int* x);
	int GetFeatureDimension() {return m_iNumFeatures;}
	virtual void ComputeSampleFeatures(Sample& sample);

	void SetFeature(int x0, int y0, int width0, int height0, int w0, 
		int x1, int y1, int width1, int height1, int w1, 
		int x2 = 0, int y2 = 0, int width2 = 0, int height2 = 0, int w2 = 0);

	int				m_iNumFeatures;		// number of features
	HaarFeature*	m_pFeatures;		// pointer of features
	vector<HaarFeature>	m_features;		// features
	Mat				m_integral;			// integral image
	Mat				m_data;				// integral image data
	int*			m_pValue;			// feature values
};

/*
 * HOG feature space.
 */
class CHOGSpace : public CFeatureSpace
{
public:
	~CHOGSpace();
	CHOGSpace();

	int GetFeatureDimension() {return _iNumFeatures;}
	void ComputeSampleFeatures(Sample& sample);

private:
	int _cellSize;
	int _channels;
	int	_iNumFeatures;		// number of features
};


#endif /* _FEATURE_SPACE_H_ */
