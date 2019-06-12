#include "opencv2/opencv.hpp"
#include "FeatureSpace.h"


/*******************************************************************************
	Haar feature implementation.
*******************************************************************************/

/*******************************************************************************
	Destructor.
*******************************************************************************/
HaarFeature::~HaarFeature()
{
}

/*******************************************************************************
	Constructor.
*******************************************************************************/
HaarFeature::HaarFeature()
{
	memset(rect, 0, sizeof rect);
	w[0] = w[1] = w[2] = 0;
	memset(p, 0, sizeof(p));
	thr = 0;
}

/*******************************************************************************
	Load from file.
*******************************************************************************/
void HaarFeature::Load(FILE* pf)
{
	fread(rect, sizeof(Rect), 3, pf);
	fread(w, sizeof(int), 3, pf);
	fread(&thr, sizeof(int), 1, pf);
}

/*******************************************************************************
	Store into file.
*******************************************************************************/
void HaarFeature::Store(FILE* pf)
{
	fwrite(rect, sizeof(Rect), 3, pf);
	fwrite(w, sizeof(int), 3, pf);
	fwrite(&thr, sizeof(int), 1, pf);
}


/*******************************************************************************
	Haar space implementation.
*******************************************************************************/

/*******************************************************************************
	Destructor.
*******************************************************************************/
CHaarSpace::~CHaarSpace()
{
	if (m_pValue != NULL) {
		delete m_pValue;
	}
}

/*******************************************************************************
	Constructor.
*******************************************************************************/
CHaarSpace::CHaarSpace(FILE* pf) :
	m_pFeatures(NULL),
	m_pValue(NULL),
	m_iNumFeatures(0)
{
}

/*******************************************************************************
	Set Haar feature.
*******************************************************************************/
void CHaarSpace::SetFeature(int x0, int y0, int width0, int height0, int w0, 
	int x1, int y1, int width1, int height1, int w1, 
	int x2, int y2, int width2, int height2, int w2)
{
	HaarFeature feature;
	int thr = 0;

	// rectangle 0
	feature.rect[0].x = x0;
	feature.rect[0].y = y0;
	feature.rect[0].width = width0;
	feature.rect[0].height = height0;
	feature.w[0] = w0;
	thr += width0 * height0 * w0;

	// rectangle 1
	feature.rect[1].x = x1;
	feature.rect[1].y = y1;
	feature.rect[1].width = width1;
	feature.rect[1].height = height1;
	feature.w[1] = w1;
	thr += width1 * height1 * w1;

	// rectangle 2
	feature.rect[2].x = x2;
	feature.rect[2].y = y2;
	feature.rect[2].width = width2;
	feature.rect[2].height = height2;
	feature.w[2] = w2;
	thr += width2 * height2 * w2;

	// threshold
	feature.thr = thr * HAAR_DC;

	m_features.push_back(feature);
}

/*******************************************************************************
	Constructor.
*******************************************************************************/
int CHaarSpace::InitFeatures(int step)
{
	Size winsize;
	int mode = 1, symmetric = 1;
	winsize.width = winsize.height = HAAR_SIZE;

	if (m_iNumFeatures > 0) {
		return m_iNumFeatures;
	}

	int s0 = 1; //36; // minimum total area size of basic haar feature
	int s1 = 1; //12; // minimum total area size of tilted haar features 2
	int s2 = 1; //18; // minimum total area size of tilted haar features 3
	int s3 = 1; //24; // minimum total area size of tilted haar features 4

	int x  = 0;
	int y  = 0;
	int dx = 0;
	int dy = 0;

	const int xstep = step;
	const int ystep = step;
	const int dx_step = step;
	const int dy_step = step;
	for (x = 0; x < winsize.width; x += xstep) {
		for (y = 0; y < winsize.height; y += ystep) {
			for (dx = dx_step; dx <= winsize.width; dx += dx_step) {
				for (dy = dy_step; dy <= winsize.height; dy += dy_step) {
					// haar_x2
					if ((x+dx*2 <= winsize.width) && (y+dy <= winsize.height)) {
						if (dx*2*dy < s0) 
							continue;
						if (!symmetric || (x+x+dx*2 <= winsize.width)) {
							SetFeature(x, y, dx*2, dy, -1, x+dx, y, dx, dy, +2);
						}
					}

					// haar_y2
					if ((x+dx <= winsize.width) && (y+dy*2 <= winsize.height)) {
						if (dx*2*dy < s0) 
							continue;
						if (!symmetric || (x+x+dx <= winsize.width)) {
							SetFeature(x, y, dx, dy*2, -1, x, y+dy, dx, dy, +2);
						}
					}

					// haar_x3
					if ((x+dx*3 <= winsize.width) && (y+dy <= winsize.height)) {
						if (dx*3*dy < s0) 
							continue;
						if (!symmetric || (x+x+dx*3 <= winsize.width)) {
							SetFeature(x, y, dx*3, dy, -1, x+dx, y, dx, dy, +3);
						}
					}

					// haar_y3
					if ((x+dx <= winsize.width) && (y+dy*3 <= winsize.height)) {
						if (dx*3*dy < s0) 
							continue;
						if (!symmetric || (x+x+dx <= winsize.width)) {
							SetFeature(x, y, dx, dy*3, -1, x, y+dy, dx, dy, +3);
						}
					}

					if (mode != 0 /*BASIC*/) {
						// haar_x4
						if ((x+dx*4 <= winsize.width) && (y+dy <= winsize.height)) {
							if (dx*4*dy < s0) 
								continue;
							if (!symmetric || (x+x+dx*4 <= winsize.width)) {
								SetFeature(x, y, dx*4, dy, -1, x+dx, y, dx*2, dy, +2);
							}
						}

						// haar_y4
						if ((x+dx <= winsize.width) && (y+dy*4 <= winsize.height)) {
							if (dx*4*dy < s0) continue;
							if (!symmetric || (x+x+dx <= winsize.width)) {
								SetFeature(x, y, dx, dy*4, -1, x, y+dy, dx, dy*2, +2);
							}
						}
					}

					// x2_y2
					if ((x+dx*2 <= winsize.width) && (y+dy*2 <= winsize.height)) {
						if (dx*4*dy < s0) 
							continue;
						if (!symmetric || (x+x+dx*2 <= winsize.width)) {
							SetFeature(x, y, dx*2, dy*2, -1, x, y, dx, dy, +2,
								x+dx, y+dy, dx, dy, +2);
						}
					}

					if (mode != 0 /*BASIC*/) {                
						// point
						if ((x+dx*3 <= winsize.width) && (y+dy*3 <= winsize.height)) {
							if (dx*9*dy < s0) 
								continue;
							if (!symmetric || (x+x+dx*3 <= winsize.width))  {
								SetFeature(x, y, dx*3, dy*3, -1, x+dx, y+dy, dx, dy, +9);
							}
						}
					}

					if (mode == 2 /*ALL*/) {                
						// tilted haar_x2 
						if ((x+2*dx <= winsize.width) && (y+2*dx+dy <= winsize.height) && (x-dy>= 0)) {
							if (dx*2*dy < s1) 
								continue;
							if (!symmetric || (x <= (winsize.width / 2))) {
								SetFeature(x, y, dx*2, dy, -1, x, y, dx, dy, +2 );
							}
						}

						// tilted haar_y2 
						if ( (x+dx <= winsize.width) && (y+dx+2*dy <= winsize.height) && (x-2*dy>= 0) ) {
							if (dx*2*dy < s1) 
								continue;
							if (!symmetric || (x <= (winsize.width / 2))) {
								SetFeature(x, y, dx, 2*dy, -1, x, y, dx, dy, +2 );
							}
						}

						// tilted haar_x3 
						if ( (x+3*dx <= winsize.width) && (y+3*dx+dy <= winsize.height) && (x-dy>= 0)) {
							if (dx*3*dy < s2) 
								continue;
							if (!symmetric || (x <= (winsize.width / 2))) {
								SetFeature(x, y, dx*3, dy, -1, x+dx, y+dx, dx, dy, +3);
							}
						}

						// tilted haar_y3 
						if ( (x+dx <= winsize.width) && (y+dx+3*dy <= winsize.height) && (x-3*dy>= 0)) {
							if (dx*3*dy < s2) 
								continue;
							if (!symmetric || (x <= (winsize.width / 2))) {
								SetFeature(x, y, dx, 3*dy, -1, x-dy, y+dy, dx, dy, +3);
							}
						}


						// tilted haar_x4 
						if ( (x+4*dx <= winsize.width) && (y+4*dx+dy <= winsize.height) && (x-dy>= 0)) {
							if (dx*4*dy < s3) 
								continue;
							if (!symmetric || (x <= (winsize.width / 2))) {
								SetFeature(x, y, dx*4, dy, -1, x+dx, y+dx, dx*2, dy, +2);
							}
						}

						// tilted haar_y4 
						if ( (x+dx <= winsize.width) && (y+dx+4*dy <= winsize.height) && (x-4*dy>= 0)) {
							if (dx*4*dy < s3) 
								continue;
							if (!symmetric || (x <= (winsize.width / 2))) {
								SetFeature(x, y, dx, 4*dy, -1, x-dy, y+dy, dx, 2*dy, +2);
							}
						}
					}
				}
			}
		}
	}

	m_iNumFeatures = m_features.size();
	if (m_iNumFeatures > 0) {
		m_pValue = new int[m_iNumFeatures];
	}

	// update feature pointer
	return m_iNumFeatures;
}

#define CV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
    /* (x, y) */                                                          \
    (p0) = sum + (rect).x + (step) * (rect).y,                            \
    /* (x + w, y) */                                                      \
    (p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
    /* (x + w, y) */                                                      \
    (p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
    /* (x + w, y + h) */                                                  \
    (p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

/*******************************************************************************
	Set integral image.
*******************************************************************************/
void CHaarSpace::SetImage(const Mat &image, Size origWinSize)
{
	if (image.cols < origWinSize.width || image.rows < origWinSize.height) {
		return;
	}

	int rn = image.rows + 1, cn = image.cols + 1;
	if (m_data.rows < rn || m_data.cols < cn) {
		m_data.create(rn, cn, CV_32S);
	}
	m_integral = Mat(rn, cn, CV_32S, m_data.data);

	integral(image, m_integral);

	const int* ptr = (const int*)m_integral.data;
	int step = m_integral.step / sizeof(ptr[0]);
	for (int i = 0; i < m_iNumFeatures; i++) {
		HaarFeature& f = m_features[i];
		f.p[0][0] = ptr + f.rect[0].x + step * f.rect[0].y;
		f.p[0][1] = ptr + f.rect[0].x + f.rect[0].width + (step) * f.rect[0].y;
		f.p[0][2] = ptr + f.rect[0].x + step * (f.rect[0].y + f.rect[0].height);
		f.p[0][3] = ptr + f.rect[0].x + f.rect[0].width + step * (f.rect[0].y + f.rect[0].height);
		f.p[1][0] = ptr + f.rect[1].x + step * f.rect[1].y;
		f.p[1][1] = ptr + f.rect[1].x + f.rect[1].width + step * f.rect[1].y;
		f.p[1][2] = ptr + f.rect[1].x + step * (f.rect[1].y + f.rect[1].height);
		f.p[1][3] = ptr + f.rect[1].x + f.rect[1].width + step * (f.rect[1].y + f.rect[1].height);
		f.p[2][0] = ptr + f.rect[2].x + step * f.rect[2].y;
		f.p[2][1] = ptr + f.rect[2].x + f.rect[2].width + step * f.rect[2].y;
		f.p[2][2] = ptr + f.rect[2].x + step * (f.rect[2].y + f.rect[2].height);
		f.p[2][3] = ptr + f.rect[2].x + f.rect[2].width + step * (f.rect[2].y + f.rect[2].height);
	}
}

#define CALC_SUM_(p0, p1, p2, p3, offset) ((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])   
#define CALC_SUM(rect, offset) CALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)

/*******************************************************************************
	Compute features.
*******************************************************************************/
void CHaarSpace::ComputeFeatures(Point& point, int* x)
{
    int offset = point.y * (m_integral.step / sizeof(int)) + point.x;

	for (int i = 0; i < m_iNumFeatures; i++) {
		HaarFeature& f = m_features[i];
		x[i] = f.w[0] * CALC_SUM(f.p[0], offset) + f.w[1] * CALC_SUM(f.p[1], offset);
		if (f.w[2] != 0) {
			x[i] += f.w[2] * CALC_SUM(f.p[2], offset);
		}
	}
}

/*******************************************************************************
	Compute sample features.
*******************************************************************************/
void CHaarSpace::ComputeSampleFeatures(Sample& sample)
{
	if (sample.locX >= 0 && sample.locY >= 0) {
		Point p = Point(sample.locX, sample.locY);
		ComputeFeatures(p, m_pValue);
		sample.x = m_pValue;
	}
}

/*******************************************************************************
	Load from file.
*******************************************************************************/
bool CHaarSpace::Load(FILE* pf)
{
	fread(&m_iNumFeatures, sizeof(int), 1, pf);
	if (m_iNumFeatures > 0) {
		for (unsigned i = 0; i < m_iNumFeatures; i++) {
			HaarFeature feature;
			feature.Load(pf);
			m_features.push_back(feature);
		}
		m_pValue = new int[m_iNumFeatures];
	}

	return true;
}

/*******************************************************************************
	Store into file.
*******************************************************************************/
void CHaarSpace::Store(FILE* pf)
{
	fwrite(&m_iNumFeatures, sizeof(int), 1, pf);
	for (unsigned i = 0; i < m_features.size(); i++) {
		m_features[i].Store(pf);
	}
}

