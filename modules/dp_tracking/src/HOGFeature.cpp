#include "opencv2/opencv.hpp"
#include "FeatureSpace.h"


/*
 * Compute 32-D HOG feature.
 * imageM (in): image patch
 * featM (out): feature matrix
 * sbin (in): block size
 * pad_x (in): x padding
 * pad_y (in): y padding
 */
static void computeHOG32D(const Mat &imageM, Mat &featM, const int sbin, const int pad_x,
		const int pad_y)
{
	//printf("computeHOG32D\n");
	const int dimHOG = 32;
	CV_Assert(pad_x >= 0);
	CV_Assert(pad_y >= 0);
	CV_Assert(imageM.channels() == 3);
	CV_Assert(imageM.depth() == CV_64F);

	// epsilon to avoid division by zero
	const double eps = 0.0001;
	// number of orientation bins
	const int numOrient = 18;
	// unit vectors to compute gradient orientation
	const double uu[9] = { 1.000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397 };
	const double vv[9] = { 0.000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660, 0.6428, 0.3420 };

	// image size
	const Size imageSize = imageM.size();
	// block size
	int bW = cvFloor((double)imageSize.width / (double)sbin);
	int bH = cvFloor((double)imageSize.height / (double)sbin);
	//printf("width = %d, height = %d, sbin = %d, bW = %d, bH = %d\n", imageSize.width, imageSize.height, sbin, bW, bH);

	// size of HOG features in blocks
	const Size blockSize(bW, bH);
	int oW = max(blockSize.width - 2, 0) + 2 * pad_x;
	int oH = max(blockSize.height - 2, 0) + 2 * pad_y;
	Size outSize = Size(oW, oH);
	// size of visible
	const Size visible = blockSize * sbin;

	// initialize histogram, norm, output feature matrices
	Mat histM = Mat::zeros(Size(blockSize.width * numOrient, blockSize.height), CV_64F);
	Mat normM = Mat::zeros(Size(blockSize.width, blockSize.height), CV_64F);
	featM = Mat::zeros(Size(outSize.width * dimHOG, outSize.height), CV_64F);

	// get the stride of each matrix
	const size_t imStride = imageM.step1();
	const size_t histStride = histM.step1();
	const size_t normStride = normM.step1();
	const size_t featStride = featM.step1();

	// calculate the zero offset
	const double* im = imageM.ptr<double>(0);
	double* const hist = histM.ptr<double>(0);
	double* const norm = normM.ptr<double>(0);
	double* const feat = featM.ptr<double>(0);

	// compute gradient and orientation histogram
	for (int y = 1; y < visible.height - 1; y++) {
		for (int x = 1; x < visible.width - 1; x++) {
			// OpenCV uses an interleaved format: BGR-BGR-BGR
			const double* s = im + 3 * min(x, imageM.cols - 2) + min(y, imageM.rows - 2) * imStride;

			// blue image channel
			double dyb = *(s + imStride) - *(s - imStride);
			double dxb = *(s + 3) - *(s - 3);
			double vb = dxb * dxb + dyb * dyb;

			// green image channel
			s += 1;
			double dyg = *(s + imStride) - *(s - imStride);
			double dxg = *(s + 3) - *(s - 3);
			double vg = dxg * dxg + dyg * dyg;

			// red image channel
			s += 1;
			double dy = *(s + imStride) - *(s - imStride);
			double dx = *(s + 3) - *(s - 3);
			double v = dx * dx + dy * dy;

			// pick the channel with the strongest gradient
			if (vg > v) {
				v = vg;
				dx = dxg;
				dy = dyg;
			}
			if (vb > v) {
				v = vb;
				dx = dxb;
				dy = dyb;
			}

			// snap to one of the 18 orientations
			double best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < (int) numOrient / 2; o++) {
				double dot = uu[o] * dx + vv[o] * dy;
				if (dot > best_dot) {
					best_dot = dot;
					best_o = o;
				} else if (-dot > best_dot) {
					best_dot = -dot;
					best_o = o + (int) (numOrient / 2);
				}
			}

			// add to 4 histograms around pixel using bilinear interpolation
			double yp = ((double) y + 0.5) / (double) sbin - 0.5;
			double xp = ((double) x + 0.5) / (double) sbin - 0.5;
			int iyp = (int) cvFloor(yp);
			int ixp = (int) cvFloor(xp);
			double vy0 = yp - iyp;
			double vx0 = xp - ixp;
			double vy1 = 1.0 - vy0;
			double vx1 = 1.0 - vx0;
			v = sqrt(v);

			// fill the value into the 4 neighborhood cells
			if (iyp >= 0 && ixp >= 0)
				*(hist + iyp * histStride + ixp * numOrient + best_o) += vy1 * vx1 * v;
			if (iyp >= 0 && ixp + 1 < blockSize.width)
				*(hist + iyp * histStride + (ixp + 1) * numOrient + best_o) += vx0 * vy1 * v;
			if (iyp + 1 < blockSize.height && ixp >= 0)
				*(hist + (iyp + 1) * histStride + ixp * numOrient + best_o) += vy0 * vx1 * v;
			if (iyp + 1 < blockSize.height && ixp + 1 < blockSize.width)
				*(hist + (iyp + 1) * histStride + (ixp + 1) * numOrient + best_o) += vy0 * vx0 * v;
		} // for y
	} // for x

	// compute the energy in each block by summing over orientation
	for (int y = 0; y < blockSize.height; y++) {
		const double* src = hist + y * histStride;
		double* dst = norm + y * normStride;
		double const* const dst_end = dst + blockSize.width;
		// loop for blocks to accumulate orientation energy
		while (dst < dst_end) {
			*dst = 0;
			for (int o = 0; o < (int) (numOrient / 2); o++) {
				*dst += (*src + *(src + numOrient / 2)) * (*src + *(src + numOrient / 2));
				src++;
			}
			dst++;
			src += numOrient / 2;
		}
	}

	// compute the features
	for (int y = pad_y; y < outSize.height - pad_y; y++) {
		for (int x = pad_x; x < outSize.width - pad_x; x++) {
			double* dst = feat + y * featStride + x * dimHOG;
			double* p, n1, n2, n3, n4;
			const double* src;

			// compute 4 norms ??
			p = norm + (y - pad_y + 1) * normStride + (x - pad_x + 1);
			n1 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
			p = norm + (y - pad_y) * normStride + (x - pad_x + 1);
			n2 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
			p = norm + (y - pad_y + 1) * normStride + x - pad_x;
			n3 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
			p = norm + (y - pad_y) * normStride + x - pad_x;
			n4 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);

			double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;

			// contrast-sensitive features
			src = hist + (y - pad_y + 1) * histStride + (x - pad_x + 1) * numOrient;
			for (int o = 0; o < numOrient; o++) {
				double val = *src;
				double h1 = min(val * n1, 0.2);
				double h2 = min(val * n2, 0.2);
				double h3 = min(val * n3, 0.2);
				double h4 = min(val * n4, 0.2);
				*(dst++) = 0.5 * (h1 + h2 + h3 + h4);

				src++;
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
			}

			// contrast-insensitive features
			src = hist + (y - pad_y + 1) * histStride + (x - pad_x + 1) * numOrient;
			for (int o = 0; o < numOrient / 2; o++) {
				double sum = *src + *(src + numOrient / 2);
				double h1 = min(sum * n1, 0.2);
				double h2 = min(sum * n2, 0.2);
				double h3 = min(sum * n3, 0.2);
				double h4 = min(sum * n4, 0.2);
				*(dst++) = 0.5 * (h1 + h2 + h3 + h4);
				src++;
			}

			// texture features
			*(dst++) = 0.2357 * t1;
			*(dst++) = 0.2357 * t2;
			*(dst++) = 0.2357 * t3;
			*(dst++) = 0.2357 * t4;
			*dst = 0;
		} // for x
	} // for y

	// truncate features
	for (int m = 0; m < featM.rows; m++) {
		for (int n = 0; n < featM.cols; n += dimHOG) {
			if (m > pad_y - 1 && m < featM.rows - pad_y && n > pad_x * dimHOG - 1
					&& n < featM.cols - pad_x * dimHOG)
				continue;
			else
				featM.at<double>(m, n + dimHOG - 1) = 1;
		}
	}
}

/*
 * Destructor.
 */
CHOGSpace::~CHOGSpace()
{
}

/*
 * Constructor.
 */
CHOGSpace::CHOGSpace()
{
	_cellSize = 16;
	_channels = 32;
	_iNumFeatures = 64 * 128 / (_cellSize * _cellSize) * _channels;
}

/*
 * Compute sample features.
 */
void CHOGSpace::ComputeSampleFeatures(Sample& sample)
{
	//printf("ComputeSampleFeatures\n");

	// convert image patch to floating point
	Mat im_;
	sample.imagePatch.convertTo(im_, CV_64FC3, 1.0 / 255.0);

	// compute HOG feature
	Mat hog;
	computeHOG32D(im_, hog, _cellSize, 1, 1);
	//printf("size = %d, %d, %d\n", hog.rows, hog.cols, hog.step[0]);

	// convert HOG feature to integer
	int *pDst = sample.x;
	for (int y = 0; y < hog.rows; y++) {
		double *pSrc = (double*)hog.data + y * hog.step[0] / sizeof(double);
		for (int x = 0; x < hog.cols; x++, pDst++) {
			*pDst = (int)floor(pSrc[x] * 1024);
		}
	}
}

