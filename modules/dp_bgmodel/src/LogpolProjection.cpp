#include "precomp.hpp"
#include "LogpolProjection.hpp"

#include <cmath>
#include <iostream>

namespace cv {
namespace dp_bgmodel {
// constructor
ImageLogPolProjection::ImageLogPolProjection(const unsigned int nbRows,
		const unsigned int nbColumns, const PROJECTIONTYPE projection, const bool colorModeCapable) :
		BasicRetinaFilter(nbRows, nbColumns), _sampledFrame(0), _tempBuffer(_localBuffer), _transformTable(
				0), _irregularLPfilteredFrame(_filterOutput)
{
	_inputDoubleNBpixels = nbRows * nbColumns * 2;
	_selectedProjection = projection;
	_reductionFactor = 0;
	_initOK = false;
	_usefullpixelIndex = 0;
	_colorModeCapable = colorModeCapable;
#ifdef IMAGELOGPOLPROJECTION_DEBUG
	std::cout<<"ImageLogPolProjection::allocating"<<std::endl;
#endif
	if (_colorModeCapable) {
		_tempBuffer.resize(nbRows * nbColumns * 3);
	}
#ifdef IMAGELOGPOLPROJECTION_DEBUG
	std::cout<<"ImageLogPolProjection::done"<<std::endl;
#endif

	clearAllBuffers();
}

// destructor
ImageLogPolProjection::~ImageLogPolProjection()
{

}

// reset buffers method
void ImageLogPolProjection::clearAllBuffers()
{
	_sampledFrame = 0;
	_tempBuffer = 0;
	BasicRetinaFilter::clearAllBuffers();
}

/**
 * resize retina color filter object (resize all allocated buffers)
 * @param NBrows: the new height size
 * @param NBcolumns: the new width size
 */
void ImageLogPolProjection::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
	BasicRetinaFilter::resize(NBrows, NBcolumns);
	initProjection(_reductionFactor, _samplingStrenght);

	// reset buffers method
	clearAllBuffers();

}

// init functions depending on the projection type
bool ImageLogPolProjection::initProjection(const double reductionFactor,
		const double samplingStrenght)
{
	switch (_selectedProjection) {
	case RETINALOGPROJECTION:
		return _initLogRetinaSampling(reductionFactor, samplingStrenght);
		break;
	case CORTEXLOGPOLARPROJECTION:
		return _initLogPolarCortexSampling(reductionFactor, samplingStrenght);
		break;
	default:
		std::cout
				<< "ImageLogPolProjection::no projection setted up... performing default retina projection... take care"
				<< std::endl;
		return _initLogRetinaSampling(reductionFactor, samplingStrenght);
		break;
	}
}

// -> private init functions dedicated to each projection
bool ImageLogPolProjection::_initLogRetinaSampling(const double reductionFactor,
		const double samplingStrenght)
{
	_initOK = false;

	if (_selectedProjection != RETINALOGPROJECTION) {
		std::cerr
				<< "ImageLogPolProjection::initLogRetinaSampling: could not initialize logPolar projection for a log projection system\n -> you probably chose the wrong init function, use initLogPolarCortexSampling() instead"
				<< std::endl;
		return false;
	}
	if (reductionFactor < 1.0) {
		std::cerr
				<< "ImageLogPolProjection::initLogRetinaSampling: reduction factor must be superior to 0, skeeping initialisation..."
				<< std::endl;
		return false;
	}

	// compute image output size
	_outputNBrows = predictOutputSize(this->getNBrows(), reductionFactor);
	_outputNBcolumns = predictOutputSize(this->getNBcolumns(), reductionFactor);
	_outputNBpixels = _outputNBrows * _outputNBcolumns;
	_outputDoubleNBpixels = _outputNBrows * _outputNBcolumns * 2;

#ifdef IMAGELOGPOLPROJECTION_DEBUG
	std::cout<<"ImageLogPolProjection::initLogRetinaSampling: Log resampled image resampling factor: "<<reductionFactor<<", strenght:"<<samplingStrenght<<std::endl;
	std::cout<<"ImageLogPolProjection::initLogRetinaSampling: Log resampled image size: "<<_outputNBrows<<"*"<<_outputNBcolumns<<std::endl;
#endif

	// setup progressive prefilter that will be applied BEFORE log sampling
	setProgressiveFilterConstants_CentredAccuracy(0.f, 0.f, 0.99f);

	// (re)create the image output buffer and transform table if the reduction factor changed
	_sampledFrame.resize(_outputNBpixels * (1 + (unsigned int) _colorModeCapable * 2));

	// specifiying new reduction factor after preliminar checks
	_reductionFactor = reductionFactor;
	_samplingStrenght = samplingStrenght;

	// compute the rlim for symetric rows/columns sampling, then, the rlim is based on the smallest dimension
	_minDimension = (double) (
			_filterOutput.getNBrows() < _filterOutput.getNBcolumns() ?
					_filterOutput.getNBrows() : _filterOutput.getNBcolumns());

	// input frame dimensions dependent log sampling:
	//double rlim=1.0/reductionFactor*(minDimension/2.0+samplingStrenght);

	// input frame dimensions INdependent log sampling:
	_azero = (1.0 + reductionFactor * std::sqrt(samplingStrenght))
			/ (reductionFactor * reductionFactor * samplingStrenght - 1.0);
	_alim = (1.0 + _azero) / reductionFactor;
#ifdef IMAGELOGPOLPROJECTION_DEBUG
	std::cout<<"ImageLogPolProjection::initLogRetinaSampling: rlim= "<<rlim<<std::endl;
	std::cout<<"ImageLogPolProjection::initLogRetinaSampling: alim= "<<alim<<std::endl;
#endif

	// get half frame size
	unsigned int halfOutputRows = _outputNBrows / 2 - 1;
	unsigned int halfOutputColumns = _outputNBcolumns / 2 - 1;
	unsigned int halfInputRows = _filterOutput.getNBrows() / 2 - 1;
	unsigned int halfInputColumns = _filterOutput.getNBcolumns() / 2 - 1;

	// computing log sampling matrix by computing quarters of images
	// the original new image center (_filterOutput.getNBrows()/2, _filterOutput.getNBcolumns()/2) being at coordinate (_filterOutput.getNBrows()/(2*_reductionFactor), _filterOutput.getNBcolumns()/(2*_reductionFactor))

	// -> use a temporary transform table which is bigger than the final one, we only report pixels coordinates that are included in the sampled picture
	std::valarray<unsigned int> tempTransformTable(2 * _outputNBpixels); // the structure would be: (pixelInputCoordinate n)(pixelOutputCoordinate n)(pixelInputCoordinate n+1)(pixelOutputCoordinate n+1)
	_usefullpixelIndex = 0;

	double rMax = 0;
	halfInputRows < halfInputColumns ? rMax = (double) (halfInputRows * halfInputRows) : rMax =
												(double) (halfInputColumns * halfInputColumns);

	for (unsigned int idRow = 0; idRow < halfOutputRows; ++idRow) {
		for (unsigned int idColumn = 0; idColumn < halfOutputColumns; ++idColumn) {
			// get the pixel position in the original picture

			// -> input frame dimensions dependent log sampling:
			//double scale = samplingStrenght/(rlim-(double)std::sqrt(idRow*idRow+idColumn*idColumn));

			// -> input frame dimensions INdependent log sampling:
			double scale = getOriginalRadiusLength(
					(double) std::sqrt((double) (idRow * idRow + idColumn * idColumn)));
#ifdef IMAGELOGPOLPROJECTION_DEBUG
			std::cout<<"ImageLogPolProjection::initLogRetinaSampling: scale= "<<scale<<std::endl;
			std::cout<<"ImageLogPolProjection::initLogRetinaSampling: scale2= "<<scale2<<std::endl;
#endif
			if (scale < 0) ///check it later
				scale = 10000;

#ifdef IMAGELOGPOLPROJECTION_DEBUG
			//            std::cout<<"ImageLogPolProjection::initLogRetinaSampling: scale= "<<scale<<std::endl;
#endif

			unsigned int u = (unsigned int) floor((double) idRow * scale);
			unsigned int v = (unsigned int) floor((double) idColumn * scale);

			// manage border effects
			double length = u * u + v * v;
			double radiusRatio = std::sqrt(rMax / length);

#ifdef IMAGELOGPOLPROJECTION_DEBUG
			std::cout<<"ImageLogPolProjection::(inputH, inputW)="<<halfInputRows<<", "<<halfInputColumns<<", Rmax2="<<rMax<<std::endl;
			std::cout<<"before ==> ImageLogPolProjection::(u, v)="<<u<<", "<<v<<", r="<<u*u+v*v<<std::endl;
			std::cout<<"ratio ="<<radiusRatio<<std::endl;
#endif

			if (radiusRatio < 1.0) {
				u = (unsigned int) floor(radiusRatio * double(u));
				v = (unsigned int) floor(radiusRatio * double(v));
			}
#ifdef IMAGELOGPOLPROJECTION_DEBUG
			std::cout<<"after ==> ImageLogPolProjection::(u, v)="<<u<<", "<<v<<", r="<<u*u+v*v<<std::endl;
			std::cout<<"ImageLogPolProjection::("<<(halfOutputRows-idRow)<<", "<<idColumn+halfOutputColumns<<") <- ("<<halfInputRows-u<<", "<<v+halfInputColumns<<")"<<std::endl;
			std::cout<<(halfOutputRows-idRow)+(halfOutputColumns+idColumn)*_outputNBrows<<" -> "<<(halfInputRows-u)+_filterOutput.getNBrows()*(halfInputColumns+v)<<std::endl;
#endif

			if ((u < halfInputRows) && (v < halfInputColumns)) {

#ifdef IMAGELOGPOLPROJECTION_DEBUG
				std::cout<<"*** VALID ***"<<std::endl;
#endif

				// set pixel coordinate of the input picture in the transform table at the current log sampled pixel
				// 1st quadrant
				tempTransformTable[_usefullpixelIndex++] = (halfOutputColumns + idColumn)
						+ (halfOutputRows - idRow) * _outputNBcolumns;
				tempTransformTable[_usefullpixelIndex++] = _filterOutput.getNBcolumns()
						* (halfInputRows - u) + (halfInputColumns + v);
				// 2nd quadrant
				tempTransformTable[_usefullpixelIndex++] = (halfOutputColumns + idColumn)
						+ (halfOutputRows + idRow) * _outputNBcolumns;
				tempTransformTable[_usefullpixelIndex++] = _filterOutput.getNBcolumns()
						* (halfInputRows + u) + (halfInputColumns + v);
				// 3rd quadrant
				tempTransformTable[_usefullpixelIndex++] = (halfOutputColumns - idColumn)
						+ (halfOutputRows - idRow) * _outputNBcolumns;
				tempTransformTable[_usefullpixelIndex++] = _filterOutput.getNBcolumns()
						* (halfInputRows - u) + (halfInputColumns - v);
				// 4td quadrant
				tempTransformTable[_usefullpixelIndex++] = (halfOutputColumns - idColumn)
						+ (halfOutputRows + idRow) * _outputNBcolumns;
				tempTransformTable[_usefullpixelIndex++] = _filterOutput.getNBcolumns()
						* (halfInputRows + u) + (halfInputColumns - v);
			}
		}
	}

	// (re)creating and filling the transform table
	_transformTable.resize(_usefullpixelIndex);
	memcpy(&_transformTable[0], &tempTransformTable[0], sizeof(unsigned int) * _usefullpixelIndex);

	// reset all buffers
	clearAllBuffers();

#ifdef IMAGELOGPOLPROJECTION_DEBUG
	std::cout<<"ImageLogPolProjection::initLogRetinaSampling: init done successfully"<<std::endl;
#endif
	_initOK = true;
	return _initOK;
}

bool ImageLogPolProjection::_initLogPolarCortexSampling(const double reductionFactor, const double)
{
	_initOK = false;

	if (_selectedProjection != CORTEXLOGPOLARPROJECTION) {
		std::cerr
				<< "ImageLogPolProjection::could not initialize log projection for a logPolar projection system\n -> you probably chose the wrong init function, use initLogRetinaSampling() instead"
				<< std::endl;
		return false;
	}

	if (reductionFactor < 1.0) {
		std::cerr
				<< "ImageLogPolProjection::reduction factor must be superior to 0, skeeping initialisation..."
				<< std::endl;
		return false;
	}

	// compute the smallest image size
	unsigned int minDimension = (
			_filterOutput.getNBrows() < _filterOutput.getNBcolumns() ?
					_filterOutput.getNBrows() : _filterOutput.getNBcolumns());
	// specifiying new reduction factor after preliminar checks
	_reductionFactor = reductionFactor;
	// compute image output size
	_outputNBrows = (unsigned int) ((double) minDimension / reductionFactor);
	_outputNBcolumns = (unsigned int) ((double) minDimension / reductionFactor);
	_outputNBpixels = _outputNBrows * _outputNBcolumns;
	_outputDoubleNBpixels = _outputNBrows * _outputNBcolumns * 2;

	// get half frame size
	//unsigned int halfOutputRows = _outputNBrows/2-1;
	//unsigned int halfOutputColumns = _outputNBcolumns/2-1;
	unsigned int halfInputRows = _filterOutput.getNBrows() / 2 - 1;
	unsigned int halfInputColumns = _filterOutput.getNBcolumns() / 2 - 1;

#ifdef IMAGELOGPOLPROJECTION_DEBUG
	std::cout<<"ImageLogPolProjection::Log resampled image size: "<<_outputNBrows<<"*"<<_outputNBcolumns<<std::endl;
#endif

	// setup progressive prefilter that will be applied BEFORE log sampling
	setProgressiveFilterConstants_CentredAccuracy(0.f, 0.f, 0.99f);

	// (re)create the image output buffer and transform table if the reduction factor changed
	_sampledFrame.resize(_outputNBpixels * (1 + (unsigned int) _colorModeCapable * 2));

	// create the radius and orientation axis and fill them, radius E [0;1], orientation E[-pi, pi]
	std::valarray<double> radiusAxis(_outputNBcolumns);
	double radiusStep = 2.30 / (double) _outputNBcolumns;
	for (unsigned int i = 0; i < _outputNBcolumns; ++i) {
		radiusAxis[i] = i * radiusStep;
	}
	std::valarray<double> orientationAxis(_outputNBrows);
	double orientationStep = -2.0 * CV_PI / (double) _outputNBrows;
	for (unsigned int io = 0; io < _outputNBrows; ++io) {
		orientationAxis[io] = io * orientationStep;
	}
	// -> use a temporay transform table which is bigger than the final one, we only report pixels coordinates that are included in the sampled picture
	std::valarray<unsigned int> tempTransformTable(2 * _outputNBpixels); // the structure would be: (pixelInputCoordinate n)(pixelOutputCoordinate n)(pixelInputCoordinate n+1)(pixelOutputCoordinate n+1)
	_usefullpixelIndex = 0;

	//std::cout<<"ImageLogPolProjection::Starting cortex projection"<<std::endl;
	// compute transformation, get theta and Radius in reagrd of the output sampled pixel
	double diagonalLenght = std::sqrt(
			(double) (_outputNBcolumns * _outputNBcolumns + _outputNBrows * _outputNBrows));
	for (unsigned int radiusIndex = 0; radiusIndex < _outputNBcolumns; ++radiusIndex)
		for (unsigned int orientationIndex = 0; orientationIndex < _outputNBrows;
				++orientationIndex) {
			double x = 1.0 + sinh(radiusAxis[radiusIndex]) * cos(orientationAxis[orientationIndex]);
			double y = sinh(radiusAxis[radiusIndex]) * sin(orientationAxis[orientationIndex]);
			// get the input picture coordinate
			double R = diagonalLenght * std::sqrt(x * x + y * y) / (5.0 + std::sqrt(x * x + y * y));
			double theta = atan2(y, x);
			// convert input polar coord into cartesian/C compatble coordinate
			unsigned int columnIndex = (unsigned int) (cos(theta) * R) + halfInputColumns;
			unsigned int rowIndex = (unsigned int) (sin(theta) * R) + halfInputRows;
			//std::cout<<"ImageLogPolProjection::R="<<R<<" / Theta="<<theta<<" / (x, y)="<<columnIndex<<", "<<rowIndex<<std::endl;
			if ((columnIndex < _filterOutput.getNBcolumns()) && (columnIndex > 0)
					&& (rowIndex < _filterOutput.getNBrows()) && (rowIndex > 0)) {
				// set coordinate
				tempTransformTable[_usefullpixelIndex++] = radiusIndex
						+ orientationIndex * _outputNBcolumns;
				tempTransformTable[_usefullpixelIndex++] = columnIndex
						+ rowIndex * _filterOutput.getNBcolumns();
			}
		}

	// (re)creating and filling the transform table
	_transformTable.resize(_usefullpixelIndex);
	memcpy(&_transformTable[0], &tempTransformTable[0], sizeof(unsigned int) * _usefullpixelIndex);

	// reset all buffers
	clearAllBuffers();
	_initOK = true;
	return true;
}

// action function
std::valarray<float> &ImageLogPolProjection::runProjection(const std::valarray<float> &inputFrame,
		const bool colorMode)
{
	if (_colorModeCapable && colorMode) {
		// progressive filtering and storage of the result in _tempBuffer
		_spatiotemporalLPfilter_Irregular(get_data(inputFrame), &_irregularLPfilteredFrame[0]);
		_spatiotemporalLPfilter_Irregular(&_irregularLPfilteredFrame[0], &_tempBuffer[0]); // warning, temporal issue may occur, if the temporal constant is not NULL !!!

		_spatiotemporalLPfilter_Irregular(get_data(inputFrame) + _filterOutput.getNBpixels(),
				&_irregularLPfilteredFrame[0]);
		_spatiotemporalLPfilter_Irregular(&_irregularLPfilteredFrame[0],
				&_tempBuffer[0] + _filterOutput.getNBpixels());

		_spatiotemporalLPfilter_Irregular(get_data(inputFrame) + _filterOutput.getNBpixels() * 2,
				&_irregularLPfilteredFrame[0]);
		_spatiotemporalLPfilter_Irregular(&_irregularLPfilteredFrame[0],
				&_tempBuffer[0] + _filterOutput.getNBpixels() * 2);

		// applying image projection/resampling
		unsigned int *transformTablePTR = &_transformTable[0];
		for (unsigned int i = 0; i < _usefullpixelIndex; i += 2, transformTablePTR += 2) {
#ifdef IMAGELOGPOLPROJECTION_DEBUG
			std::cout<<"ImageLogPolProjection::i:"<<i<<"output(max="<<_outputNBpixels<<")="<<_transformTable[i]<<" / intput(max="<<_filterOutput.getNBpixels()<<")="<<_transformTable[i+1]<<std::endl;
#endif
			_sampledFrame[*(transformTablePTR)] = _tempBuffer[*(transformTablePTR + 1)];
			_sampledFrame[*(transformTablePTR) + _outputNBpixels] = _tempBuffer[*(transformTablePTR
					+ 1) + _filterOutput.getNBpixels()];
			_sampledFrame[*(transformTablePTR) + _outputDoubleNBpixels] =
					_tempBuffer[*(transformTablePTR + 1) + _inputDoubleNBpixels];
		}

#ifdef IMAGELOGPOLPROJECTION_DEBUG
		std::cout<<"ImageLogPolProjection::runProjection: color image projection OK"<<std::endl;
#endif
		//normalizeGrayOutput_0_maxOutputValue(_sampledFrame, _outputNBpixels);
	} else {
		_spatiotemporalLPfilter_Irregular(get_data(inputFrame), &_irregularLPfilteredFrame[0]);
		_spatiotemporalLPfilter_Irregular(&_irregularLPfilteredFrame[0],
				&_irregularLPfilteredFrame[0]);
		// applying image projection/resampling
		unsigned int *transformTablePTR = &_transformTable[0];
		for (unsigned int i = 0; i < _usefullpixelIndex; i += 2, transformTablePTR += 2) {
#ifdef IMAGELOGPOLPROJECTION_DEBUG
			std::cout<<"i:"<<i<<"output(max="<<_outputNBpixels<<")="<<_transformTable[i]<<" / intput(max="<<_filterOutput.getNBpixels()<<")="<<_transformTable[i+1]<<std::endl;
#endif
			_sampledFrame[*(transformTablePTR)] =
					_irregularLPfilteredFrame[*(transformTablePTR + 1)];
		}
		//normalizeGrayOutput_0_maxOutputValue(_sampledFrame, _outputNBpixels);
#ifdef IMAGELOGPOLPROJECTION_DEBUG
		std::cout<<"ImageLogPolProjection::runProjection: gray level image projection OK"<<std::endl;
#endif
	}

	return _sampledFrame;
}

} // end of namespace bioinspired
} // end of namespace cv