#include "precomp.hpp"
#include "RetinaFilter.hpp"

#include <iostream>
#include <cmath>

namespace cv {
namespace dp_bgmodel {
// standard constructor without any log sampling of the input frame
RetinaFilter::RetinaFilter(const unsigned int sizeRows, const unsigned int sizeColumns,
		const bool colorMode, const int samplingMethod, const bool useRetinaLogSampling,
		const double reductionFactor, const double samplingStrenght) :
		_retinaParvoMagnoMappedFrame(0), _retinaParvoMagnoMapCoefTable(0),
		_photoreceptorsPrefilter((1 - (int) useRetinaLogSampling) * sizeRows + useRetinaLogSampling
				* ImageLogPolProjection::predictOutputSize(sizeRows, reductionFactor),
				(1 - (int) useRetinaLogSampling) * sizeColumns + useRetinaLogSampling
				* ImageLogPolProjection::predictOutputSize(sizeColumns,	reductionFactor), 4),
				_ParvoRetinaFilter((1 - (int) useRetinaLogSampling) * sizeRows + useRetinaLogSampling
				* ImageLogPolProjection::predictOutputSize(sizeRows, reductionFactor),
				(1 - (int) useRetinaLogSampling) * sizeColumns + useRetinaLogSampling
				* ImageLogPolProjection::predictOutputSize(sizeColumns, reductionFactor)),
				_MagnoRetinaFilter((1 - (int) useRetinaLogSampling) * sizeRows + useRetinaLogSampling
				* ImageLogPolProjection::predictOutputSize(sizeRows, reductionFactor),
				(1 - (int) useRetinaLogSampling) * sizeColumns + useRetinaLogSampling
				* ImageLogPolProjection::predictOutputSize(sizeColumns, reductionFactor)),
				_colorEngine((1 - (int) useRetinaLogSampling) * sizeRows + useRetinaLogSampling
				* ImageLogPolProjection::predictOutputSize(sizeRows, reductionFactor),
				(1 - (int) useRetinaLogSampling) * sizeColumns + useRetinaLogSampling
				* ImageLogPolProjection::predictOutputSize(sizeColumns, reductionFactor), samplingMethod),
		// configure retina photoreceptors log sampling... if necessary
		_photoreceptorsLogSampling(NULL)
{

#ifdef RETINADEBUG
	std::cout<<"RetinaFilter::size( "<<_photoreceptorsPrefilter.getNBrows()<<", "<<_photoreceptorsPrefilter.getNBcolumns()<<")"<<" =? "<<_photoreceptorsPrefilter.getNBpixels()<<std::endl;
#endif
	if (useRetinaLogSampling) {
		_photoreceptorsLogSampling = new ImageLogPolProjection(sizeRows, sizeColumns,
				ImageLogPolProjection::RETINALOGPROJECTION, true);
		if (!_photoreceptorsLogSampling->initProjection(reductionFactor, samplingStrenght)) {
			std::cerr
					<< "RetinaFilter::Problem initializing photoreceptors log sampling, could not setup retina filter"
					<< std::endl;
			delete _photoreceptorsLogSampling;
			_photoreceptorsLogSampling = NULL;
		} else {
#ifdef RETINADEBUG
			std::cout<<"_photoreceptorsLogSampling::size( "<<_photoreceptorsLogSampling->getNBrows()<<", "<<_photoreceptorsLogSampling->getNBcolumns()<<")"<<" =? "<<_photoreceptorsLogSampling->getNBpixels()<<std::endl;
#endif
		}
	}

	// set default processing activities
	_useParvoOutput = true;
	_useMagnoOutput = true;

	_useColorMode = colorMode;

	// create hybrid output and related coefficient table
	_createHybridTable();

	// set default parameters
	setGlobalParameters();

	// stability controls values init
	_setInitPeriodCount();
	_globalTemporalConstant = 25;

	// reset all buffers
	clearAllBuffers();

	//  std::cout<<"RetinaFilter::size( "<<this->getNBrows()<<", "<<this->getNBcolumns()<<")"<<_filterOutput.size()<<" =? "<<_filterOutput.getNBpixels()<<std::endl;
}

// destructor
RetinaFilter::~RetinaFilter()
{
	if (_photoreceptorsLogSampling != NULL)
		delete _photoreceptorsLogSampling;
}

// function that clears all buffers of the object
void RetinaFilter::clearAllBuffers()
{
	_photoreceptorsPrefilter.clearAllBuffers();
	_ParvoRetinaFilter.clearAllBuffers();
	_MagnoRetinaFilter.clearAllBuffers();
	_colorEngine.clearAllBuffers();
	if (_photoreceptorsLogSampling != NULL)
		_photoreceptorsLogSampling->clearAllBuffers();

	// stability controls value init
	_setInitPeriodCount();
}

/**
 * resize retina filter object (resize all allocated buffers
 * @param NBrows: the new height size
 * @param NBcolumns: the new width size
 */
void RetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
	unsigned int rows = NBrows, cols = NBcolumns;

	// resize optionnal member and adjust other modules size if required
	if (_photoreceptorsLogSampling) {
		_photoreceptorsLogSampling->resize(NBrows, NBcolumns);
		rows = _photoreceptorsLogSampling->getOutputNBrows();
		cols = _photoreceptorsLogSampling->getOutputNBcolumns();
	}

	_photoreceptorsPrefilter.resize(rows, cols);
	_ParvoRetinaFilter.resize(rows, cols);
	_MagnoRetinaFilter.resize(rows, cols);
	_colorEngine.resize(rows, cols);

	// reset parvo magno mapping
	_createHybridTable();

	// clean buffers
	clearAllBuffers();
}

// stability controls value init
void RetinaFilter::_setInitPeriodCount()
{
	// find out the maximum temporal constant value and apply a security factor
	// false value (obviously too long) but appropriate for simple use
	_globalTemporalConstant = (unsigned int) (_ParvoRetinaFilter.getPhotoreceptorsTemporalConstant()
			+ _ParvoRetinaFilter.getHcellsTemporalConstant()
			+ _MagnoRetinaFilter.getTemporalConstant());
	// reset frame counter
	_ellapsedFramesSinceLastReset = 0;
}

void RetinaFilter::_createHybridTable()
{
	// create hybrid output and related coefficient table
	_retinaParvoMagnoMappedFrame.resize(_photoreceptorsPrefilter.getNBpixels());
	_retinaParvoMagnoMapCoefTable.resize(_photoreceptorsPrefilter.getNBpixels() * 2);

	// fill _hybridParvoMagnoCoefTable
	int i, j, halfRows = _photoreceptorsPrefilter.getNBrows() / 2, halfColumns =
			_photoreceptorsPrefilter.getNBcolumns() / 2;
	float *hybridParvoMagnoCoefTablePTR = &_retinaParvoMagnoMapCoefTable[0];
	float minDistance = MIN(halfRows, halfColumns) * 0.7f;
	for (i = 0; i < (int) _photoreceptorsPrefilter.getNBrows(); ++i) {
		for (j = 0; j < (int) _photoreceptorsPrefilter.getNBcolumns(); ++j) {
			float distanceToCenter = std::sqrt(((float) (i - halfRows) * (i - halfRows)
					+ (j - halfColumns) * (j - halfColumns)));
			if (distanceToCenter < minDistance) {
				float a = *(hybridParvoMagnoCoefTablePTR++) = 0.5f
						+ 0.5f * (float) cos(CV_PI * distanceToCenter / minDistance);
				*(hybridParvoMagnoCoefTablePTR++) = 1.f - a;
			} else {
				*(hybridParvoMagnoCoefTablePTR++) = 0.f;
				*(hybridParvoMagnoCoefTablePTR++) = 1.f;
			}
		}
	}
}

// setup parameters function and global data filling
void RetinaFilter::setGlobalParameters(const float OPLspatialResponse1,
		const float OPLtemporalresponse1, const float OPLassymetryGain,
		const float OPLspatialResponse2, const float OPLtemporalresponse2,
		const float LPfilterSpatialResponse, const float LPfilterGain,
		const float LPfilterTemporalresponse, const float MovingContoursExtractorCoefficient,
		const bool normalizeParvoOutput_0_maxOutputValue,
		const bool normalizeMagnoOutput_0_maxOutputValue, const float maxOutputValue,
		const float maxInputValue, const float meanValue)
{
	_normalizeParvoOutput_0_maxOutputValue = normalizeParvoOutput_0_maxOutputValue;
	_normalizeMagnoOutput_0_maxOutputValue = normalizeMagnoOutput_0_maxOutputValue;
	_maxOutputValue = maxOutputValue;
	_photoreceptorsPrefilter.setV0CompressionParameter(0.9f, maxInputValue, meanValue);
	_photoreceptorsPrefilter.setLPfilterParameters(10, 0, 1.5, 1); // keeps low pass filter with high cut frequency in memory (usefull for the tone mapping function)
	_photoreceptorsPrefilter.setLPfilterParameters(10, 0, 3.0, 2); // keeps low pass filter with low cut frequency in memory (usefull for the tone mapping function)
	_photoreceptorsPrefilter.setLPfilterParameters(0, 0, 10, 3); // keeps low pass filter with low cut frequency in memory (usefull for the tone mapping function)
	//this->setV0CompressionParameter(0.6, maxInputValue, meanValue); // keeps log compression sensitivity parameter (usefull for the tone mapping function)
	_ParvoRetinaFilter.setOPLandParvoFiltersParameters(0, OPLtemporalresponse1, OPLspatialResponse1,
			OPLassymetryGain, OPLtemporalresponse2, OPLspatialResponse2);
	_ParvoRetinaFilter.setV0CompressionParameter(0.9f, maxInputValue, meanValue);
	_MagnoRetinaFilter.setCoefficientsTable(LPfilterGain, LPfilterTemporalresponse,
			LPfilterSpatialResponse, MovingContoursExtractorCoefficient, 0,
			2.0f * LPfilterSpatialResponse);
	_MagnoRetinaFilter.setV0CompressionParameter(0.7f, maxInputValue, meanValue);

	// stability controls value init
	_setInitPeriodCount();
}

bool RetinaFilter::checkInput(const std::valarray<float> &input, const bool)
{
	BasicRetinaFilter *inputTarget = &_photoreceptorsPrefilter;
	if (_photoreceptorsLogSampling)
		inputTarget = _photoreceptorsLogSampling;

	bool test = input.size() == inputTarget->getNBpixels()
			|| input.size() == (inputTarget->getNBpixels() * 3);
	if (!test) {
		std::cerr
				<< "RetinaFilter::checkInput: input buffer does not match retina buffer size, conversion aborted"
				<< std::endl;
		std::cout << "RetinaFilter::checkInput: input size=" << input.size() << " / "
				<< "retina size=" << inputTarget->getNBpixels() << std::endl;
		return false;
	}

	return true;
}

// main function that runs the filter for a given input frame
bool RetinaFilter::runFilter(const std::valarray<float> &imageInput,
		const bool useAdaptiveFiltering, const bool processRetinaParvoMagnoMapping,
		const bool useColorMode, const bool inputIsColorMultiplexed)
{
	// preliminary check
	bool processSuccess = true;
	if (!checkInput(imageInput, useColorMode))
		return false;

	// run the color multiplexing if needed and compute each sub-filter of the retina:
	// -> local adaptation
	// -> contours OPL extraction
	// -> moving contours extraction

	// stability controls value update
	++_ellapsedFramesSinceLastReset;

	_useColorMode = useColorMode;

	/* pointer to the appropriate input data after,
	 * by default, if graylevel mode, the input is processed,
	 * if color or something else must be considered, specific preprocessing are applied
	 */

	const std::valarray<float> *selectedPhotoreceptorsLocalAdaptationInput = &imageInput;
	const std::valarray<float> *selectedPhotoreceptorsColorInput = &imageInput;

	// input data specific photoreceptors processing
	if (_photoreceptorsLogSampling) {
		_photoreceptorsLogSampling->runProjection(imageInput, useColorMode);
		selectedPhotoreceptorsColorInput = selectedPhotoreceptorsLocalAdaptationInput =
				&(_photoreceptorsLogSampling->getSampledFrame());
	}
	if (useColorMode && (!inputIsColorMultiplexed)) { // not multiplexed color input case
		_colorEngine.runColorMultiplexing(*selectedPhotoreceptorsColorInput);
		selectedPhotoreceptorsLocalAdaptationInput = &(_colorEngine.getMultiplexedFrame());
	}

	// generic retina processing

	// photoreceptors local adaptation
	_photoreceptorsPrefilter.runFilter_LocalAdapdation(*selectedPhotoreceptorsLocalAdaptationInput,
			_ParvoRetinaFilter.getHorizontalCellsOutput());
	// safety pixel values checks
	//_photoreceptorsPrefilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);

	// run parvo filter
	_ParvoRetinaFilter.runFilter(_photoreceptorsPrefilter.getOutput(), _useParvoOutput);
	if (_useParvoOutput) {
		// model the saturation of the cells, useful for visualization of the ON-OFF Parvo Output,
		// Bipolar cells outputs do not change !!!
		_ParvoRetinaFilter.normalizeGrayOutputCentredSigmoide();
		_ParvoRetinaFilter.centerReductImageLuminance(); // best for further spectrum analysis
		if (_normalizeParvoOutput_0_maxOutputValue)
			_ParvoRetinaFilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);
	}

	// run magno filter
	if (_useParvoOutput && _useMagnoOutput) {
		_MagnoRetinaFilter.runFilter(_ParvoRetinaFilter.getBipolarCellsON(),
				_ParvoRetinaFilter.getBipolarCellsOFF());
		if (_normalizeMagnoOutput_0_maxOutputValue) {
			_MagnoRetinaFilter.normalizeGrayOutput_0_maxOutputValue(_maxOutputValue);
		}
		_MagnoRetinaFilter.normalizeGrayOutputNearZeroCentreredSigmoide();
	}

	if (_useParvoOutput && _useMagnoOutput && processRetinaParvoMagnoMapping) {
		_processRetinaParvoMagnoMapping();
		if (_useColorMode) {
			_colorEngine.runColorDemultiplexing(_retinaParvoMagnoMappedFrame, useAdaptiveFiltering,
					_maxOutputValue);
			//_ColorEngine->getMultiplexedFrame());
			//_ParvoRetinaFilter->getPhotoreceptorsLPfilteringOutput());
		}

		return processSuccess;
	}

	if (_useParvoOutput && _useColorMode) {
		_colorEngine.runColorDemultiplexing(_ParvoRetinaFilter.getOutput(), useAdaptiveFiltering,
				_maxOutputValue);
		//_ColorEngine->getMultiplexedFrame());
		//_ParvoRetinaFilter->getPhotoreceptorsLPfilteringOutput());

		// compute A Cr1 Cr2 to LMS color space conversion
		//if (true)
		//  _applyImageColorSpaceConversion(_ColorEngine->getChrominance(), lmsTempBuffer.Buffer(), _LMStoACr1Cr2);
	}

	return processSuccess;
}

const std::valarray<float> &RetinaFilter::getContours()
{
	if (_useColorMode)
		return _colorEngine.getLuminance();
	else
		return _ParvoRetinaFilter.getOutput();
}

// run the initilized retina filter in order to perform gray image tone mapping, after this call all retina outputs are updated
void RetinaFilter::runGrayToneMapping(const std::valarray<float> &grayImageInput,
		std::valarray<float> &grayImageOutput, const float PhotoreceptorsCompression,
		const float ganglionCellsCompression)
{
	// preliminary check
	if (!checkInput(grayImageInput, false))
		return;

	this->_runGrayToneMapping(grayImageInput, grayImageOutput, PhotoreceptorsCompression,
			ganglionCellsCompression);
}

// run the initilized retina filter in order to perform gray image tone mapping, after this call all retina outputs are updated
void RetinaFilter::_runGrayToneMapping(const std::valarray<float> &grayImageInput,
		std::valarray<float> &grayImageOutput, const float PhotoreceptorsCompression,
		const float ganglionCellsCompression)
{
	// stability controls value update
	++_ellapsedFramesSinceLastReset;

	std::valarray<float> temp2(grayImageInput.size());

	// apply tone mapping on the multiplexed image
	// -> photoreceptors local adaptation (large area adaptation)
	_photoreceptorsPrefilter.runFilter_LPfilter(grayImageInput, grayImageOutput, 2); // compute low pass filtering modeling the horizontal cells filtering to acess local luminance
	_photoreceptorsPrefilter.setV0CompressionParameterToneMapping(1.f - PhotoreceptorsCompression,
			grayImageOutput.max(),
			1.f * grayImageOutput.sum() / (float) _photoreceptorsPrefilter.getNBpixels());
	_photoreceptorsPrefilter.runFilter_LocalAdapdation(grayImageInput, grayImageOutput, temp2); // adapt contrast to local luminance

	// -> ganglion cells local adaptation (short area adaptation)
	_photoreceptorsPrefilter.runFilter_LPfilter(temp2, grayImageOutput, 1); // compute low pass filtering (high cut frequency (remove spatio-temporal noise)
	_photoreceptorsPrefilter.setV0CompressionParameterToneMapping(1.f - ganglionCellsCompression,
			temp2.max(), 1.f * temp2.sum() / (float) _photoreceptorsPrefilter.getNBpixels());
	_photoreceptorsPrefilter.runFilter_LocalAdapdation(temp2, grayImageOutput, grayImageOutput); // adapt contrast to local luminance
}

// run the initilized retina filter in order to perform color tone mapping, after this call all retina outputs are updated
void RetinaFilter::runRGBToneMapping(const std::valarray<float> &RGBimageInput,
		std::valarray<float> &RGBimageOutput, const bool useAdaptiveFiltering,
		const float PhotoreceptorsCompression, const float ganglionCellsCompression)
{
	// preliminary check
	if (!checkInput(RGBimageInput, true))
		return;

	// multiplex the image with the color sampling method specified in the constructor
	_colorEngine.runColorMultiplexing(RGBimageInput);

	// apply tone mapping on the multiplexed image
	_runGrayToneMapping(_colorEngine.getMultiplexedFrame(), RGBimageOutput,
			PhotoreceptorsCompression, ganglionCellsCompression);

	// demultiplex tone maped image
	_colorEngine.runColorDemultiplexing(RGBimageOutput, useAdaptiveFiltering,
			_photoreceptorsPrefilter.getMaxInputValue()); //_ColorEngine->getMultiplexedFrame());//_ParvoRetinaFilter->getPhotoreceptorsLPfilteringOutput());

	// rescaling result between 0 and 255
	_colorEngine.normalizeRGBOutput_0_maxOutputValue(255.0);

	// return the result
	RGBimageOutput = _colorEngine.getDemultiplexedColorFrame();
}

void RetinaFilter::runLMSToneMapping(const std::valarray<float> &, std::valarray<float> &,
		const bool, const float, const float)
{
	std::cerr << "not working, sorry" << std::endl;

	/*  // preliminary check
	 const std::valarray<float> &bufferInput=checkInput(LMSimageInput, true);
	 if (!bufferInput)
	 return NULL;

	 if (!_useColorMode)
	 std::cerr<<"RetinaFilter::Can not call tone mapping oeration if the retina filter was created for gray scale images"<<std::endl;

	 // create a temporary buffer of size nrows, Mcolumns, 3 layers
	 std::valarray<float> lmsTempBuffer(LMSimageInput);
	 std::cout<<"RetinaFilter::--->min LMS value="<<lmsTempBuffer.min()<<std::endl;

	 // setup local adaptation parameter at the photoreceptors level
	 setV0CompressionParameter(PhotoreceptorsCompression, _maxInputValue);
	 // get the local energy of each color channel
	 // ->L
	 _spatiotemporalLPfilter(LMSimageInput, _filterOutput, 1);
	 setV0CompressionParameterToneMapping(PhotoreceptorsCompression, _maxInputValue, this->sum()/_NBpixels);
	 _localLuminanceAdaptation(LMSimageInput, _filterOutput, lmsTempBuffer.Buffer());
	 // ->M
	 _spatiotemporalLPfilter(LMSimageInput+_NBpixels, _filterOutput, 1);
	 setV0CompressionParameterToneMapping(PhotoreceptorsCompression, _maxInputValue, this->sum()/_NBpixels);
	 _localLuminanceAdaptation(LMSimageInput+_NBpixels, _filterOutput, lmsTempBuffer.Buffer()+_NBpixels);
	 // ->S
	 _spatiotemporalLPfilter(LMSimageInput+_NBpixels*2, _filterOutput, 1);
	 setV0CompressionParameterToneMapping(PhotoreceptorsCompression, _maxInputValue, this->sum()/_NBpixels);
	 _localLuminanceAdaptation(LMSimageInput+_NBpixels*2, _filterOutput, lmsTempBuffer.Buffer()+_NBpixels*2);

	 // eliminate negative values
	 for (unsigned int i=0;i<lmsTempBuffer.size();++i)
	 if (lmsTempBuffer.Buffer()[i]<0)
	 lmsTempBuffer.Buffer()[i]=0;
	 std::cout<<"RetinaFilter::->min LMS value="<<lmsTempBuffer.min()<<std::endl;

	 // compute LMS to A Cr1 Cr2 color space conversion
	 _applyImageColorSpaceConversion(lmsTempBuffer.Buffer(), lmsTempBuffer.Buffer(), _LMStoACr1Cr2);

	 TemplateBuffer <float> acr1cr2TempBuffer(_NBrows, _NBcolumns, 3);
	 memcpy(acr1cr2TempBuffer.Buffer(), lmsTempBuffer.Buffer(), sizeof(float)*_NBpixels*3);

	 // compute A Cr1 Cr2 to LMS color space conversion
	 _applyImageColorSpaceConversion(acr1cr2TempBuffer.Buffer(), lmsTempBuffer.Buffer(), _ACr1Cr2toLMS);

	 // eliminate negative values
	 for (unsigned int i=0;i<lmsTempBuffer.size();++i)
	 if (lmsTempBuffer.Buffer()[i]<0)
	 lmsTempBuffer.Buffer()[i]=0;

	 // rewrite output to the appropriate buffer
	 _colorEngine->setDemultiplexedColorFrame(lmsTempBuffer.Buffer());
	 */
}

// return image with center Parvo and peripheral Magno channels
void RetinaFilter::_processRetinaParvoMagnoMapping()
{
	float *hybridParvoMagnoPTR = &_retinaParvoMagnoMappedFrame[0];
	const float *parvoOutputPTR = get_data(_ParvoRetinaFilter.getOutput());
	const float *magnoXOutputPTR = get_data(_MagnoRetinaFilter.getOutput());
	float *hybridParvoMagnoCoefTablePTR = &_retinaParvoMagnoMapCoefTable[0];

	for (unsigned int i = 0; i < _photoreceptorsPrefilter.getNBpixels();
			++i, hybridParvoMagnoCoefTablePTR += 2) {
		float hybridValue = *(parvoOutputPTR++) * *(hybridParvoMagnoCoefTablePTR)
				+ *(magnoXOutputPTR++) * *(hybridParvoMagnoCoefTablePTR + 1);
		*(hybridParvoMagnoPTR++) = hybridValue;
	}

	TemplateBuffer<float>::normalizeGrayOutput_0_maxOutputValue(&_retinaParvoMagnoMappedFrame[0],
			_photoreceptorsPrefilter.getNBpixels());
}

bool RetinaFilter::getParvoFoveaResponse(std::valarray<float> &parvoFovealResponse)
{
	if (!_useParvoOutput)
		return false;
	if (parvoFovealResponse.size() != _ParvoRetinaFilter.getNBpixels())
		return false;

	const float *parvoOutputPTR = get_data(_ParvoRetinaFilter.getOutput());
	float *fovealParvoResponsePTR = &parvoFovealResponse[0];
	float *hybridParvoMagnoCoefTablePTR = &_retinaParvoMagnoMapCoefTable[0];

	for (unsigned int i = 0; i < _photoreceptorsPrefilter.getNBpixels();
			++i, hybridParvoMagnoCoefTablePTR += 2) {
		*(fovealParvoResponsePTR++) = *(parvoOutputPTR++) * *(hybridParvoMagnoCoefTablePTR);
	}

	return true;
}

// method to retrieve the parafoveal magnocellular pathway response (no energy motion in fovea)
bool RetinaFilter::getMagnoParaFoveaResponse(std::valarray<float> &magnoParafovealResponse)
{
	if (!_useMagnoOutput)
		return false;
	if (magnoParafovealResponse.size() != _MagnoRetinaFilter.getNBpixels())
		return false;

	const float *magnoXOutputPTR = get_data(_MagnoRetinaFilter.getOutput());
	float *parafovealMagnoResponsePTR = &magnoParafovealResponse[0];
	float *hybridParvoMagnoCoefTablePTR = &_retinaParvoMagnoMapCoefTable[0] + 1;

	for (unsigned int i = 0; i < _photoreceptorsPrefilter.getNBpixels();
			++i, hybridParvoMagnoCoefTablePTR += 2) {
		*(parafovealMagnoResponsePTR++) = *(magnoXOutputPTR++) * *(hybridParvoMagnoCoefTablePTR);
	}

	return true;
}
} // end of namespace bioinspired
} // end of namespace cv