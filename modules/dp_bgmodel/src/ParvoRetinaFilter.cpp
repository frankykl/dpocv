#include "precomp.hpp"
#include "ParvoRetinaFilter.hpp"

#include <iostream>
#include <cmath>

namespace cv {
namespace dp_bgmodel {
//////////////////////////////////////////////////////////
//                 OPL RETINA FILTER
//////////////////////////////////////////////////////////

// Constructor and Desctructor of the OPL retina filter

ParvoRetinaFilter::ParvoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns) :
		BasicRetinaFilter(NBrows, NBcolumns, 3), _photoreceptorsOutput(NBrows * NBcolumns), _horizontalCellsOutput(
				NBrows * NBcolumns), _parvocellularOutputON(NBrows * NBcolumns), _parvocellularOutputOFF(
				NBrows * NBcolumns), _bipolarCellsOutputON(NBrows * NBcolumns), _bipolarCellsOutputOFF(
				NBrows * NBcolumns), _localAdaptationOFF(NBrows * NBcolumns)
{
	// link to the required local parent adaptation buffers
	_localAdaptationON = &_localBuffer;
	_parvocellularOutputONminusOFF = &_filterOutput;
	// (*_localAdaptationON)=&_localBuffer;
	// (*_parvocellularOutputONminusOFF)=&(BasicRetinaFilter::TemplateBuffer);

	// init: set all the values to 0
	clearAllBuffers();

#ifdef OPL_RETINA_ELEMENT_DEBUG
	std::cout<<"ParvoRetinaFilter::Init OPL retina filter at specified frame size OK\n"<<std::endl;
#endif

}

ParvoRetinaFilter::~ParvoRetinaFilter()
{

#ifdef OPL_RETINA_ELEMENT_DEBUG
	std::cout<<"ParvoRetinaFilter::Delete OPL retina filter OK"<<std::endl;
#endif
}

////////////////////////////////////
// functions of the PARVO filter
////////////////////////////////////

// function that clears all buffers of the object
void ParvoRetinaFilter::clearAllBuffers()
{
	BasicRetinaFilter::clearAllBuffers();
	_photoreceptorsOutput = 0;
	_horizontalCellsOutput = 0;
	_parvocellularOutputON = 0;
	_parvocellularOutputOFF = 0;
	_bipolarCellsOutputON = 0;
	_bipolarCellsOutputOFF = 0;
	_localAdaptationOFF = 0;
}

/**
 * resize parvo retina filter object (resize all allocated buffers
 * @param NBrows: the new height size
 * @param NBcolumns: the new width size
 */
void ParvoRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
	BasicRetinaFilter::resize(NBrows, NBcolumns);
	_photoreceptorsOutput.resize(NBrows * NBcolumns);
	_horizontalCellsOutput.resize(NBrows * NBcolumns);
	_parvocellularOutputON.resize(NBrows * NBcolumns);
	_parvocellularOutputOFF.resize(NBrows * NBcolumns);
	_bipolarCellsOutputON.resize(NBrows * NBcolumns);
	_bipolarCellsOutputOFF.resize(NBrows * NBcolumns);
	_localAdaptationOFF.resize(NBrows * NBcolumns);

	// link to the required local parent adaptation buffers
	_localAdaptationON = &_localBuffer;
	_parvocellularOutputONminusOFF = &_filterOutput;

	// clean buffers
	clearAllBuffers();
}

// change the parameters of the filter
void ParvoRetinaFilter::setOPLandParvoFiltersParameters(const float beta1, const float tau1,
		const float k1, const float beta2, const float tau2, const float k2)
{
	// init photoreceptors low pass filter
	setLPfilterParameters(beta1, tau1, k1);
	// init horizontal cells low pass filter
	setLPfilterParameters(beta2, tau2, k2, 1);
	// init parasol ganglion cells low pass filter (default parameters)
	setLPfilterParameters(0, tau1, k1, 2);

}

// update/set size of the frames

// run filter for a new frame input
// output return is (*_parvocellularOutputONminusOFF)
const std::valarray<float> &ParvoRetinaFilter::runFilter(const std::valarray<float> &inputFrame,
		const bool useParvoOutput)
{
	_spatiotemporalLPfilter(get_data(inputFrame), &_photoreceptorsOutput[0]);
	_spatiotemporalLPfilter(&_photoreceptorsOutput[0], &_horizontalCellsOutput[0], 1);
	_OPL_OnOffWaysComputing();

	if (useParvoOutput) {
		// local adaptation processes on ON and OFF ways
		_spatiotemporalLPfilter(&_bipolarCellsOutputON[0], &(*_localAdaptationON)[0], 2);
		_localLuminanceAdaptation(&_parvocellularOutputON[0], &(*_localAdaptationON)[0]);

		_spatiotemporalLPfilter(&_bipolarCellsOutputOFF[0], &_localAdaptationOFF[0], 2);
		_localLuminanceAdaptation(&_parvocellularOutputOFF[0], &_localAdaptationOFF[0]);

		//// Final loop that computes the main output of this filter
		//
		//// loop that makes the difference between photoreceptor cells output and horizontal cells
		//// positive part goes on the ON way, negative pat goes on the OFF way
		float *parvocellularOutputONminusOFF_PTR = &(*_parvocellularOutputONminusOFF)[0];
		float *parvocellularOutputON_PTR = &_parvocellularOutputON[0];
		float *parvocellularOutputOFF_PTR = &_parvocellularOutputOFF[0];

		for (unsigned int i = 0; i < _filterOutput.getNBpixels(); i++)
			parvocellularOutputONminusOFF_PTR[i] = parvocellularOutputON_PTR[i]
					- parvocellularOutputOFF_PTR[i];
	}
	return (*_parvocellularOutputONminusOFF);
}

void ParvoRetinaFilter::_OPL_OnOffWaysComputing() // WARNING : this method requires many buffer accesses, parallelizing can increase bandwith & core efficacy
{
	// loop that makes the difference between photoreceptor cells output and horizontal cells
	// positive part goes on the ON way, negative pat goes on the OFF way

#ifdef MAKE_PARALLEL
	cv::parallel_for_(cv::Range(0, _filterOutput.getNBpixels()),
			Parallel_OPL_OnOffWaysComputing(&_photoreceptorsOutput[0], &_horizontalCellsOutput[0],
					&_bipolarCellsOutputON[0], &_bipolarCellsOutputOFF[0],
					&_parvocellularOutputON[0], &_parvocellularOutputOFF[0]));
#else
	float *photoreceptorsOutput_PTR = &_photoreceptorsOutput[0];
	float *horizontalCellsOutput_PTR = &_horizontalCellsOutput[0];
	float *bipolarCellsON_PTR = &_bipolarCellsOutputON[0];
	float *bipolarCellsOFF_PTR = &_bipolarCellsOutputOFF[0];
	float *parvocellularOutputON_PTR = &_parvocellularOutputON[0];
	float *parvocellularOutputOFF_PTR = &_parvocellularOutputOFF[0];
	// compute bipolar cells response equal to photoreceptors minus horizontal cells response
	// and copy the result on parvo cellular outputs... keeping time before their local contrast adaptation for final result
	for (unsigned int IDpixel = 0; IDpixel < _filterOutput.getNBpixels(); ++IDpixel)
	{
		float pixelDifference = *(photoreceptorsOutput_PTR++) - *(horizontalCellsOutput_PTR++);
		// test condition to allow write pixelDifference in ON or OFF buffer and 0 in the over
		float isPositive = (float) (pixelDifference > 0.0f);

		// ON and OFF channels writing step
		*(parvocellularOutputON_PTR++) = *(bipolarCellsON_PTR++) = isPositive * pixelDifference;
		*(parvocellularOutputOFF_PTR++) = *(bipolarCellsOFF_PTR++) = (isPositive - 1.0f) * pixelDifference;
	}
#endif
}
}    // end of namespace bioinspired
}    // end of namespace cv
