#include "precomp.hpp"
#include <iostream>
#include "MagnoRetinaFilter.hpp"
#include <cmath>

namespace cv {
namespace dp_bgmodel {
// Constructor and Desctructor of the OPL retina filter
MagnoRetinaFilter::MagnoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns) :
		BasicRetinaFilter(NBrows, NBcolumns, 2), _previousInput_ON(NBrows * NBcolumns), _previousInput_OFF(
				NBrows * NBcolumns), _amacrinCellsTempOutput_ON(NBrows * NBcolumns), _amacrinCellsTempOutput_OFF(
				NBrows * NBcolumns), _magnoXOutputON(NBrows * NBcolumns), _magnoXOutputOFF(
				NBrows * NBcolumns), _localProcessBufferON(NBrows * NBcolumns), _localProcessBufferOFF(
				NBrows * NBcolumns)
{
	_magnoYOutput = &_filterOutput;
	_magnoYsaturated = &_localBuffer;

	clearAllBuffers();

#ifdef IPL_RETINA_ELEMENT_DEBUG
	std::cout<<"MagnoRetinaFilter::Init IPL retina filter at specified frame size OK"<<std::endl;
#endif
}

MagnoRetinaFilter::~MagnoRetinaFilter()
{
#ifdef IPL_RETINA_ELEMENT_DEBUG
	std::cout<<"MagnoRetinaFilter::Delete IPL retina filter OK"<<std::endl;
#endif
}

// function that clears all buffers of the object
void MagnoRetinaFilter::clearAllBuffers()
{
	BasicRetinaFilter::clearAllBuffers();
	_previousInput_ON = 0;
	_previousInput_OFF = 0;
	_amacrinCellsTempOutput_ON = 0;
	_amacrinCellsTempOutput_OFF = 0;
	_magnoXOutputON = 0;
	_magnoXOutputOFF = 0;
	_localProcessBufferON = 0;
	_localProcessBufferOFF = 0;

}

/**
 * resize retina magno filter object (resize all allocated buffers
 * @param NBrows: the new height size
 * @param NBcolumns: the new width size
 */
void MagnoRetinaFilter::resize(const unsigned int NBrows, const unsigned int NBcolumns)
{
	BasicRetinaFilter::resize(NBrows, NBcolumns);
	_previousInput_ON.resize(NBrows * NBcolumns);
	_previousInput_OFF.resize(NBrows * NBcolumns);
	_amacrinCellsTempOutput_ON.resize(NBrows * NBcolumns);
	_amacrinCellsTempOutput_OFF.resize(NBrows * NBcolumns);
	_magnoXOutputON.resize(NBrows * NBcolumns);
	_magnoXOutputOFF.resize(NBrows * NBcolumns);
	_localProcessBufferON.resize(NBrows * NBcolumns);
	_localProcessBufferOFF.resize(NBrows * NBcolumns);

	// to be sure, relink buffers
	_magnoYOutput = &_filterOutput;
	_magnoYsaturated = &_localBuffer;

	// reset all buffers
	clearAllBuffers();
}

void MagnoRetinaFilter::setCoefficientsTable(const float parasolCells_beta,
		const float parasolCells_tau, const float parasolCells_k,
		const float amacrinCellsTemporalCutFrequency, const float localAdaptIntegration_tau,
		const float localAdaptIntegration_k)
{
	_temporalCoefficient = (float) std::exp(-1.0f / amacrinCellsTemporalCutFrequency);
	// the first set of parameters is dedicated to the low pass filtering property of the ganglion cells
	BasicRetinaFilter::setLPfilterParameters(parasolCells_beta, parasolCells_tau, parasolCells_k,
			0);
	// the second set of parameters is dedicated to the ganglion cells output intergartion for their local adaptation property
	BasicRetinaFilter::setLPfilterParameters(0, localAdaptIntegration_tau, localAdaptIntegration_k,
			1);
}

void MagnoRetinaFilter::_amacrineCellsComputing(const float *OPL_ON, const float *OPL_OFF)
{
#ifdef MAKE_PARALLEL
	cv::parallel_for_(cv::Range(0, _filterOutput.getNBpixels()),
			Parallel_amacrineCellsComputing(OPL_ON, OPL_OFF, &_previousInput_ON[0],
					&_previousInput_OFF[0], &_amacrinCellsTempOutput_ON[0],
					&_amacrinCellsTempOutput_OFF[0], _temporalCoefficient));
#else
	const float *OPL_ON_PTR=OPL_ON;
	const float *OPL_OFF_PTR=OPL_OFF;
	float *previousInput_ON_PTR= &_previousInput_ON[0];
	float *previousInput_OFF_PTR= &_previousInput_OFF[0];
	float *amacrinCellsTempOutput_ON_PTR= &_amacrinCellsTempOutput_ON[0];
	float *amacrinCellsTempOutput_OFF_PTR= &_amacrinCellsTempOutput_OFF[0];

	for (unsigned int IDpixel=0; IDpixel<this->getNBpixels(); ++IDpixel)
	{

		/* Compute ON and OFF amacrin cells high pass temporal filter */
		float magnoXonPixelResult = _temporalCoefficient*(*amacrinCellsTempOutput_ON_PTR+ *OPL_ON_PTR-*previousInput_ON_PTR);
		*(amacrinCellsTempOutput_ON_PTR++)=((float)(magnoXonPixelResult>0))*magnoXonPixelResult;

		float magnoXoffPixelResult = _temporalCoefficient*(*amacrinCellsTempOutput_OFF_PTR+ *OPL_OFF_PTR-*previousInput_OFF_PTR);
		*(amacrinCellsTempOutput_OFF_PTR++)=((float)(magnoXoffPixelResult>0))*magnoXoffPixelResult;

		/* prepare next loop */
		*(previousInput_ON_PTR++)=*(OPL_ON_PTR++);
		*(previousInput_OFF_PTR++)=*(OPL_OFF_PTR++);

	}
#endif
}

/*
 * Magno retina filter.
 * OPL_ON: OPL on channel input
 * OPL_OFF: OPL off channel input
 * Return: Magno output
 */
const std::valarray<float> &MagnoRetinaFilter::runFilter(const std::valarray<float> &OPL_ON,
		const std::valarray<float> &OPL_OFF)
{
	// Compute the high pass temporal filter
	_amacrineCellsComputing(get_data(OPL_ON), get_data(OPL_OFF));

	// apply low pass filtering on ON and OFF ways after temporal high pass filtering
	_spatiotemporalLPfilter(&_amacrinCellsTempOutput_ON[0], &_magnoXOutputON[0], 0);
	_spatiotemporalLPfilter(&_amacrinCellsTempOutput_OFF[0], &_magnoXOutputOFF[0], 0);

	// local adaptation of the ganglion cells to the local contrast of the moving contours
	_spatiotemporalLPfilter(&_magnoXOutputON[0], &_localProcessBufferON[0], 1);
	_localLuminanceAdaptation(&_magnoXOutputON[0], &_localProcessBufferON[0]);
	_spatiotemporalLPfilter(&_magnoXOutputOFF[0], &_localProcessBufferOFF[0], 1);
	_localLuminanceAdaptation(&_magnoXOutputOFF[0], &_localProcessBufferOFF[0]);

	/* Compute MagnoY */
	float *magnoYOutput = &(*_magnoYOutput)[0];
	float *magnoXOutputON_PTR = &_magnoXOutputON[0];
	float *magnoXOutputOFF_PTR = &_magnoXOutputOFF[0];
	for (unsigned int i = 0; i < _filterOutput.getNBpixels(); i++)
		magnoYOutput[i] = magnoXOutputON_PTR[i] + magnoXOutputOFF_PTR[i];

	return (*_magnoYOutput);
}
} // end of namespace bioinspired
} // end of namespace cv
