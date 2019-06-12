#ifndef MagnoRetinaFilter_H_
#define MagnoRetinaFilter_H_

/**
 * @class MagnoRetinaFilter
 * @brief class which describes the magnocellular channel of the retina:
 * -> performs a moving contours extraction with powerfull local data enhancement
 *
 * TYPICAL USE:
 *
 * // create object at a specified picture size
 * MagnoRetinaFilter *movingContoursExtractor;
 * movingContoursExtractor =new MagnoRetinaFilter(frameSizeRows, frameSizeColumns);
 *
 * // init gain, spatial and temporal parameters:
 * movingContoursExtractor->setCoefficientsTable(0, 0.7, 5, 3);
 *
 * // during program execution, call the filter for contours extraction for an input picture called "FrameBuffer":
 * movingContoursExtractor->runfilter(FrameBuffer);
 *
 * // get the output frame, check in the class description below for more outputs:
 * const float *movingContours=movingContoursExtractor->getMagnoYsaturated();
 *
 * // at the end of the program, destroy object:
 * delete movingContoursExtractor;

 * @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC : www.listic.univ-savoie.fr, Gipsa-Lab, France: www.gipsa-lab.inpg.fr/
 * Creation date 2007
 * Based on Alexandre BENOIT thesis: "Le système visuel humain au secours de la vision par ordinateur"
 */

#include "BasicRetinaFilter.hpp"

//#define _IPL_RETINA_ELEMENT_DEBUG

namespace cv {
namespace dp_bgmodel {
class MagnoRetinaFilter: public BasicRetinaFilter {
public:
	/**
	 * constructor parameters are only linked to image input size
	 * @param NBrows: number of rows of the input image
	 * @param NBcolumns: number of columns of the input image
	 */
	MagnoRetinaFilter(const unsigned int NBrows, const unsigned int NBcolumns);

	/**
	 * destructor
	 */
	virtual ~MagnoRetinaFilter();

	/**
	 * function that clears all buffers of the object
	 */
	void clearAllBuffers();

	/**
	 * resize retina magno filter object (resize all allocated buffers)
	 * @param NBrows: the new height size
	 * @param NBcolumns: the new width size
	 */
	void resize(const unsigned int NBrows, const unsigned int NBcolumns);

	/**
	 * set parameters values
	 * @param parasolCells_beta: the low pass filter gain used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), typical value is 0
	 * @param parasolCells_tau: the low pass filter time constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is frame, typical value is 0 (immediate response)
	 * @param parasolCells_k: the low pass filter spatial constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is pixels, typical value is 5
	 * @param amacrinCellsTemporalCutFrequency: the time constant of the first order high pass fiter of the magnocellular way (motion information channel), unit is frames, tipicall value is 5
	 * @param localAdaptIntegration_tau: specifies the temporal constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
	 * @param localAdaptIntegration_k: specifies the spatial constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
	 */
	void setCoefficientsTable(const float parasolCells_beta, const float parasolCells_tau,
			const float parasolCells_k, const float amacrinCellsTemporalCutFrequency,
			const float localAdaptIntegration_tau, const float localAdaptIntegration_k);

	/**
	 * launch filter that runs all the IPL magno filter (model of the magnocellular channel of the Inner Plexiform Layer of the retina)
	 * @param OPL_ON: the output of the bipolar ON cells of the retina (available from the ParvoRetinaFilter class (getBipolarCellsON() function)
	 * @param OPL_OFF: the output of the bipolar OFF cells of the retina (available from the ParvoRetinaFilter class (getBipolarCellsOFF() function)
	 * @return the processed result without post-processing
	 */
	const std::valarray<float> &runFilter(const std::valarray<float> &OPL_ON,
			const std::valarray<float> &OPL_OFF);

	/**
	 * @return the Magnocellular ON channel filtering output
	 */
	inline const std::valarray<float> &getMagnoON() const
	{
		return _magnoXOutputON;
	}

	/**
	 * @return the Magnocellular OFF channel filtering output
	 */
	inline const std::valarray<float> &getMagnoOFF() const
	{
		return _magnoXOutputOFF;
	}

	/**
	 * @return the Magnocellular Y (sum of the ON and OFF magno channels) filtering output
	 */
	inline const std::valarray<float> &getMagnoYsaturated() const
	{
		return *_magnoYsaturated;
	}

	/**
	 * applies an image normalization which saturates the high output values by the use of an assymetric sigmoide
	 */
	inline void normalizeGrayOutputNearZeroCentreredSigmoide()
	{
		_filterOutput.normalizeGrayOutputNearZeroCentreredSigmoide(&(*_magnoYOutput)[0],
				&(*_magnoYsaturated)[0]);
	}

	/**
	 * @return the horizontal cells' temporal constant
	 */
	inline float getTemporalConstant()
	{
		return _filteringCoeficientsTable[2];
	}

private:

	// related pointers to these buffers
	std::valarray<float> _previousInput_ON;
	std::valarray<float> _previousInput_OFF;
	std::valarray<float> _amacrinCellsTempOutput_ON;
	std::valarray<float> _amacrinCellsTempOutput_OFF;
	std::valarray<float> _magnoXOutputON;
	std::valarray<float> _magnoXOutputOFF;
	std::valarray<float> _localProcessBufferON;
	std::valarray<float> _localProcessBufferOFF;
	// reference to parent buffers and allow better readability
	TemplateBuffer<float> *_magnoYOutput;
	std::valarray<float> *_magnoYsaturated;

	// varialbles
	float _temporalCoefficient;

	// amacrine cells filter : high pass temporal filter
	void _amacrineCellsComputing(const float *ONinput, const float *OFFinput);
#ifdef MAKE_PARALLEL
	/******************************************************
	 ** IF some parallelizing thread methods are available, then, main loops are parallelized using these functors
	 ** ==> main idea paralellise main filters loops, then, only the most used methods are parallelized... TODO : increase the number of parallelised methods as necessary
	 ** ==> functors names = Parallel_$$$ where $$$= the name of the serial method that is parallelised
	 ** ==> functors constructors can differ from the parameters used with their related serial functions
	 */
	class Parallel_amacrineCellsComputing: public cv::ParallelLoopBody {
	private:
		const float *OPL_ON, *OPL_OFF;
		float *previousInput_ON, *previousInput_OFF, *amacrinCellsTempOutput_ON,
				*amacrinCellsTempOutput_OFF;
		float temporalCoefficient;
	public:
		Parallel_amacrineCellsComputing(const float *OPL_ON_PTR, const float *OPL_OFF_PTR,
				float *previousInput_ON_PTR, float *previousInput_OFF_PTR,
				float *amacrinCellsTempOutput_ON_PTR, float *amacrinCellsTempOutput_OFF_PTR,
				float temporalCoefficientVal) :
				OPL_ON(OPL_ON_PTR), OPL_OFF(OPL_OFF_PTR), previousInput_ON(previousInput_ON_PTR), previousInput_OFF(
						previousInput_OFF_PTR), amacrinCellsTempOutput_ON(
						amacrinCellsTempOutput_ON_PTR), amacrinCellsTempOutput_OFF(
						amacrinCellsTempOutput_OFF_PTR), temporalCoefficient(temporalCoefficientVal)
		{
		}

		virtual void operator()(const Range& r) const CV_OVERRIDE {
			const float *OPL_ON_PTR=OPL_ON+r.start;
			const float *OPL_OFF_PTR=OPL_OFF+r.start;
			float *previousInput_ON_PTR= previousInput_ON+r.start;
			float *previousInput_OFF_PTR= previousInput_OFF+r.start;
			float *amacrinCellsTempOutput_ON_PTR= amacrinCellsTempOutput_ON+r.start;
			float *amacrinCellsTempOutput_OFF_PTR= amacrinCellsTempOutput_OFF+r.start;

			for (int IDpixel=r.start; IDpixel!=r.end; ++IDpixel)
			{

				/* Compute ON and OFF amacrin cells high pass temporal filter */
				float magnoXonPixelResult = temporalCoefficient*(*amacrinCellsTempOutput_ON_PTR+ *OPL_ON_PTR-*previousInput_ON_PTR);
				*(amacrinCellsTempOutput_ON_PTR++)=((float)(magnoXonPixelResult>0))*magnoXonPixelResult;

				float magnoXoffPixelResult = temporalCoefficient*(*amacrinCellsTempOutput_OFF_PTR+ *OPL_OFF_PTR-*previousInput_OFF_PTR);
				*(amacrinCellsTempOutput_OFF_PTR++)=((float)(magnoXoffPixelResult>0))*magnoXoffPixelResult;

				/* prepare next loop */
				*(previousInput_ON_PTR++)=*(OPL_ON_PTR++);
				*(previousInput_OFF_PTR++)=*(OPL_OFF_PTR++);

			}
		}

	};
#endif
};

} // end of namespace bioinspired
} // end of namespace cv

#endif /*MagnoRetinaFilter_H_*/
