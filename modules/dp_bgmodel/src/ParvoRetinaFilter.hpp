#ifndef ParvoRetinaFilter_H_
#define ParvoRetinaFilter_H_

/**
 * @class ParvoRetinaFilter
 * @brief class which describes the OPL retina model and the Inner Plexiform Layer parvocellular channel of the retina:
 * -> performs a contours extraction with powerfull local data enhancement as at the retina level
 * -> spectrum whitening occurs at the OPL (Outer Plexiform Layer) of the retina: corrects the 1/f spectrum tendancy of natural images
 * ---> enhances details with mid spatial frequencies, attenuates low spatial frequencies (luminance), attenuates high temporal frequencies and high spatial frequencies, etc.
 *
 * TYPICAL USE:
 *
 * // create object at a specified picture size
 * ParvoRetinaFilter *contoursExtractor;
 * contoursExtractor =new ParvoRetinaFilter(frameSizeRows, frameSizeColumns);
 *
 * // init gain, spatial and temporal parameters:
 * contoursExtractor->setCoefficientsTable(0, 0.7, 1, 0, 7, 1);
 *
 * // during program execution, call the filter for contours extraction for an input picture called "FrameBuffer":
 * contoursExtractor->runfilter(FrameBuffer);
 *
 * // get the output frame, check in the class description below for more outputs:
 * const float *contours=contoursExtractor->getParvoONminusOFF();
 *
 * // at the end of the program, destroy object:
 * delete contoursExtractor;

 * @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC : www.listic.univ-savoie.fr, Gipsa-Lab, France: www.gipsa-lab.inpg.fr/
 * Creation date 2007
 * Based on Alexandre BENOIT thesis: "Le système visuel humain au secours de la vision par ordinateur"
 *
 */

#include "BasicRetinaFilter.hpp"

//#define _OPL_RETINA_ELEMENT_DEBUG

namespace cv {
namespace dp_bgmodel {

//retina classes that derivate from the Basic Retrina class
class ParvoRetinaFilter: public BasicRetinaFilter {

public:
	/**
	 * constructor parameters are only linked to image input size
	 * @param NBrows: number of rows of the input image
	 * @param NBcolumns: number of columns of the input image
	 */
	ParvoRetinaFilter(const unsigned int NBrows = 480, const unsigned int NBcolumns = 640);

	/**
	 * standard desctructor
	 */
	virtual ~ParvoRetinaFilter();

	/**
	 * resize method, keeps initial parameters, all buffers are flushed
	 * @param NBrows: number of rows of the input image
	 * @param NBcolumns: number of columns of the input image
	 */
	void resize(const unsigned int NBrows, const unsigned int NBcolumns);

	/**
	 * function that clears all buffers of the object
	 */
	void clearAllBuffers();

	/**
	 * setup the OPL and IPL parvo channels
	 * @param beta1: gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, the amplitude is boosted but it should only be used for values rescaling... if needed
	 * @param tau1: the time constant of the first order low pass filter of the photoreceptors, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 1 frame
	 * @param k1: the spatial constant of the first order low pass filter of the photoreceptors, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 1 pixel
	 * @param beta2: gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, then, the luminance is not filtered and is still reachable at the output, typicall value is 0
	 * @param tau2: the time constant of the first order low pass filter of the horizontal cells, use it to cut low temporal frequencies (local luminance variations), unit is frames, typical value is 1 frame, as the photoreceptors
	 * @param k2: the spatial constant of the first order low pass filter of the horizontal cells, use it to cut low spatial frequencies (local luminance), unit is pixels, typical value is 5 pixel, this value is also used for local contrast computing when computing the local contrast adaptation at the ganglion cells level (Inner Plexiform Layer parvocellular channel model)
	 */
	void setOPLandParvoFiltersParameters(const float beta1, const float tau1, const float k1,
			const float beta2, const float tau2, const float k2);

	/**
	 * setup more precisely the low pass filter used for the ganglion cells low pass filtering (used for local luminance adaptation)
	 * @param tau: time constant of the filter (unit is frame for video processing)
	 * @param k: spatial constant of the filter (unit is pixels)
	 */
	void setGanglionCellsLocalAdaptationLPfilterParameters(const float tau, const float k)
	{
		BasicRetinaFilter::setLPfilterParameters(0, tau, k, 2);
	}  // change the parameters of the filter

	/**
	 * launch filter that runs the OPL spatiotemporal filtering and optionally finalizes IPL Pagno filter (model of the Parvocellular channel of the Inner Plexiform Layer of the retina)
	 * @param inputFrame: the input image to be processed, this can be the direct gray level input frame, but a better efficacy is expected if the input is preliminary processed by the photoreceptors local adaptation possible to acheive with the help of a BasicRetinaFilter object
	 * @param useParvoOutput: set true if the final IPL filtering step has to be computed (local contrast enhancement)
	 * @return the processed Parvocellular channel output (updated only if useParvoOutput is true)
	 * @details: in any case, after this function call, photoreceptors and horizontal cells output are updated, use getPhotoreceptorsLPfilteringOutput() and getHorizontalCellsOutput() to get them
	 * also, bipolar cells output are accessible (difference between photoreceptors and horizontal cells, ON output has positive values, OFF ouput has negative values), use the following access methods: getBipolarCellsON() and getBipolarCellsOFF()if useParvoOutput is true,
	 * if useParvoOutput is true, the complete Parvocellular channel is computed, more outputs are updated and can be accessed threw: getParvoON(), getParvoOFF() and their difference with getOutput()
	 */
	const std::valarray<float> &runFilter(const std::valarray<float> &inputFrame,
			const bool useParvoOutput = true); // output return is _parvocellularOutputONminusOFF

	/**
	 * @return the output of the photoreceptors filtering step (high cut frequency spatio-temporal low pass filter)
	 */
	inline const std::valarray<float> &getPhotoreceptorsLPfilteringOutput() const
	{
		return _photoreceptorsOutput;
	}

	/**
	 * @return the output of the photoreceptors filtering step (low cut frequency spatio-temporal low pass filter)
	 */
	inline const std::valarray<float> &getHorizontalCellsOutput() const
	{
		return _horizontalCellsOutput;
	}

	/**
	 * @return the output Parvocellular ON channel of the retina model
	 */
	inline const std::valarray<float> &getParvoON() const
	{
		return _parvocellularOutputON;
	}

	/**
	 * @return the output Parvocellular OFF channel of the retina model
	 */
	inline const std::valarray<float> &getParvoOFF() const
	{
		return _parvocellularOutputOFF;
	}

	/**
	 * @return the output of the Bipolar cells of the ON channel of the retina model same as function getParvoON() but without luminance local adaptation
	 */
	inline const std::valarray<float> &getBipolarCellsON() const
	{
		return _bipolarCellsOutputON;
	}

	/**
	 * @return the output of the Bipolar cells of the OFF channel of the retina model same as function getParvoON() but without luminance local adaptation
	 */
	inline const std::valarray<float> &getBipolarCellsOFF() const
	{
		return _bipolarCellsOutputOFF;
	}

	/**
	 * @return the photoreceptors's temporal constant
	 */
	inline float getPhotoreceptorsTemporalConstant()
	{
		return _filteringCoeficientsTable[2];
	}

	/**
	 * @return the horizontal cells' temporal constant
	 */
	inline float getHcellsTemporalConstant()
	{
		return _filteringCoeficientsTable[5];
	}

private:
	// template buffers
	std::valarray<float> _photoreceptorsOutput;
	std::valarray<float> _horizontalCellsOutput;
	std::valarray<float> _parvocellularOutputON;
	std::valarray<float> _parvocellularOutputOFF;
	std::valarray<float> _bipolarCellsOutputON;
	std::valarray<float> _bipolarCellsOutputOFF;
	std::valarray<float> _localAdaptationOFF;
	std::valarray<float> *_localAdaptationON;
	TemplateBuffer<float> *_parvocellularOutputONminusOFF;
	// private functions
	void _OPL_OnOffWaysComputing();

#ifdef MAKE_PARALLEL
	/******************************************************
	 ** IF some parallelizing thread methods are available, then, main loops are parallelized using these functors
	 ** ==> main idea paralellise main filters loops, then, only the most used methods are parallelized... TODO : increase the number of parallelised methods as necessary
	 ** ==> functors names = Parallel_$$$ where $$$= the name of the serial method that is parallelised
	 ** ==> functors constructors can differ from the parameters used with their related serial functions
	 */
	class Parallel_OPL_OnOffWaysComputing: public cv::ParallelLoopBody {
	private:
		float *photoreceptorsOutput, *horizontalCellsOutput, *bipolarCellsON, *bipolarCellsOFF,
				*parvocellularOutputON, *parvocellularOutputOFF;
	public:
		Parallel_OPL_OnOffWaysComputing(float *photoreceptorsOutput_PTR,
				float *horizontalCellsOutput_PTR, float *bipolarCellsON_PTR,
				float *bipolarCellsOFF_PTR, float *parvocellularOutputON_PTR,
				float *parvocellularOutputOFF_PTR) :
				photoreceptorsOutput(photoreceptorsOutput_PTR), horizontalCellsOutput(horizontalCellsOutput_PTR),
				bipolarCellsON(bipolarCellsON_PTR), bipolarCellsOFF(bipolarCellsOFF_PTR),
				parvocellularOutputON(parvocellularOutputON_PTR),
				parvocellularOutputOFF(parvocellularOutputOFF_PTR)
		{}

		virtual void operator()(const Range& r) const CV_OVERRIDE {
			// compute bipolar cells response equal to photoreceptors minus horizontal cells response
			// and copy the result on parvo cellular outputs... keeping time before their local contrast
			// adaptation for final result
			float *photoreceptorsOutput_PTR = photoreceptorsOutput + r.start;
			float *horizontalCellsOutput_PTR = horizontalCellsOutput + r.start;
			float *bipolarCellsON_PTR = bipolarCellsON + r.start;
			float *bipolarCellsOFF_PTR = bipolarCellsOFF + r.start;
			float *parvocellularOutputON_PTR = parvocellularOutputON + r.start;
			float *parvocellularOutputOFF_PTR = parvocellularOutputOFF + r.start;
			for (int i = 0; i < r.end - r.start; i++) {
				float pixelDifference = photoreceptorsOutput_PTR[i] - horizontalCellsOutput_PTR[i];
				parvocellularOutputON_PTR[i] = bipolarCellsON_PTR[i] =
						(pixelDifference > 0.0f) ? pixelDifference : 0.0f;
				parvocellularOutputOFF_PTR[i] = bipolarCellsOFF_PTR[i] =
						(pixelDifference > 0.0f) ? 0.0f : -pixelDifference;
			}
		}
	};
#endif

};

}  // end of namespace bioinspired
}  // end of namespace cv
#endif
