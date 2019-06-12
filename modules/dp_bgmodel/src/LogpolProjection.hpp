#ifndef IMAGELOGPOLPROJECTION_H_
#define IMAGELOGPOLPROJECTION_H_

/**
 * @class ImageLogPolProjection
 * @brief class able to perform a log sampling of an image input (models the log sampling of the photoreceptors of the retina)
 * or a log polar projection which models the retina information projection on the primary visual cortex: a linear projection in the center for detail analysis and a log projection of the borders (low spatial frequency motion information in general)
 *
 * collaboration: Barthelemy DURETTE who experimented the retina log projection
 -> "Traitement visuels Bio mimtiques pour la supplance perceptive", internal technical report, May 2005, Gipsa-lab/DIS, Grenoble, FRANCE
 *
 * * TYPICAL USE:
 *
 * // create object, here for a log sampling (keyword:RETINALOGPROJECTION): (dynamic object allocation sample)
 * ImageLogPolProjection *imageSamplingTool;
 * imageSamplingTool = new ImageLogPolProjection(frameSizeRows, frameSizeColumns, RETINALOGPROJECTION);
 *
 * // init log projection:
 * imageSamplingTool->initProjection(1.0, 15.0);
 *
 * // during program execution, call the log transform applied to a frame called "FrameBuffer" :
 * imageSamplingTool->runProjection(FrameBuffer);
 * // get output frame and its size:
 * const unsigned int logSampledFrame_nbRows=imageSamplingTool->getOutputNBrows();
 * const unsigned int logSampledFrame_nbColumns=imageSamplingTool->getOutputNBcolumns();
 * const double *logSampledFrame=imageSamplingTool->getSampledFrame();
 *
 * // at the end of the program, destroy object:
 * delete imageSamplingTool;
 *
 * @author Alexandre BENOIT, benoit.alexandre.vision@gmail.com, LISTIC : www.listic.univ-savoie.fr, Gipsa-Lab, France: www.gipsa-lab.inpg.fr/
 * Creation date 2007
 */

//#define __IMAGELOGPOLPROJECTION_DEBUG // used for std output debug information
#include "BasicRetinaFilter.hpp"

namespace cv {
namespace dp_bgmodel {

class ImageLogPolProjection: public BasicRetinaFilter {
public:

	enum PROJECTIONTYPE {
		RETINALOGPROJECTION, CORTEXLOGPOLARPROJECTION
	};

	/**
	 * constructor, just specifies the image input size and the projection type, no projection initialisation is done
	 * -> use initLogRetinaSampling() or initLogPolarCortexSampling() for that
	 * @param nbRows: number of rows of the input image
	 * @param nbColumns: number of columns of the input image
	 * @param projection: the type of projection, RETINALOGPROJECTION or CORTEXLOGPOLARPROJECTION
	 * @param colorMode: specifies if the projection is applied on a grayscale image (false) or color images (3 layers) (true)
	 */
	ImageLogPolProjection(const unsigned int nbRows, const unsigned int nbColumns,
			const PROJECTIONTYPE projection, const bool colorMode = false);

	/**
	 * standard destructor
	 */
	virtual ~ImageLogPolProjection();

	/**
	 * function that clears all buffers of the object
	 */
	void clearAllBuffers();

	/**
	 * resize retina color filter object (resize all allocated buffers)
	 * @param NBrows: the new height size
	 * @param NBcolumns: the new width size
	 */
	void resize(const unsigned int NBrows, const unsigned int NBcolumns);

	/**
	 * init function depending on the projection type
	 * @param reductionFactor: the size reduction factor of the ouptup image in regard of the size of the input image, must be superior to 1
	 * @param samplingStrenght: specifies the strenght of the log compression effect (magnifying coefficient)
	 * @return true if the init was performed without any errors
	 */
	bool initProjection(const double reductionFactor, const double samplingStrenght);

	/**
	 * main funtion of the class: run projection function
	 * @param inputFrame: the input frame to be processed
	 * @param colorMode: the input buffer color mode: false=gray levels, true = 3 color channels mode
	 * @return the output frame
	 */
	std::valarray<float> &runProjection(const std::valarray<float> &inputFrame,
			const bool colorMode = false);

	/**
	 * @return the numbers of rows (height) of the images OUTPUTS of the object
	 */
	inline unsigned int getOutputNBrows()
	{
		return _outputNBrows;
	}

	/**
	 * @return the numbers of columns (width) of the images OUTPUTS of the object
	 */
	inline unsigned int getOutputNBcolumns()
	{
		return _outputNBcolumns;
	}

	/**
	 * main funtion of the class: run projection function
	 * @param size: one of the input frame initial dimensions to be processed
	 * @return the output frame dimension
	 */
	inline static unsigned int predictOutputSize(const unsigned int size,
			const double reductionFactor)
	{
		return (unsigned int) ((double) size / reductionFactor);
	}

	/**
	 * @return the output of the filter which applies an irregular Low Pass spatial filter to the imag input (see function
	 */
	inline const std::valarray<float> &getIrregularLPfilteredInputFrame() const
	{
		return _irregularLPfilteredFrame;
	}

	/**
	 * function which allows to retrieve the output frame which was updated after the "runProjection(...) function BasicRetinaFilter::runProgressiveFilter(...)
	 * @return the projection result
	 */
	inline const std::valarray<float> &getSampledFrame() const
	{
		return _sampledFrame;
	}

	/**
	 * function which allows gives the tranformation table, its size is (getNBrows()*getNBcolumns()*2)
	 * @return the transformation matrix [outputPixIndex_i, inputPixIndex_i, outputPixIndex_i+1, inputPixIndex_i+1....]
	 */
	inline const std::valarray<unsigned int> &getSamplingMap() const
	{
		return _transformTable;
	}

	inline double getOriginalRadiusLength(const double projectedRadiusLength)
	{
		return _azero / (_alim - projectedRadiusLength * 2.0 / _minDimension);
	}

	//    unsigned int getInputPixelIndex(const unsigned int ){ return  _transformTable[index*2+1]};

private:
	PROJECTIONTYPE _selectedProjection;

	// size of the image output
	unsigned int _outputNBrows;
	unsigned int _outputNBcolumns;
	unsigned int _outputNBpixels;
	unsigned int _outputDoubleNBpixels;
	unsigned int _inputDoubleNBpixels;

	// is the object able to manage color flag
	bool _colorModeCapable;
	// sampling strenght factor
	double _samplingStrenght;
	// sampling reduction factor
	double _reductionFactor;

	// log sampling parameters
	double _azero;
	double _alim;
	double _minDimension;

	// template buffers
	std::valarray<float> _sampledFrame;
	std::valarray<float>&_tempBuffer;
	std::valarray<unsigned int> _transformTable;

	std::valarray<float> &_irregularLPfilteredFrame; // just a reference for easier understanding
	unsigned int _usefullpixelIndex;

	// init transformation tables
	bool _computeLogProjection();
	bool _computeLogPolarProjection();

	// specifies if init was done correctly
	bool _initOK;
	// private init projections functions called by "initProjection(...)" function
	bool _initLogRetinaSampling(const double reductionFactor, const double samplingStrenght);
	bool _initLogPolarCortexSampling(const double reductionFactor, const double samplingStrenght);

	ImageLogPolProjection(const ImageLogPolProjection&);
	ImageLogPolProjection& operator=(const ImageLogPolProjection&);

};

} // end of namespace bioinspired
} // end of namespace cv
#endif /*IMAGELOGPOLPROJECTION_H_*/
