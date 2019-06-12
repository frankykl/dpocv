#ifndef SEGMENTATIONMODULE_HPP_
#define SEGMENTATIONMODULE_HPP_

#include "opencv2/core.hpp" // for all OpenCV core functionalities access, including cv::Exception support

namespace cv {
namespace dp_bgmodel {
//! @addtogroup bioinspired
//! @{

/** @brief parameter structure that stores the transient events detector setup parameters
 */
struct SegmentationParameters { // CV_EXPORTS_W_MAP to export to python native dictionnaries
	// default structure instance construction with default values
	SegmentationParameters() :
			thresholdON(100), thresholdOFF(100), localEnergy_temporalConstant(0.5), localEnergy_spatialConstant(
					5), neighborhoodEnergy_temporalConstant(1), neighborhoodEnergy_spatialConstant(
					15), contextEnergy_temporalConstant(1), contextEnergy_spatialConstant(75)
	{
	}
	;
	// all properties list
	float thresholdON;
	float thresholdOFF;
	//! the time constant of the first order low pass filter, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 0.5 frame
	float localEnergy_temporalConstant;
	//! the spatial constant of the first order low pass filter, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 5 pixel
	float localEnergy_spatialConstant;
	//! local neighborhood energy filtering parameters : the aim is to get information about the energy neighborhood to perform a center surround energy analysis
	float neighborhoodEnergy_temporalConstant;
	float neighborhoodEnergy_spatialConstant;
	//! context neighborhood energy filtering parameters : the aim is to get information about the energy on a wide neighborhood area to filtered out local effects
	float contextEnergy_temporalConstant;
	float contextEnergy_spatialConstant;
};

/** @brief class which provides a transient/moving areas segmentation module

 perform a locally adapted segmentation by using the retina magno input data Based on Alexandre
 BENOIT thesis: "Le système visuel humain au secours de la vision par ordinateur"

 3 spatio temporal filters are used:
 - a first one which filters the noise and local variations of the input motion energy
 - a second (more powerfull low pass spatial filter) which gives the neighborhood motion energy the
 segmentation consists in the comparison of these both outputs, if the local motion energy is higher
 to the neighborhood otion energy, then the area is considered as moving and is segmented
 - a stronger third low pass filter helps decision by providing a smooth information about the
 "motion context" in a wider area
 */

class CV_EXPORTS_W TransientAreasSegmentationModule: public Algorithm {
public:

	/** @brief return the sze of the manage input and output images
	 */
	CV_WRAP
	virtual Size getSize()=0;

	/** @brief try to open an XML segmentation parameters file to adjust current segmentation instance setup

	 - if the xml file does not exist, then default setup is applied
	 - warning, Exceptions are thrown if read XML file is not valid
	 @param segmentationParameterFile : the parameters filename
	 @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
	 */
	CV_WRAP
	virtual void setup(String segmentationParameterFile = "",
			const bool applyDefaultSetupOnFailure = true)=0;

	/** @brief try to open an XML segmentation parameters file to adjust current segmentation instance setup

	 - if the xml file does not exist, then default setup is applied
	 - warning, Exceptions are thrown if read XML file is not valid
	 @param fs : the open Filestorage which contains segmentation parameters
	 @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
	 */
	virtual void setup(cv::FileStorage &fs, const bool applyDefaultSetupOnFailure = true)=0;

	/** @brief try to open an XML segmentation parameters file to adjust current segmentation instance setup

	 - if the xml file does not exist, then default setup is applied
	 - warning, Exceptions are thrown if read XML file is not valid
	 @param newParameters : a parameters structures updated with the new target configuration
	 */
	virtual void setup(SegmentationParameters newParameters)=0;

	/** @brief return the current parameters setup
	 */
	virtual SegmentationParameters getParameters()=0;

	/** @brief parameters setup display method
	 @return a string which contains formatted parameters information
	 */
	CV_WRAP
	virtual const String printSetup()=0;

	/** @brief write xml/yml formated parameters information
	 @param fs : the filename of the xml file that will be open and writen with formatted parameters information
	 */
	CV_WRAP
	virtual void write(String fs) const=0;

	/** @brief write xml/yml formated parameters information
	 @param fs : a cv::Filestorage object ready to be filled
	 */
	virtual void write(cv::FileStorage& fs) const CV_OVERRIDE = 0;

	/** @brief main processing method, get result using methods getSegmentationPicture()
	 @param inputToSegment : the image to process, it must match the instance buffer size !
	 @param channelIndex : the channel to process in case of multichannel images
	 */
	CV_WRAP
	virtual void run(InputArray inputToSegment, const int channelIndex = 0)=0;

	/** @brief access function
	 @return the last segmentation result: a boolean picture which is resampled between 0 and 255 for a display purpose
	 */
	CV_WRAP
	virtual void getSegmentationPicture(OutputArray transientAreas)=0;

	/** @brief cleans all the buffers of the instance
	 */
	CV_WRAP
	virtual void clearAllBuffers()=0;

	/** @brief allocator
	 @param inputSize : size of the images input to segment (output will be the same size)
	 */
	CV_WRAP
	static Ptr<TransientAreasSegmentationModule> create(Size inputSize);
};

//! @}

}
} // namespaces end : cv and bioinspired

#endif

