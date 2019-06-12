/*
 * Copyright Deep Photon.
 */

/*
 * This class implements background modeler.
 */

#include "precomp.hpp"
#include "opencv2/core/utility.hpp"
#include "BGModel.hpp"

using namespace cv;

namespace cv {
namespace dp_bgmodel {

class DPBackgroundModelerImpl: public DPBackgroundModeler {
public:
	/*
	 * Constructor.
	 */
	DPBackgroundModelerImpl() {
		// Default Parameter Values. Override with algorithm "set" method.
		_name = "DPBackgroundModeler";
		_maxProcSize = Size(960, 600);
		_inputSize = Size(0, 0);
	}

	/*
	 * Destructor.
	 */
	~DPBackgroundModelerImpl()
	{
	}

	/*
	 * Process a frame.
	 */
	virtual void ProcessFrame(Mat& inputFrame) {
		// Initialize if input size changes
		if (inputFrame.size() != _inputSize) {
			_inputSize = inputFrame.size();
			Initialize();
		}

		// scale input
		Mat scaledFrame;
		resize(inputFrame, scaledFrame, _retinaSize);

		//int64 t0 = getTickCount();
		// run retina filter
		_retina->run(scaledFrame);
		//int64 t1 = getTickCount();
		//double time = (double)(t1 - t0) / getTickFrequency();
		//printf("retina time = %6.3f\n", time);

		Mat parvo, magno;
		_retina->getParvo(parvo);
		resize(parvo, _parvo, _inputSize);

		// analyze motion
		_retina->getMagno(magno);
		resize(magno, _motionMap, _inputSize);

		// generate motion mask
		threshold(_motionMap, _motionMask, 16, 128, THRESH_BINARY);

		//t0 = getTickCount();
		// run background modeling and generate foreground mask
		_bgModel.apply(inputFrame, _foreground, &_motionMask);
		threshold(_foreground, _foreground, 254, 255, THRESH_BINARY);
		_foreground &= _motionMask;
		//t1 = getTickCount();
		//time = (double)(t1 - t0) / getTickFrequency();
		//printf("bg time = %6.3f\n", time);
	}

	/*
	 * Get parvo output.
	 */
	virtual Mat& GetParvoOutput() {
		return _parvo;
	}

	/*
	 * Get motion map.
	 */
	virtual Mat& GetMotionMap() {
		return _motionMap;
	}

	/*
	 * Get motion mask.
	 */
	virtual Mat& GetMotionMask() {
		return _motionMask;
	}

	/*
	 * Get background image.
	 */
	virtual Mat& GetBackgroundFrame() {
		_bgModel.getBackgroundImage(_background);
		return _background;
	}

	/*
	 * Get foreground image.
	 */
	virtual Mat& GetForegroundFrame() {
		return _foreground;
	}

	virtual void write(FileStorage& fs) const {
		fs << "name" << _name;
	}

	virtual void read(const FileNode& fn) {
		CV_Assert((String) fn["name"] == _name);
	}

protected:
	/*
	 * Initialization.
	 */
	void Initialize() {
		// set proper retina processing size
		_retinaSize = _inputSize;
		if (_inputSize.width > _maxProcSize.width || _inputSize.height > _maxProcSize.height) {
			float scale = min((float)_maxProcSize.width / _inputSize.width,
					(float)_maxProcSize.height / _inputSize.height);
			_retinaSize.width = (int)(scale * _inputSize.width);
			_retinaSize.height = (int) (scale * _inputSize.height);
		}

		// create retina object
		_retina = Retina::create(_retinaSize);

		// save default retina parameters to file
		_retina->write("RetinaDefaultParameters.xml");

		// load parameters if file exists
		_retina->setup("RetinaSpecificParameters.xml");
		_retina->clearBuffers();
	}


private:
	String _name;       // class name
	Size _maxProcSize;  // max processing size
	Size _inputSize;    // input frame size
	Size _retinaSize;   // retina processing size
	Mat _parvo;         // parvo output
	Mat _motionMap;     // motion map (magno)
	Mat _motionMask;    // motion mask
	Mat _background;    // background image
	Mat _foreground;    // foreground image
	cv::Ptr<Retina> _retina;  // pointer of retina filter
	BGModelMOG _bgModel;  // background model
};

/*
 * Create a background modeler and return its pointer.
 */
Ptr<DPBackgroundModeler> CreateDPBackgroundModeler() {
	Ptr < DPBackgroundModeler > bg = makePtr<DPBackgroundModelerImpl>();
	return bg;
}

} // namespace dp_bgmodel
} // namespace cv

