#ifndef __MOSSETRACKER_HPP__
#define __MOSSETRACKER_HPP__

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/tracking.hpp"
#include <vector>

using namespace std;

namespace cv {
namespace dp_tracking {

class CMOSSETracker : public Tracker {
public:
    virtual ~CMOSSETracker() {};

	static Ptr<CMOSSETracker> create();

	virtual bool initImpl(const Mat& image, const Rect2d& boundingBox) = 0;
	virtual bool updateImpl(const Mat& image, Rect2d& boundingBox) = 0;
    virtual void SetBoundingBox(const Mat& image, Rect2d boundingBox) = 0;
};

} // namespace dp_tracking
} // namespace cv

#endif /* __MOSSETRACKER_HPP__ */
