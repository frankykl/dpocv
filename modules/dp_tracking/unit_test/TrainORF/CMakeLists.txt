project(TrainORF)

FIND_PACKAGE(OpenCV REQUIRED)

set(the_target TrainORF)
file(GLOB SRC_FILES TrainORF.cpp
	../../src/ORF.h ../../src/FeatureSpace.h ../../src/ORFinternal.h
	../../src/HaarFeature.cpp ../../src/HOGFeature.cpp
	../../src/ORF.cpp ../../src/ORFTree.cpp ../../src/ORFNode.cpp
	../../src/ORFRandomTest.cpp ../../src/ORFUtils.cpp ../../src/ORFParameter.cpp)

add_executable(${the_target} ${SRC_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${the_target} ${OpenCV_LIBS})
