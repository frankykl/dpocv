project(DetectNet)

set (CMAKE_CXX_STANDARD 11)
#set (CMAKE_BUILD_TYPE Debug)
FIND_PACKAGE(OpenCV REQUIRED)

set(the_target RunDetector)
file(GLOB SRC_FILES RunDetector.cpp DpDetector.cpp )

add_executable(${the_target} ${SRC_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${the_target} ${OpenCV_LIBS})
