project(DNNProbe)

set (CMAKE_CXX_STANDARD 11)
FIND_PACKAGE(OpenCV REQUIRED)

set(the_target DNNProbe)
file(GLOB SRC_FILES DNNProbe.cpp)

add_executable(${the_target} ${SRC_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${the_target} ${OpenCV_LIBS})
