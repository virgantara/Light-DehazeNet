cmake_minimum_required(VERSION 3.10)
project(DehazeCAP)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(dehaze main.cpp CAP.cpp guidedfilter.cpp)
target_link_libraries(dehaze ${OpenCV_LIBS})
