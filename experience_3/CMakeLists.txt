cmake_minimum_required(VERSION 3.23)
project(experience_3)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV) #搜索opencv
include_directories(${OpenCV_INCLUDE_DIRS}) #include路径

add_executable(experience_3 main.cpp)

target_link_libraries(experience_3 ${OpenCV_LIBS}) #链接项目和opencv库