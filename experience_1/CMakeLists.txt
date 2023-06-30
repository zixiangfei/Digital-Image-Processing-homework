cmake_minimum_required(VERSION 3.23)
project(experience_1)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED) #搜索opencv
include_directories(${OpenCV_INCLUDE_DIRS}) #include路径

add_executable(experience_1 main.cpp)


target_link_libraries(experience_1 ${OpenCV_LIBS}) #链接项目和opencv库