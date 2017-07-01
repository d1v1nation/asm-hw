cmake_minimum_required(VERSION 3.8)
project(asmeme)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-masm=intel -O0 -mavx")
set(SOURCE_FILES memcpy.cpp)
add_executable(asmeme ${SOURCE_FILES})
#add_executable(tests tests.cpp)

#set(CMAKE_CXX_FLAGS "-masm=intel -O0 -mavx")
#add_executable(wc wcnt.cpp)