cmake_minimum_required(VERSION 3.8)
project(include/liblm)
include_directories(include)


set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCE_FILES "src/*.cpp" "include/liblm/*.h")

set(demo demo.cpp)

add_library(liblm SHARED ${SOURCE_FILES})
add_executable(demo ${demo})

set_target_properties(liblm PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(demo liblm)


set(RUNTIME_OUTPUT_DIRECTORY bin)
set(RUNTIME_OUTPUT_NAME lm)

set(LIBRARY_OUTPUT_DIRECTORY bin)
set(LIBRARY_OUTPUT_NAME liblm)