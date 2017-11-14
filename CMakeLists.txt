cmake_minimum_required(VERSION 3.8)
project(liblm)
include_directories(include)

set(CMAKE_CXX_STANDARD 11)


file(GLOB LIBLM_FILES "src/liblm/*.h" "src/liblm/*.cpp")
add_library(liblm SHARED ${LIBLM_FILES})

set(demo demo.cpp)
add_executable(demo ${demo})


set_target_properties(liblm PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(demo liblm)


#set(RUNTIME_OUTPUT_DIRECTORY bin)
#set(RUNTIME_OUTPUT_NAME lm)
#
#set(LIBRARY_OUTPUT_DIRECTORY bin)
#set(LIBRARY_OUTPUT_NAME liblm)