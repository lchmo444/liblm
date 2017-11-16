cmake_minimum_required(VERSION 3.8)
project(liblm)

# Eigen dependence.
include_directories(include)

# Use OpenMP support.
IF (APPLE)
    message(STATUS "clang does not support OpenMP!")
ELSEIF (WIN32 OR UNIX)
    find_package(OpenMP REQUIRED)
    IF(OPENMP_FOUND)
        message(STATUS "OPENMP FOUND!")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    ENDIF()
ENDIF ()

# Compile Eigen need C++11.
set(CMAKE_CXX_STANDARD 11)

# Source files.
file(GLOB LIBLM_FILES "src/liblm/*.h" "src/liblm/*.cpp")

# A demo.
set(DEMO_FILES demo.cpp)

# Build library.
add_library(liblm SHARED ${LIBLM_FILES})

# Build demo.
add_executable(demo ${DEMO_FILES})

# C
set_target_properties(liblm PROPERTIES LINKER_LANGUAGE C)

# Link library with demo.
target_link_libraries(demo liblm)

