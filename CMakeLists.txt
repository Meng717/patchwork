cmake_minimum_required(VERSION 3.0.2)
project(patchwork)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0057 NEW)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# option(USE_SYSTEM_TBB "Use system pre-installed oneAPI/tbb" OFF)
# include(3rdparty/find_dependencies.cmake)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(Boost 1.54 REQUIRED)
find_package(TBB REQUIRED)

pybind11_add_module(patchwork src/estimate.cpp)

target_include_directories(patchwork PRIVATE 
    ${PCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    include
)
target_link_libraries(patchwork PRIVATE
    pybind11::module
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    TBB::tbb
)