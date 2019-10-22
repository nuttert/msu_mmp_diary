cmake_minimum_required(VERSION 3.10) # change this to your needs

project(foo VERSION 0.0 LANGUAGES CXX C)


#Год плюсов
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-stdlib=libc++ -std=c++17 -I/usr/local/include")



find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######
find_package(MTL)

if(MTL_FOUND)
    message(FOUND\ LIBRARY: MTL)
    include_directories(${MTL_INCLUDE_DIRS})
    add_definitions(${MTL_CXX_DEFINITIONS})
endif()
######

######
find_package(Boost REQUIRED COMPONENTS system thread)

if(Boost_FOUND)
    message(FOUND\ LIBRARY: Boost)
    include_directories(${Boost_INCLUDE_DIRS})
    add_definitions(${Boost_CXX_DEFINITIONS})
endif()
######


set(CMAKE_SWIG_FLAGS "")



add_subdirectory(cpp_modules)
add_subdirectory(bridge)
