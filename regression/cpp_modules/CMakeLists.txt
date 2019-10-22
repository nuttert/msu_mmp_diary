cmake_minimum_required(VERSION 3.10) # change this to your needs

#Ищем все исполняемые файлы
file(GLOB_RECURSE _SOURSCES "*.cpp")
file(GLOB_RECURSE _HEADERS "*.hpp")



add_library(ML STATIC ${_SOURSCES} ${_HEADERS})
target_include_directories(ML PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
