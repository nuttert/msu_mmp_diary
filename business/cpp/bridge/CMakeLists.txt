cmake_minimum_required(VERSION 3.10) # change this to your needs

set_source_files_properties(ML.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(ML.i PROPERTIES SWIG_FLAGS "-includeall")


SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../python_modules")
SET(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/../python_modules")

swig_add_library(ML LANGUAGE python SOURCES ML.i OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../python_modules" OUTFILE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../python_modules") 
target_include_directories(ML PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)



swig_link_libraries(ML ${PYTHON_LIBRARIES} ML)
