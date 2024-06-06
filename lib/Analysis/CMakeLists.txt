cmake_minimum_required(VERSION 3.18)

aux_source_directory(. SRC)
aux_source_directory(./DataStructure SRC)
aux_source_directory(./IPA SRC)
aux_source_directory(./LiveVar SRC)

add_library(Analysis STATIC ${SRC})
target_include_directories(Analysis PRIVATE ${CMAKE_SOURCE_DIR}/include)
