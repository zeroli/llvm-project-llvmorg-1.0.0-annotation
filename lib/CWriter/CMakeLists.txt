cmake_minimum_required(VERSION 3.18)

aux_source_directory(. SRC)
add_library(CWriter STATIC ${SRC})
target_include_directories(CWriter PRIVATE ${CMAKE_SOURCE_DIR}/include)
