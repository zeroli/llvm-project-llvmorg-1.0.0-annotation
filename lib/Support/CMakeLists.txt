cmake_minimum_required(VERSION 3.18)

aux_source_directory(. SRC)
add_library(support STATIC ${SRC})
target_include_directories(support PRIVATE ${CMAKE_SOURCE_DIR}/include)
