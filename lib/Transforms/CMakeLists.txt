cmake_minimum_required(VERSION 3.18)

aux_source_directory(. SRC)
aux_source_directory(./Hello SRC)
aux_source_directory(./Instrumentation SRC)
aux_source_directory(./IPO SRC)
aux_source_directory(./Scalar SRC)
aux_source_directory(./Utils SRC)

add_library(Transforms STATIC ${SRC})
target_include_directories(Transforms PRIVATE ${CMAKE_SOURCE_DIR}/include)
