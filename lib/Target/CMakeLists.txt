cmake_minimum_required(VERSION 3.18)

aux_source_directory(. SRC)
aux_source_directory(./X86 SRC)

add_library(Target STATIC ${SRC})
target_include_directories(Target PRIVATE ${CMAKE_SOURCE_DIR}/include)
