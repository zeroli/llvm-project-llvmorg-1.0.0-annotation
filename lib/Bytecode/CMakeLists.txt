cmake_minimum_required(VERSION 3.18)

aux_source_directory(./Reader SRC)
aux_source_directory(./Writer SRC)
add_library(Bytecode STATIC ${SRC})
target_include_directories(Bytecode PRIVATE ${CMAKE_SOURCE_DIR}/include)
