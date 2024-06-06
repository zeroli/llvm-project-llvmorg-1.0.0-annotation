cmake_minimum_required(VERSION 3.18)

aux_source_directory(. SRC)
aux_source_directory(./Interpreter SRC)
aux_source_directory(./JIT SRC)

add_library(ExecutionEngine STATIC ${SRC})
target_include_directories(ExecutionEngine PRIVATE ${CMAKE_SOURCE_DIR}/include)
