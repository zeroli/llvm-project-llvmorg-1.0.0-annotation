cmake_minimum_required(VERSION 3.18)

project(llvm)

set(CMAKE_CXX_STANDARD 11)
add_subdirectory(lib/Support)
add_subdirectory(lib/VMCore)
add_subdirectory(lib/Bytecode)
add_subdirectory(lib/CWriter)
add_subdirectory(lib/ExecutionEngine)
add_subdirectory(lib/Transforms)
add_subdirectory(lib/Analysis)
