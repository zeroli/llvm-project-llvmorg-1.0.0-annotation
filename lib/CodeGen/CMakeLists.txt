aux_source_directory(. SRC)
aux_source_directory(./InstrSched SRC)
aux_source_directory(./InstrSelection SRC)
aux_source_directory(./RegAlloc SRC)
aux_source_directory(./SelectionDAG SRC)

add_library(CodeGen STATIC ${SRC})
target_include_directories(CodeGen PRIVATE ${CMAKE_SOURCE_DIR}/include)
