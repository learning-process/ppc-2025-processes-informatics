# tasks/gusev_d_sentence_count/CMakeLists.txt

find_package(MPI REQUIRED)

add_library(gusev_d_sentence_count_mpi 
    mpi/src/ops_mpi.cpp
)

target_include_directories(gusev_d_sentence_count_mpi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
)

target_link_libraries(gusev_d_sentence_count_mpi PUBLIC MPI::MPI_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gusev_d_sentence_count_mpi PRIVATE --coverage)
    target_link_options(gusev_d_sentence_count_mpi PRIVATE --coverage)
endif()

add_library(gusev_d_sentence_count_seq 
    seq/src/ops_seq.cpp
)

target_include_directories(gusev_d_sentence_count_seq PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/seq/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gusev_d_sentence_count_seq PRIVATE --coverage)
    target_link_options(gusev_d_sentence_count_seq PRIVATE --coverage)
endif()
