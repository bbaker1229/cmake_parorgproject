
cmake_minimum_required (VERSION 3.8)

project ("Project1_delme" LANGUAGES CUDA C)

add_subdirectory(tools)

# Basic Matrix Multiplication
add_executable (dense_mult "matmult_basic.c" "tools/tools.h" "tools/tools.c")
target_link_libraries(dense_mult PUBLIC tools)
target_include_directories(dense_mult PUBLIC
						   "${PROJECT_BINARY_DIR}"
						   "${PROJECT_SOURCE_DIR}/tools"
)

# Basic Sparse Matrix Multiplication
add_executable (sparse_mult "sparsemult_basic.c" "tools/tools.h" "tools/tools.c")
target_link_libraries(sparse_mult PUBLIC tools)
target_include_directories(sparse_mult PUBLIC
						   "${PROJECT_BINARY_DIR}"
						   "${PROJECT_SOURCE_DIR}/tools"
)

# Matrix Multiplication using OpenMP
add_executable (omp_dense "openmp_matmult.c" "tools/tools.h" "tools/tools.c")
find_package(OpenMP)
target_link_libraries(omp_dense PUBLIC OpenMP::OpenMP_C)
target_link_libraries(omp_dense PUBLIC tools)
target_include_directories(omp_dense PUBLIC
						   "${PROJECT_BINARY_DIR}"
						   "${PROJECT_SOURCE_DIR}/tools"
)

# Sparse Matrix Multiplication using OpenMP
add_executable (omp_sparse "openmp_sparse.c" "tools/tools.h" "tools/tools.c")
find_package(OpenMP)
target_link_libraries(omp_sparse PUBLIC OpenMP::OpenMP_C)
target_link_libraries(omp_sparse PUBLIC tools)
target_include_directories(omp_sparse PUBLIC
						   "${PROJECT_BINARY_DIR}"
						   "${PROJECT_SOURCE_DIR}/tools"
)

# Matrix Multiplication using MPI
add_executable (mpi_dense "mpi_dense.c" "tools/tools.h" "tools/tools.c")
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(mpi_dense ${MPI_LIBRARIES})
if(MPI_COMPILE_FLAGS)
  set_target_properties(mpi_dense PROPERTIES 
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
  set_target_properties(mpi_dense PROPERTIES 
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif() 
target_include_directories(mpi_dense PUBLIC
						   "${PROJECT_BINARY_DIR}"
						   "${PROJECT_SOURCE_DIR}/tools"
)

# Sparse Matrix Multiplication using MPI
add_executable (mpi_sparse "mpi_sparse.c" "tools/tools.h" "tools/tools.c")
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(mpi_sparse ${MPI_LIBRARIES})
if(MPI_COMPILE_FLAGS)
  set_target_properties(mpi_sparse PROPERTIES 
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
  set_target_properties(mpi_sparse PROPERTIES 
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif() 
target_include_directories(mpi_sparse PUBLIC
						   "${PROJECT_BINARY_DIR}"
						   "${PROJECT_SOURCE_DIR}/tools"
)

# Matrix Multiplication using Cuda
find_package(CUDA)
add_executable(cuda_dense "cuda_dense.cu" "tools/tools.h" "tools/tools.c")
set_target_properties(cuda_dense PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_dense PROPERTIES CUDA_ARCHITECTURES "72")
target_link_libraries(cuda_dense PRIVATE tools)
include_directories("${CUDA_INCLUDE_DIRS}")
target_include_directories(cuda_dense PUBLIC
						   "${CMAKE_CURRENT_SOURCE_DIR}"
						   "${PROJECT_BINARY_DIR}"
						   "${PROJECT_SOURCE_DIR}/tools"
						   "${PROJECT_SOURCE_DIR}/Common"
						   )

# Sparse Matrix Multiplication using Cuda
find_package(CUDA)
add_executable(cuda_sparse "cuda_sparse.cu" "tools/tools.h" "tools/tools.c")
set_target_properties(cuda_sparse PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_sparse PROPERTIES CUDA_ARCHITECTURES "72")
target_link_libraries(cuda_sparse PRIVATE tools)
include_directories("${CUDA_INCLUDE_DIRS}")
target_include_directories(cuda_sparse PUBLIC
						   "${CMAKE_CURRENT_SOURCE_DIR}"
						   "${PROJECT_BINARY_DIR}"
						   "${PROJECT_SOURCE_DIR}/tools"
						   "${PROJECT_SOURCE_DIR}/Common"
						   )

# TODO: Add tests and install targets if needed.
