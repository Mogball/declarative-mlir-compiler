cmake_minimum_required(VERSION 3.10)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

project(declarative-compiler)

# Enable RTTI and exceptions, as required by Boost
set(LLVM_ENABLE_RTTI ON CACHE INTERNAL "")
set(LLVM_ENABLE_EH   ON CACHE INTERNAL "")
# Configure LLVM and MLIR directories
set(LLVM_PROJECT_DIR llvm-project)
set(LLVM_DIR ${LLVM_PROJECT_DIR}/llvm)
set(MLIR_DIR ${LLVM_PROJECT_DIR}/mlir)
set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR}/${LLVM_DIR})
set(MLIR_BINARY_DIR ${LLVM_BINARY_DIR}/tools/mlir)
# Add LLVM and MLIR targets
set(LLVM_TARGETS_TO_BUILD "X86" CACHE INTERNAL "")
set(LLVM_ENABLE_PROJECTS "mlir" CACHE INTERNAL "")
add_subdirectory(${LLVM_DIR})
# Configure mlir-tblgen from built target
set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
# Configure LLVM and MLIR include directories
set(LLVM_INCLUDE_DIRS ${LLVM_DIR}/include ${LLVM_BINARY_DIR}/include)
set(MLIR_INCLUDE_DIRS ${MLIR_DIR}/include ${MLIR_BINARY_DIR}/include)

message(STATUS "Building project in directory: ${CMAKE_BINARY_DIR}")
message(STATUS "LLVM directory: ${LLVM_DIR}")
message(STATUS "MLIR directory: ${MLIR_DIR}")

# Include project-specific configs
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/module")
include(AddDMC)

# Boost::Python
find_package(PythonLibs 3 REQUIRED)
## Explicitly specify `python38` as `python` gives a Python 2 warning
find_package(Boost REQUIRED COMPONENTS python38)

# Global options
add_compile_options(-Wall -Werror -fdiagnostics-color)

# Global includes
include_directories(
  include
  ${CMAKE_BINARY_DIR}/include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  )

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)


### TEST
python_add_module(hello hello.cpp)
target_include_directories(hello PUBLIC ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(hello ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
