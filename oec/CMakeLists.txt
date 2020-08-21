add_subdirectory(open-earth-compiler)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else ()
  message(SEND_ERROR "OEC requires CUDA")
endif ()
find_library(CUDA_RUNTIME_LIBRARY cuda)

add_library(dl_stencil dl_stencil.cpp)
target_include_directories(dl_stencil PUBLIC
  ${Python3_INCLUDE_DIRS}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
target_link_libraries(dl_stencil PUBLIC
  ${Python3_LIBRARIES}
  ${CUDA_RUNTIME_LIBRARY}
  pybind11
  )
