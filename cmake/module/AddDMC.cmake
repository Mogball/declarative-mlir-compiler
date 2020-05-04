# LLVM sets -fvisibility-inlines-hidden. We must match the setting or else
# the linker will give warnings.
if(NOT WIN32 AND NOT CYGWIN AND NOT (${CMAKE_SYSTEM_NAME} MATCHES "AIX" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
  check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
  append_if(SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG "-fvisibility-inlines-hidden" CMAKE_CXX_FLAGS)
endif()

