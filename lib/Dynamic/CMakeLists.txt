add_library(DMCDynamic
  DynamicContext.cpp
  DynamicDialect.cpp
  DynamicObject.cpp
  TypeIDAllocator.cpp
  )
target_link_libraries(DMCDynamic MLIRIR)
