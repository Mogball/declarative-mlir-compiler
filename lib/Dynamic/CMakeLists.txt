add_library(DMCDynamic
    DynamicDialect.cpp
    TypeIDAllocator.cpp
    )
target_link_libraries(DMCDynamic MLIRIR)
