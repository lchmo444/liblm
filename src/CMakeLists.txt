
file(GLOB LIBLM_FILES "liblm/*.h" "liblm/*.cpp")
add_library(liblm SHARED ${LIBLM_FILES})

#INSTALL(FILES liblm/c_api.h DESTINATION include)
#if(WIN32)
#    INSTALL(TARGETS liblm RUNTIME DESTINATION lib)
#else()
#    INSTALL(TARGETS liblm LIBRARY DESTINATION lib)
#endif()
