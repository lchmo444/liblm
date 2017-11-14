
file(GLOB LIBLM_FILES "liblm/*.h" "liblm/*.cpp")
add_library(liblm SHARED ${LIBLM_FILES})