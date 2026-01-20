# Include external dependency function
include(${CMAKE_CURRENT_SOURCE_DIR}/external_dependency.cmake)

# -- GTest (always build from source, exclude from all)
external_dependency(GTest
  GIT_REPO https://github.com/google/googletest
  GIT_TAG main
  BUILD_ALWAYS
  EXCLUDE_FROM_ALL
)

# -- nda (DEV_EXPR_INDEXED branch)
external_dependency(nda
  GIT_REPO https://github.com/triqs/nda
  GIT_TAG DEV_EXPR_INDEXED
  EXCLUDE_FROM_ALL
)

# -- cppdlr
external_dependency(cppdlr
  GIT_REPO https://github.com/flatironinstitute/cppdlr
  GIT_TAG main
  EXCLUDE_FROM_ALL
)
