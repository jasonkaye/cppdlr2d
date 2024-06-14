
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
cmake_policy(VERSION 3.20)

project(nddlr VERSION 0.1 LANGUAGES C CXX)

# Set default install path to my_build_directory/install if none specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "default install path" FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT CACHE BOOL FALSE FORCE)
  message(STATUS "No installation path specified; setting to ${CMAKE_INSTALL_PREFIX}")
endif()

# -- Add google test --
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG        main
)

# -- Add cppdlr --
# Tell cppdlr that google test has already been added
set_property(GLOBAL PROPERTY GTest_FOUND TRUE)
FetchContent_Declare(
  cppdlr
  GIT_REPOSITORY https://github.com/flatironinstitute/cppdlr
  GIT_TAG        main
)

FetchContent_MakeAvailable(googletest cppdlr)

enable_testing()

add_subdirectory(src)
add_subdirectory(programs)


#cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
#cmake_policy(VERSION 3.20)
#
#project(nddlr VERSION 0.1 LANGUAGES C CXX)
#
#find_package(cppdlr REQUIRED)
#
#
## -- Add google test --
#include(FetchContent)
#FetchContent_Declare(
#  googletest
#  GIT_REPOSITORY https://github.com/google/googletest
#  GIT_TAG        main
#)
#FetchContent_MakeAvailable(googletest)
#
#enable_testing()
#
#add_subdirectory(src)
#add_subdirectory(test)