add_library(nddlr_c STATIC
  polarization.cpp
  dlr2d.cpp
  utils.cpp
  )

target_link_libraries(nddlr_c cppdlr::cppdlr_c)

file(GLOB nddlr_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
set_target_properties(nddlr_c PROPERTIES PUBLIC_HEADER "${nddlr_HEADERS}")

install(TARGETS nddlr_c LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include/nddlr )
