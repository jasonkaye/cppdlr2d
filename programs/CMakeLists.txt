
set(program_sources
  dlr2dfit_mf.cpp
  dlr2dsingfit_mf.cpp
  hubatom.cpp
  generate_dlr2d_if.cpp
  siam_vertex.cpp
  est_rank_tst.cpp
  )

foreach(source_file ${program_sources})
  STRING(REGEX REPLACE ".cpp" "" program_name "${source_file}")
  add_executable(${program_name} ${source_file})
  #target_link_libraries(${program_name} nddlr_c gtest_main)
  target_link_libraries(${program_name} nddlr_c)
  target_include_directories(${program_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/)
  #add_test(NAME ${program_name} COMMAND ${program_name})
endforeach(source_file)
