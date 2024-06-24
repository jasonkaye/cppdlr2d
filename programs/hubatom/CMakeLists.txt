# Set main program files
set(main_program_sources
  hubatom_allfuncs.cpp
  hubatom_allfuncs_manylambda.cpp
  )

# Set test program files
set(test_program_sources
  hubatom_test.cpp
  )

# Set implementation files
set(implementation_sources
  hubatom.cpp
  )

# Main programs
foreach(main_source ${main_program_sources})
  get_filename_component(program_name ${main_source} NAME_WE)
  add_executable(${program_name} ${main_source} ${implementation_sources})
  target_link_libraries(${program_name} PRIVATE nddlr_c)
  target_include_directories(${program_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/)
endforeach()

# Unit tests
foreach(test_source ${test_program_sources})
  get_filename_component(test_name ${test_source} NAME_WE)
  add_executable(${test_name} ${test_source} ${implementation_sources})
  target_link_libraries(${test_name} PRIVATE nddlr_c gtest_main)
  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()