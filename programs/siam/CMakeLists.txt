# Set main program files
set(main_program_sources
  siam_allfuncs.cpp
  siam_allfuncs_manyeps.cpp
  )

# Set implementation files
set(implementation_sources
  siam.cpp
  )

# Main programs
foreach(main_source ${main_program_sources})
  get_filename_component(program_name ${main_source} NAME_WE)
  add_executable(${program_name} ${main_source} ${implementation_sources})
  target_link_libraries(${program_name} PRIVATE nddlr_c)
  target_include_directories(${program_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/)
endforeach()