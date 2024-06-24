# Set main program files
set(main_program_sources
  hubatom_allfuncs.cpp
  hubatom_lam_s.cpp
  hubatom_chi_d.cpp
  hubatom_allfuncs_manylambda.cpp
  )

# Set implementation files
set(implementation_sources
  hubatom.cpp
  )

foreach(main_source ${main_program_sources})
  get_filename_component(program_name ${main_source} NAME_WE)
  add_executable(${program_name} ${main_source} ${implementation_sources})
  target_link_libraries(${program_name} PRIVATE nddlr_c)
  target_include_directories(${program_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/)
endforeach()