
set(program_sources
  hubatom.cpp
  )

foreach(source_file ${program_sources})
  STRING(REGEX REPLACE ".cpp" "" program_name "${source_file}")
  add_executable(${program_name} ${source_file})
  target_link_libraries(${program_name} nddlr_c)
  target_include_directories(${program_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/)
endforeach(source_file)