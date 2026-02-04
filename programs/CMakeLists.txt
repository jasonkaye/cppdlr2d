set(program_sources
  generate_dlr2d_if.cpp
  est_rank_tst.cpp
)

foreach(source_file ${program_sources})
  STRING(REGEX REPLACE ".cpp" "" program_name "${source_file}")
  add_executable(${program_name} ${source_file})
  target_link_libraries(${program_name} ${PROJECT_NAME}::${PROJECT_NAME}_c fmt::fmt)
endforeach(source_file)

add_subdirectory(hubatom)
add_subdirectory(siam)
add_subdirectory(hubatom_mc)