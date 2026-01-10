function(parse_params_to_defs PARAMS_FILE OUT_DEFS OUT_DPU_DEF_FLAGS)
  if(NOT EXISTS ${PARAMS_FILE})
    message(FATAL_ERROR "Params file not found: ${PARAMS_FILE}")
  endif()

  set(_parser "${CMAKE_SOURCE_DIR}/scripts/params_to_cmake.py")
  if(NOT EXISTS ${_parser})
    message(FATAL_ERROR "Params parser script missing: ${_parser}")
  endif()

  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${_parser} --params-file ${PARAMS_FILE} --format cmake-defs
    OUTPUT_VARIABLE _defs_raw
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _status
  )
  if(NOT _status EQUAL 0)
    message(FATAL_ERROR "Failed to parse runtime params from ${PARAMS_FILE} (status ${_status})")
  endif()

  string(REGEX REPLACE "[\r\n]+" ";" _defs_raw "${_defs_raw}")
  separate_arguments(_defs_raw)
  set(${OUT_DEFS} ${_defs_raw} PARENT_SCOPE)

  set(_dpu_flags "")
  foreach(_def ${_defs_raw})
    list(APPEND _dpu_flags "-D${_def}")
  endforeach()
  set(${OUT_DPU_DEF_FLAGS} "${_dpu_flags}" PARENT_SCOPE)
endfunction()
