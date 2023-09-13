function(ConfigureExecutable EXE_NAME EXE_SRC EXE_DEST)
    add_executable(${EXE_NAME} "${EXE_SRC}")
    set_target_properties(${EXE_NAME} PROPERTIES
                                          CUDA_ARCHITECTURES ${GPU_ARCHS}
                                          RUNTIME_OUTPUT_DIRECTORY "${EXE_DEST}")
    target_include_directories(${EXE_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${EXE_NAME} PRIVATE gallatin)

    if (GAL_DYNAMIC)

      message("Adding resolve target property to ${EXE_NAME}")
      set_target_properties(${EXE_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)  # this is required for some reason
    endif(GAL_DYNAMIC)


endfunction(ConfigureExecutable)