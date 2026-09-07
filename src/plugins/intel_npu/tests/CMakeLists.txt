# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (MSVC)
    # NPU plugin tests don't have macros that have `defined`
    # in their expansion
    ov_add_compiler_flags(/wd5105)
endif()

option(ENABLE_VPUX_COMPILER_TEST_EXTENSION
    "Build the VPUX compiler functional-test extension against the OpenVINO NPU test infrastructure" OFF)
set(VPUX_COMPILER_SOURCE_DIR "" CACHE PATH "Path to the VPUX compiler source tree for the test extension")

add_subdirectory(functional)

if(ENABLE_VPUX_COMPILER_TEST_EXTENSION)
    if(NOT VPUX_COMPILER_SOURCE_DIR OR
       NOT EXISTS "${VPUX_COMPILER_SOURCE_DIR}/tests/functional/connector/CMakeLists.txt")
        message(FATAL_ERROR
            "ENABLE_VPUX_COMPILER_TEST_EXTENSION requires VPUX_COMPILER_SOURCE_DIR pointing to a VPUX compiler checkout")
    endif()
    add_subdirectory(
        "${VPUX_COMPILER_SOURCE_DIR}/tests/functional/connector"
        "${CMAKE_CURRENT_BINARY_DIR}/vpux_compiler_test_extension"
    )
endif()

add_subdirectory(unit)
