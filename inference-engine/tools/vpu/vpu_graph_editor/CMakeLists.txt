#
# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

function(add_graph_editor_target TARGET_NAME)
    file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cpp)

    add_executable(${TARGET_NAME} ${SOURCES})

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            inference_engine format_reader ngraph 
            Threads::Threads
        PUBLIC 
            pugixml vpu_common_lib
            ${NGRAPH_LIBRARIES} 
            ngraphFunctions)

    target_include_directories(${TARGET_NAME}
        SYSTEM PRIVATE
            "${IE_MAIN_SOURCE_DIR}/src/legacy_api/include"
            "${IE_MAIN_SOURCE_DIR}/src/plugin_api"
            "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
            "${IE_MAIN_SOURCE_DIR}/tests/ngraph_functions/include"
            "${IE_MAIN_SOURCE_DIR}/samples/common/samples"
            "${IE_MAIN_SOURCE_DIR}/samples/common/format_reader"
            "${IE_MAIN_SOURCE_DIR}/tests/ie_test_utils"
            "${IE_MAIN_SOURCE_DIR}/tools/vpu/common")
endfunction()

add_graph_editor_target(vpu_graph_editor)


