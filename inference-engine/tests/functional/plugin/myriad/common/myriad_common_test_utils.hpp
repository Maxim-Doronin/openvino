// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <string>

namespace CommonTestUtils {
namespace vpu {

inline bool CheckMyriad2() {
    if (const auto& envVar = std::getenv("IE_VPU_MYRIADX")) {
        return std::stoi(envVar) == 0;
    }
    return true;
}

}  // namespace vpu
}  // namespace CommonTestUtils
