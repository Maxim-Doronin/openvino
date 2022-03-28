// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/device_packet_length.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"

namespace vpu {

void DevicePacketLength::validate(const std::string& value) {
    int intValue;
    try {
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue > 0,
        R"(unexpected {} option value "{}", only positive numbers are supported)", key(), value);

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue % 1024 == 0,
        R"(unexpected {} option value "{}", value should be multiple of 1024)", key(), value);
    return;
}

void DevicePacketLength::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DevicePacketLength::key() {
    return InferenceEngine::MYRIAD_DEVICE_PACKET_LENGTH;
}

details::Access DevicePacketLength::access() {
    return details::Access::Private;
}

details::Category DevicePacketLength::category() {
    return details::Category::RunTime;
}

std::string DevicePacketLength::defaultValue() {
    return std::to_string(64 * 1024);
}

DevicePacketLength::value_type DevicePacketLength::parse(const std::string& value) {
    int intValue;
    try {
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue > 0,
        R"(unexpected {} option value "{}", only positive numbers are supported)", key(), value);
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue % 1024 == 0,
        R"(unexpected {} option value "{}", value should be multiple of 1024)", key(), value);
    return DevicePacketLength::value_type(intValue);
}

}  // namespace vpu
