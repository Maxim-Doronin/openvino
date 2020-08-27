
#pragma once
#include "ngraph/pass/visualize_tree.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "net_pass.h"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "../common/vpu_tools_common.hpp"

#include <ngraph/opsets/opset3.hpp>
#include <inference_engine.hpp>
#include <generic_ie.hpp>
#include <common.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include <iostream>
#include <fstream>
#include <set>
#include <vector>


#include <string>
#include <cstdlib>
#include <typeindex>
#include <map>

class SubgraphCutter {
public:
    SubgraphCutter(const std::string& modelPath,const std::string& binPath,const std::set<std::string>& names, const std::string subName);
    void CutSubgraph(const std::set<std::string>& names, const std::string subName);
   std::vector<std::vector<std::uint8_t>> getInputs(const std::string& imgPath, const std::string saveName, std::map<std::string, std::string>& inputFileNames);
   std::vector<std::vector<std::uint8_t>> getInputs(const std::string& imgPath);
    std::map<std::string, InferenceEngine::Blob::Ptr> calculateSubgraph(std::map<std::string, std::string>& inputPaths);
    std::vector<std::vector<std::uint8_t>> calculateSubgraphFunc(const std::vector<std::vector<uint8_t>>& inputs);
    std::map<std::string, short> compareResults(std::map<std::string, std::string>& inputPaths, const std::vector<std::vector<uint8_t>>& inputs);
    // void setDeviceName(std::string devName);
    void setConfig(const std::map<std::string, std::string>& config);
    InferenceEngine::StatusCode save(const std::string& xmlPath,const std::string& binPath, InferenceEngine::ResponseDesc* resp);
    

protected:
    InferenceEngine::CNNNetwork _originNet;
    std::shared_ptr<ngraph::Function> _subgraphFunc;
    std::shared_ptr<ngraph::Function> _inputGraph;
    ngraph::ResultVector _subgraphResults;
    ngraph::ResultVector _beforeResults;
    std::map<std::string, std::shared_ptr<ngraph::Node>> _subgraphNodes;
    std::map<std::string, std::shared_ptr<ngraph::Node>> _beforeNodes;
    ngraph::ParameterVector _beforeParameters;
    ngraph::ParameterVector _subgraphParameters;
    std::string _deviceName = "MYRIAD";
    std::map<std::string, std::string> _config;

};