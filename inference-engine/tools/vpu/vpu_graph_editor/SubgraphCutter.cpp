#include "SubgraphCutter.hpp"




std::shared_ptr<ngraph::Node> make_before_op(std::map<std::string, std::shared_ptr<ngraph::Node>>& before,const std::shared_ptr<ngraph::Node>& op, ngraph::ParameterVector& parameters) {
    if (strcmp(op->get_type_info().name, "Parameter") == 0) {
        const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(op->get_element_type(), op->get_shape());
        stubParameter->set_friendly_name(op->get_friendly_name());
        parameters.push_back(stubParameter);
        return stubParameter;
    }
    ngraph::OutputVector args;
    for (auto& input : op->input_values()) {
        const auto& el = input.get_node_shared_ptr();
        const auto& name = el->get_friendly_name();
        auto befNode = before.find(name);
        if (befNode != before.end()) {
            args.push_back(befNode->second);
            continue;
        }
        auto existingParameter = std::find_if(parameters.begin(), parameters.end(), [&name](std::shared_ptr<ngraph::opset3::Parameter>& param){ return param->get_friendly_name() == name;});
        if(existingParameter != parameters.end()) {
            args.push_back(*existingParameter);
            continue;

        }
        if (strcmp(el->get_type_info().name, "Constant") == 0) {
            const auto& constant= dynamic_cast<ngraph::opset3::Constant*>(el.get());
            const auto& subConstant = std::make_shared<ngraph::opset3::Constant>(constant->get_element_type(), constant->get_shape(), constant->get_value_strings());
            subConstant->set_friendly_name(el->get_friendly_name());
            before[el->get_friendly_name()] = subConstant;
            args.push_back(subConstant);
            continue;
        }
        if (strcmp(op->get_type_info().name, "Parameter") == 0) {
            const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(input.get_element_type(), input.get_partial_shape());
            stubParameter->set_friendly_name(el->get_friendly_name());
            parameters.push_back(stubParameter);
            args.push_back(stubParameter);
            continue;
        }
        args.push_back(make_before_op(before, el, parameters));
    }
    auto newOp = op->clone_with_new_inputs(args);
    newOp->set_friendly_name(op->get_friendly_name());
    before[op->get_friendly_name()] = newOp;
    return newOp;
}

std::shared_ptr<ngraph::Node> createOp(std::map<std::string, std::shared_ptr<ngraph::Node>>& nodes,const std::shared_ptr<ngraph::Node>& op, 
                                        ngraph::ParameterVector& parameters,ngraph::ResultVector& results, std::map<std::string, std::shared_ptr<ngraph::Node>>& before, 
                                        ngraph::ParameterVector& beforeParameters, ngraph::ResultVector& beforeResults, bool addNodeToBefore = false) {
        std::shared_ptr<ngraph::Node> newOp(nullptr);
            if (strcmp(op->get_type_info().name, "Parameter") != 0) {
                ngraph::OutputVector args;
                for (auto& element : op->input_values()) {
                    const auto& el = element.get_node_shared_ptr(); 
                    const auto& name = el->get_friendly_name();
                    auto subNode = nodes.find(name);
                    if(subNode != nodes.end()) {
                        args.push_back(subNode->second);
                        continue;
                    }
                    const auto& existingParameter = std::find_if(parameters.begin(), parameters.end(), [&name](std::shared_ptr<ngraph::opset3::Parameter>& param){ return param->get_friendly_name() == name;});
                    if(existingParameter != parameters.end()) {
                        args.push_back(*existingParameter);
                        continue;
                    }
                    if (strcmp(el->get_type_info().name, "Constant") == 0) {
                        const auto& constant= dynamic_cast<ngraph::opset3::Constant*>(el.get());
                        const auto& subConstant = std::make_shared<ngraph::opset3::Constant>(constant->get_element_type(), constant->get_shape(), constant->get_value_strings());
                        subConstant->set_friendly_name(el->get_friendly_name());
                        nodes[el->get_friendly_name()] = subConstant;
                        args.push_back(subConstant);
                        continue;
                    }
                    const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(element.get_element_type(), element.get_partial_shape());
                    stubParameter->set_friendly_name(name);
                    parameters.push_back(stubParameter);
                    args.push_back(stubParameter);
                    if (addNodeToBefore && (strcmp(el->get_type_info().name, "Parameter") != 0)) {
                        auto before_node = make_before_op(before, el, beforeParameters);
                        beforeResults.push_back(std::make_shared<ngraph::opset3::Result>(before_node));
                    }
                }
                if (strcmp(op->get_type_info().name, "Result") == 0) {
                    const auto& newRes = std::make_shared<ngraph::opset3::Result>(args[0].get_node_shared_ptr());
                    newRes->set_friendly_name(op->get_friendly_name());
                    results.push_back(newRes);
                } else {
                    newOp = op->clone_with_new_inputs(args);
                    newOp->set_friendly_name(op->get_friendly_name());
                    nodes[op->get_friendly_name()] = newOp;
                }
            }
            else {
                const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(op->get_element_type(), op->get_shape());
                stubParameter->set_friendly_name(op->get_friendly_name());
                newOp = stubParameter;
                parameters.push_back(stubParameter);
            }
    return newOp;
}



std::vector<std::uint8_t> loadImage(const std::string &imageFilename,std::shared_ptr<ngraph::opset3::Parameter>& param) {
    //auto& tens = param->get_output_tensor(0);
    //auto layout = tens.get_tensor_layout();

    BitMap reader(imageFilename);

    const auto dims = param->get_output_shape(0);

    const size_t N = dims[0];
    const size_t C = dims[1];
    const size_t H = dims[2];
    const size_t W = dims[3];

    const size_t img_w = reader.width();
    const size_t img_h = reader.height();

    const size_t numImageChannels = reader.size() / (reader.width() * reader.height());
    if (C != numImageChannels && C != 1) {
        std::cout << "loadImage error: Input channels mismatch: image channels " << numImageChannels << ", "
                  << "network channels " << C << ", expecting count of image channels are equal "
                  << "to count if network channels or count of network channels are equal to 1" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::uint8_t> info; 
    const unsigned char* RGB8 = reader.getData().get();
    const float xScale = 1.0f * img_w / W;
    const float yScale = 1.0f * img_h / H;

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; ++h) {
            int y = static_cast<int>(std::floor((h + 0.5f) * yScale));
            for (int w = 0; w < W; ++w) {
                int x = static_cast<int>(std::floor((w + 0.5f) * xScale));
                for (int c = 0; c < C; c++) {
                    float tmp = 0.f;
                    tmp = 1.0 * RGB8[(y * img_w + x) * numImageChannels + c];
                    auto tmpVec = reinterpret_cast<std::uint8_t*>(&tmp);
                    for (int i = 0; i < sizeof(float); ++i ) {
                        info.push_back(tmpVec[i]);
                    }
                }
            }
        }
    }

    return info;
}

void visit (const std::shared_ptr<ngraph::Node>& node, std::map<std::string, bool>& map) {
    map[node->get_friendly_name()] = true;
    for (auto& inp : node->input_values()) {
        const auto& input = inp.get_node_shared_ptr();
        if(map[input->get_friendly_name()] == false) {
            visit(input, map);
        }
    }
    if (strcmp(node->get_type_info().name, "Result") != 0) {
        for (size_t i = 0; i < node->get_output_size();++i) {
            const auto& set = node->get_output_target_inputs(i); 
            for (auto& sel : set) {
                auto el = sel.get_node();
                if (map[el->get_friendly_name()] == false) {
                    const auto& elPtr = el->output(0).get_node_shared_ptr();
                    ///auto ptr = std::make_shared<ngraph::Node>(*el);
                    visit(elPtr, map);
                }
            }
        }
    }
}

std::string connectionCheck (const std::shared_ptr<ngraph::Function>& func,const  std::set<std::string>& originalNames) {
    const auto& ops = func->get_ordered_ops();
    std::map<std::string, bool> visited;
    for (auto op : ops) {
        visited[op->get_friendly_name()] = false;
    }
    visit(ops[0], visited);
    for (auto el : visited) {
        if (el.second == false) {
            if (originalNames.count(el.first)) {
                return el.first;
            }
        }
    }
    return "";
}




short compare(const std::vector<uint8_t>& ref,const InferenceEngine::Blob::Ptr& res) {
    auto refFloat = reinterpret_cast<const float*>(ref.data());
    short* blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<short>>(res)->data();
    short max = 0;
    for(auto i = 0; i < (ref.size()/sizeof(float)); ++i) {
       if (abs(blobDataPtr[i] - InferenceEngine::PrecisionUtils::f32tof16(refFloat[i])) > max) {
           max = abs(blobDataPtr[i] - InferenceEngine::PrecisionUtils::f32tof16(refFloat[i]));
       }
    }
    return max;
}


SubgraphCutter::SubgraphCutter(const std::string& modelPath,const std::string& binPath,const std::set<std::string>& names = std::set<std::string>(), const std::string subName = "subgraph")  {
     _originNet = InferenceEngine::Core().ReadNetwork(modelPath, binPath);
    //c_originNet = InferenceEngine::Core().ReadNetwork(modelPath, binPath);
    if (names.size()) {
        this->CutSubgraph(names, subName);
    }
}

void loadBinaryTensorf32(const std::string &binaryFileName, InferenceEngine::Blob::Ptr& blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();
    if (tensDesc.getPrecision() != InferenceEngine::Precision::FP16) {
        throw std::invalid_argument("Input must have FP16 precision");
    }

    std::ifstream binaryFile(binaryFileName, std::ios_base::binary | std::ios_base::ate);
    if (!binaryFile) {
        throw std::invalid_argument("Can not open \"" + binaryFileName + "\"");
    }

    auto fileSize = static_cast<std::size_t>(binaryFile.tellg());
    binaryFile.seekg(0, std::ios_base::beg);
    if (!binaryFile.good()) {
        throw std::invalid_argument("Can not read \"" + binaryFileName + "\"");
    }
    /* try to read 32 bits data */
    std::int16_t *blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<std::int16_t>>(blob)->data();
    for (std::size_t i = 0; i < blob->size(); i++) {
        float tmp = 0.f;
        binaryFile.read(reinterpret_cast<char *>(&tmp), sizeof(float));
        blobDataPtr[i] = InferenceEngine::PrecisionUtils::f32tof16(tmp);
    }
}

void SubgraphCutter::CutSubgraph(const std::set<std::string>& names, const std::string subName = "subgraph") {
    InferenceEngine::ICNNNetwork::Ptr ptr = static_cast<InferenceEngine::ICNNNetwork::Ptr>(_originNet);
    auto ngraphNetwork = dynamic_cast<InferenceEngine::details::CNNNetworkNGraphImpl*>(ptr.get()); 
    if (ngraphNetwork == nullptr) {
        throw std::invalid_argument("failed to read network\n");
    }
    const auto& nGraphFunc = ngraphNetwork->getFunction();
    const auto& ops = nGraphFunc->get_ordered_ops();
    std::set<std::string> out = names;
    for (auto& op : ops) {                                  
        if (out.erase(op->get_friendly_name()) > 0) {  //creates an operation that's clone of op if ops name was in out(copy of names)
            auto subOp = createOp(_subgraphNodes, op, _subgraphParameters, _subgraphResults, _beforeNodes, _beforeParameters, _beforeResults, true);
            for (size_t i = 0; i < op->get_output_size();++i) {
                const auto& set = op->get_output_target_inputs(i); 
                for (auto sel : set) {
                    auto el = sel.get_node();
                    if (out.find(el->get_friendly_name()) == out.end()) {
                        auto res = std::make_shared<ngraph::opset3::Result>(subOp);
                        _subgraphResults.push_back(res);
                    }
                }
            }
        }
        if (out.empty()) {
            break;
        }
    }
    if (!out.empty()) {
        std::string missing = "";
        for (auto& name : out) {
            missing += name + " ";
        }
        throw std::invalid_argument(missing +"wasn't fount\n");
    }
    _subgraphFunc =  std::make_shared<ngraph::Function>(_subgraphResults, _subgraphParameters, subName); 
    _inputGraph =  std::make_shared<ngraph::Function>(_beforeResults, _beforeParameters, "Inputgraph"); 
    std::string connection = connectionCheck(_subgraphFunc, names); //checks connection
    if (!(connection == "")) {
        throw std::invalid_argument("Graph isn't complete: " + connection + " dosen't connect to some nodes\n");
    }
}

size_t findIndex(const ngraph::ResultVector& results,const std::string& name) {
    size_t i = 0;
    for(auto res : results) {
        if(res->get_input_node_shared_ptr(0)->get_friendly_name() == name) {
            return i;
        }
        ++i;
    }
    return results.size();
}

std::vector<std::vector<std::uint8_t>> SubgraphCutter::getInputs(const std::string& imgPath) {
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(_inputGraph);
    _inputGraph->validate_nodes_and_infer_types();
    std::vector<std::vector<std::uint8_t>> beforeInputs;
    beforeInputs.reserve(_beforeParameters.size());
    for (auto& param : _beforeParameters) {
        beforeInputs.push_back(loadImage(imgPath, param));
    }
    std::vector<std::vector<std::uint8_t>> beforeOutputs;
    auto unBeforeOutputs = ngraph::helpers::interpreterFunction(_inputGraph, beforeInputs);
    size_t last = 0;
    for (auto param : _subgraphParameters) {
        const auto& name = param->get_friendly_name();
        auto i = findIndex(_beforeResults, param->get_friendly_name());
        if( i != _beforeResults.size()) {
            beforeOutputs.push_back(unBeforeOutputs[i]);
        }
        else {
           beforeOutputs.push_back(loadImage(imgPath, param));
        }
    }
    return beforeOutputs;
}

std::vector<std::vector<std::uint8_t>> SubgraphCutter::getInputs(const std::string& imgPath, const std::string saveName, std::map<std::string, std::string>& inputFileNames) {
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(_inputGraph);
    _inputGraph->validate_nodes_and_infer_types();
    std::vector<std::vector<std::uint8_t>> beforeInputs;
    beforeInputs.reserve(_beforeParameters.size());
    for (auto& param : _beforeParameters) {
        beforeInputs.push_back(loadImage(imgPath, param));
    }
    std::vector<std::vector<std::uint8_t>> beforeOutputs;
    auto unBeforeOutputs = ngraph::helpers::interpreterFunction(_inputGraph, beforeInputs);
    size_t last = 0;
    for (auto param : _subgraphParameters) {
        const auto& name = param->get_friendly_name();
        auto i = findIndex(_beforeResults, param->get_friendly_name());
        if( i != _beforeResults.size()) {
            beforeOutputs.push_back(unBeforeOutputs[i]);
        }
        else {
           beforeOutputs.push_back(loadImage(imgPath, param));
        }
        auto saveFileName = saveName;
        for (size_t i =0; i < name.size(); ++i) {
            if(name[i] == '/') {
                saveFileName += "-";
            }
            else {
                saveFileName += name[i];
            }
        }
        saveFileName += ".bin";
        inputFileNames[name] = saveFileName;
        std::ofstream f(saveFileName, std::fstream::out);
        if (!f.is_open()) {
            throw std::runtime_error("error while opening save file\n");
        }
        for (auto el : beforeOutputs[last]) {
            f << el;
        }
        ++last;
        f.close();
    }
    return beforeOutputs;
}


std::vector<std::vector<std::uint8_t>> SubgraphCutter::calculateSubgraphFunc(const std::vector<std::vector<uint8_t>>& inputs) {
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(_subgraphFunc);
    _subgraphFunc->validate_nodes_and_infer_types();
    return ngraph::helpers::interpreterFunction(_subgraphFunc, inputs);

}



void SubgraphCutter::setConfig(const std::map<std::string, std::string>& config) {
    _config = config;
}

std::map<std::string, InferenceEngine::Blob::Ptr> SubgraphCutter::calculateSubgraph(std::map<std::string, std::string>& inputPaths) {
    InferenceEngine::CNNNetwork subNetwork(_subgraphFunc);
    InferenceEngine::Core core;
    auto netInputs = subNetwork.getInputsInfo();
    for (auto &input : netInputs) {
        const auto& inputPrecision = input.second->getPrecision();
        if (inputPrecision == InferenceEngine::Precision::FP32 ||
                inputPrecision == InferenceEngine::Precision::U8) {
            input.second->setPrecision(InferenceEngine::Precision::FP16);
        }
    }
    auto netOutputs = subNetwork.getOutputsInfo();
    for (auto &output : netOutputs) {
    const auto outputPrecision = output.second->getPrecision();
    if (outputPrecision == InferenceEngine::Precision::FP32) {
            output.second->setPrecision(InferenceEngine::Precision::FP16);
        }
    }
    auto exeNet = core.LoadNetwork(subNetwork, _deviceName, _config);
    auto netRequest = exeNet.CreateInferRequest();
    for (auto& input : netInputs) {
        auto& name = input.first;
        auto inputBlob = netRequest.GetBlob(name);
        loadBinaryTensorf32(inputPaths[name], inputBlob) ;
    }
    netRequest.Infer();
    std::map<std::string, InferenceEngine::Blob::Ptr> results;
    for (auto& out : netOutputs) {
        results[out.first] = netRequest.GetBlob(out.first);
    }
    return results;
}

std::map<std::string, short> SubgraphCutter::compareResults(std::map<std::string, std::string>& inputPaths, const std::vector<std::vector<uint8_t>>& inputs) {
    auto results =  this->calculateSubgraph(inputPaths);
    const auto& ref = this->calculateSubgraphFunc(inputs);
    std::map<std::string, short> ans;
    for (size_t i =0; i < ref.size();++i) {
        const auto& name = _subgraphResults[i]->get_input_node_shared_ptr(0)->get_friendly_name();
        ans[name] = compare(ref[i], results[name]);
    }
    return ans;
}

InferenceEngine::StatusCode SubgraphCutter::save(const std::string& xmlPath,const std::string& binPath, InferenceEngine::ResponseDesc* resp) {
    InferenceEngine::details::CNNNetworkNGraphImpl subNetwork(_subgraphFunc);
    return subNetwork.serialize(xmlPath, binPath, resp);
}