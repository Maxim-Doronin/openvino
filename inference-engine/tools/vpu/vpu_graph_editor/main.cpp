// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "SubgraphCutter.hpp"
//#include "SubgraphCutter.cpp"



//struct for file paths used to get needed data
struct FilePath {

std::string IRpath;
std::string namePath;
std::string weightsPath = "";
std::string inputPath;
std::string forVisualPath;
std::string inputFileName = "Result";
std::string deviceName =  "MYRIAD";
std::string subgraphName = "subgraph";
std::string configPath;
std::string savePath = "";
std::string helpMessage = 
"Help message: \n\tUsage: vpu_graph_editor <model IR path> [<weights path>] <txt file with nodes name path> [<switch> ...]\n\timage MUST be bmp 24bpp\n\tswitch\t\t\t short\t meaning\n\t-input <file path>\t-i\tpath to inout file\n\t-file <name>\t\t-f\tsave files first name(ex: name for inp1 is first name + inp1(firstNameinp1.bin))\n\t-calculate\t\t-c\tadd to calculate subgraph and compare results\n\t-name <name>\t\t-n\tsubgraph name\n\t-config <filePath>\t-con\tpath to file with config values\nin a subgraph names file each name should be on a different line\n";
bool calcNet = false;
bool save = false;
//void defult{};

//gets values from command line 
FilePath(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Missing required command line arguments" << std::endl;
        std::cout << helpMessage;
        std::exit(EXIT_FAILURE);
    }
    if (argc == 3) {
       IRpath = argv[1];
       namePath = argv[2];
        for(const auto& latter : IRpath){
                if(latter == '.') {
                    break;
                }
                weightsPath += latter;
            }
        weightsPath+= ".bin";
    }
    else {
        int begin = 0;
        if(argv[3][0] !='-') {
        IRpath = argv[1];
        namePath = argv[3];
        weightsPath = argv[2];
        begin = 4;
        }
        else {
            IRpath = argv[1];
            namePath = argv[2];
            for(const auto& latter : IRpath){
                if(latter == '.') {
                    break;
                }
                weightsPath += latter;
            }
            weightsPath+= ".bin";
            begin = 3;
        }
        for(int i = begin; i < argc; ++i) {
            std::string operation = argv[i];
            if (operation == "-i" || operation == "-input") {
                inputPath = argv[++i];
                continue;
            }
            if (operation == "-f" || operation == "-file") {
                inputFileName = argv[++i];
                continue;
            }
            if (operation == "-c" || operation == "-calculate") {
                calcNet = true;
                continue;
            }
            if (operation == "-n" || operation == "-name") {
                subgraphName = argv[++i];
                continue;
            } 
            if (operation == "-h"|| operation  == "-help") {
                std::cout << helpMessage;
                std::exit(0);
            }
            if (operation == "-con" || operation == "-config") {
                configPath = argv[++i];
                continue;
            }
            if (operation == "-s" || operation == "-save") {
                save = true;
                if((i+1 < argc)&&(argv[i+1][0]!= '-')) {
                    savePath = argv[++i];
                }
                continue;
            }
            std::cout<<"Unknown operaion " << argv[i]<<std::endl;
            std::cout << helpMessage;
            std::exit(EXIT_FAILURE);
        }
    }
}

};


//reads names from file
void readNames(std::set<std::string>& NamesCont, const std::string& filePath) {
    std::fstream f;
    f.open(filePath, std::fstream::in);
    if(!f.is_open()) {
        std::cout << "\nfiled to open file\n";
        std::exit(EXIT_FAILURE);
    }
    std::string s; 
    while (std::getline(f, s)) {
       if(s.size() > 0){
           NamesCont.insert(std::move(s));
       }
    }
    f.close();
}

std::map<std::string, std::string> readConfig(const std::string& path) {
    std::fstream f;
    f.open(path, std::fstream::in);
    if(!f.is_open()) {
        std::cout << "\nfiled to open file\n";
        std::exit(EXIT_FAILURE);
    }
    std::string s; 
    std::map<std::string, std::string> ans;
    while (f >> s) {
        std::string value;
        f >> value;
        if(value == "CONFIG_VALUE(YES)") {
            value = CONFIG_VALUE(YES);
        }
        if(value == "CONFIG_VALUE(NO)") {
            value = CONFIG_VALUE(NO);
        }
        ans[s] = value;
    }
    return ans;
}

int main(int argc, char* argv[]) {
    try {
        FilePath file(argc, argv);
        std::set<std::string> names; //set of subgraphs nodes names
        readNames(names, file.namePath);
        SubgraphCutter subCut(file.IRpath, file.weightsPath, names, file.subgraphName);
        std::cout<< "Subgraph created\n";
        if(file.inputPath.size()) {
            std::map<std::string, std::string> saveFileNames;
            const auto& inputs = subCut.getInputs(file.inputPath, file.inputFileName, saveFileNames);
            std::cout << "input file names: \n";
            for(auto& save : saveFileNames) {
                std::cout<<"\tfor "<<save.first<<"  "<<save.second<<"\n";
            }
            if(file.calcNet) {
                if(file.configPath.size()) {
                    subCut.setConfig(readConfig(file.configPath));
                }
                const auto& comparationResults = subCut.compareResults(saveFileNames, inputs); 
                std::cout<<"Comparing results:\n";
                for(auto& res : comparationResults) {
                    std::cout<<"\t"<<res.first<<" max diff: "<<res.second<<"\n";
                }
            }
        }
        if(file.save) {
            std::string xmlPath = file.savePath + file.subgraphName + ".xml";
            std::string binPath = file.savePath + file.subgraphName + ".bin";
            InferenceEngine::ResponseDesc* resp = new InferenceEngine::ResponseDesc();
            if(subCut.save(xmlPath, binPath, resp) != InferenceEngine::OK) {
                std::cout << "error while saving\n";
                std::cerr << resp->msg<<"\n";
                return 1;
            }
        }
        return 0;
    } catch(std::exception& e) {
        std::cout << e.what() << "\n";
        return 1;
    }
   
}
