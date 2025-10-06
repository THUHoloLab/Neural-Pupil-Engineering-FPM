#include "srcs/callfun.cuh"
#include <unordered_map>
#include <string>
#include <functional>

using MexSubFunc = void(*)(int, mxArray**, int, const mxArray**);

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const * prhs[]
){
    mxInitGPU();

    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("Mex:InvalidInput", "First argument must be a command string.");
    }

    const char* command = mxArrayToString(prhs[0]);
    std::string cmd_str(command);
    mxFree((void*)command);

    static const std::unordered_map<std::string, MexSubFunc> command_map = {
        {"forward", Call_Fwd},
        {"backward", Call_Bwd}
    };

    auto it = command_map.find(cmd_str);
    if (it != command_map.end()) {
        it->second(nlhs, plhs, nrhs - 1, prhs + 1);
    } else {
        mexErrMsgIdAndTxt("Mex:UnknownCommand", ("Unknown command: " + cmd_str).c_str());
    }
}
