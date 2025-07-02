clc
clear

disp("Building MATLAB extension for Hash Encoding");

forced_compiling_all = true;

tic

nvcc_flags = [...
 %   '-std=c++17 ',...
    '-allow-unsupported-compiler ' ...
];

setenv("NVCC_APPEND_FLAGS", nvcc_flags)

include_dirs = {...
    '', ...
    'addon'};

flags = cellfun(@(dir) ['-I"' fullfile(pwd, dir) '"'], ...
                        include_dirs, 'UniformOutput', false); 
% use the cuda.lib
flags = [flags,{'-lcuda'},{'-lcudart'}]; 


cu_path  = 'srcs/';

cu_sources = {...
    'kernels.cu',...
	'callfun.cu'};


main_file = 'fullyfused_hashEncoder.cu';

obj_path = fullfile(pwd, 'mex_obj/');
if ~exist(obj_path, 'dir')
    mkdir(obj_path);
end

% compiling for cuda file
cu_objs = cellfun(@(f) [obj_path replace(f,{'.cu'},{'.obj'})], ...
                        cu_sources, 'UniformOutput',false);
for i = 1:length(cu_sources)
    if ~exist(cu_objs{i},'file') || forced_compiling_all
        mexcuda(flags{:}, '-c', [cu_path,cu_sources{i}], ...
                          '-outdir', obj_path);
    else
        disp([cu_objs{i}, ' already exist, skip its compiling.']);
    end
end

[output_path, ~, ~] = fileparts(pwd);

mexcuda(flags{:}, main_file, cu_objs{:},'-outdir',[output_path,'\private']);

time_spend = toc;
disp(['compiling takes:',num2str(time_spend),'s'])