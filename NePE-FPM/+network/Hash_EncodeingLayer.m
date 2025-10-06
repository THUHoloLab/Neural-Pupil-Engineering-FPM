classdef Hash_EncodeingLayer < nnet.layer.Layer & ...
                               nnet.layer.Formattable
    %{
        Hash encoding layer for neural graphics - implements multi-resolution 
        hash tables for efficient coordinate encoding as described in InstantNGP
        Authors: Shuhe Zhang, Liangcai Cao
        {shuhe-zhang,clc}@tsinghua.edu.cn
    %}
    properties 
        device;              % Computation device (CPU/GPU)
        level;               % Number of resolution levels
        base_res;            % Base resolution for coarsest level
        high_res;            % Highest resolution for finest level
        log2_hashmap_size;   % Log2 of hash table size per level
        hash_map_sizes;      % Actual hash table sizes for each level
    end

    properties(Access = private)
        log_scale;           % Logarithmic scale factor between levels
        xxfunc;              % CUDA kernel for fused hash encoding
        feature_len;         % Feature dimension per entry
        total_param_size;    % Total parameters across all hash tables
        offsets;             % Parameter offsets for each level
        bounding_box;        % Spatial bounds for coordinate normalization
    end

    properties (Learnable)
        embedding;           % Learnable hash table parameters
    end

    methods 
        function self = Hash_EncodeingLayer(args)
            % Constructor for hash encoding layer
            % Inputs:
            %   device - computation device
            %   bounding_box - spatial bounds [min; max]
            %   level - number of resolution levels
            %   base_res, high_res - resolution range
            %   feature_len - output feature dimension
            %   log2_hashmap_size - hash table size exponent
            %   name - layer name
            
            arguments
                args.device;
                args.bounding_box = [0,0;1,1];
                args.level = 8;
                args.base_res = 16;
                args.high_res = 4096;
                args.feature_len = 2;
                args.log2_hashmap_size = 21;
                args.name = "Hash-Encoder"
            end

            self.Name = args.name;
            self.device             = args.device;
            self.bounding_box       = gpuArray(single(args.bounding_box));
            self.level              = args.level;
            self.base_res           = args.base_res;
            self.high_res           = args.high_res;
            self.feature_len        = args.feature_len;
            self.log2_hashmap_size  = args.log2_hashmap_size;

            % Calculate logarithmic scale between resolution levels
            self.log_scale = log2(self.high_res/self.base_res) ...
                                 / (self.level - 1);

            self.NumInputs = 1;
            self.NumOutputs = 1;  

            self.hash_map_sizes = zeros(self.level,1);
            self.offsets = zeros(self.level,1);
            
            % Initialize CUDA kernel for efficient hash encoding
            self.xxfunc = network.fused_HashEncoding2D();
        end

        function self = initialize(self,layout)
            % Initialize learnable hash table parameters
            % Sets up multi-resolution hash tables with geometric progression
            
            if isempty(self.embedding) 
                
                offset = 0;
                for levels = 1:self.level
                    % Calculate resolution for current level
                    resolution = self.base_res * 2^((levels-1) * ...
                                                     self.log_scale) - 1;
                    resolution = ceil(resolution) + 1;
                    
                    % Align to 32-element boundaries for GPU efficiency
                    full_size_aligned = ceil(resolution^2 / 32) * 32;
                    
                    % Determine hash table size (capped by log2_hashmap_size)
                    params_size_level = min(2^self.log2_hashmap_size, ...
                                              full_size_aligned);
                    self.hash_map_sizes(levels) = params_size_level;
                    self.offsets(levels) = offset;
                    offset = offset + params_size_level;
                end

                self.total_param_size = offset * self.feature_len;

                % Initialize hash table with random small values
                self.embedding = 0.002 * gpuArray.rand(self.feature_len, ...
                                                       offset, ...
                                                       'single') - 0.001;

                % Move arrays to GPU
                self.hash_map_sizes = gpuArray(single(self.hash_map_sizes));
                self.offsets = gpuArray(single(self.offsets));
            end
        end

        function Y = predict(self,X) 
            % Forward pass - encode input coordinates using hash tables
            % Input: X - coordinate tensor [batch_size, coord_dim]
            % Output: Y - encoded features [batch_size, total_feature_dim]
            
            X_dim = dims(X);
            
            % Ensure data is on GPU
            if ~isgpuarray(X)
                X = gpuArray(X);
            end

            if ~isgpuarray(self.embedding)
                self.embedding = gpuArray(self.embedding);
            end

            % Call fused CUDA kernel for efficient hash encoding
            Y = self.xxfunc( ...
                X,...
                self.embedding,...
                ... % input parameters for CUDA kernel
                self.bounding_box,...
                self.hash_map_sizes,...
                self.offsets,...
                self.level,...
                self.log_scale,...
                self.base_res,...
                self.feature_len...
            );

            Y = dlarray(Y, X_dim);  % Preserve input dimensions
        end
    end
end