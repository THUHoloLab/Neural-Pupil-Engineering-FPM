function net = NGP(args)
    % NGP - Neural Graphics Primitives network based on InstantNGP
    % 
    % Constructs a multi-layer perceptron with hash encoding for 
    % efficient neural representation learning
    %
    % Inputs (name-value pairs):
    %   output_dims - output dimension (default: 2)
    %   layers - number of fully-connected layers after encoding (default: 2)  
    %   hash_level - number of multi-resolution hash levels (default: 8)
    %   base_res - base resolution for coarsest hash level (default: 16)
    %   high_res - highest resolution for finest hash level (default: 4096)
    %   bounding_box - spatial bounds for coordinate normalization [min; max]
    %   feature_dim - feature dimension per hash table entry (default: 2)
    %   log2_hashmap_size - log2 of hash table size per level (default: 19)
    %
    % Output:
    %   net - dlnetwork ready for training/inference

    arguments
        args.output_dims = 2;
        args.layers = 2;
        args.hash_level = 8;
        args.base_res = 16;
        args.high_res = 4096;
        args.bounding_box
        args.feature_dim = 2;
        args.log2_hashmap_size = 19;
    end

    % Build network architecture:
    % 1. Input layer for 2D coordinates
    layers = [
        featureInputLayer(2,"Name","xyz-input");
        
        % 2. Multi-resolution hash encoding layer (InstantNGP)
        network.Hash_EncodeingLayer( ...
            "base_res",     args.base_res,...
            "high_res",     args.high_res,...
            "device",       gpuDevice(),...
            "bounding_box", args.bounding_box,...
            "level",        args.hash_level,...
            "feature_len",  args.feature_dim,...
            "log2_hashmap_size", args.log2_hashmap_size);
    ];

    % 3. Add fully-connected layers with ReLU activations
    for con = 1:args.layers
        layers = [layers;
            network.FC_SimpleLayer(64,"Name","fc" + con);
            reluLayer();
        ];
    end

    % 4. Final output layer (no activation)
    layers = [layers;
        network.FC_SimpleLayer(args.output_dims,"Name",'fin');
    ];

    % Create deep learning network
    net = dlnetwork(layers);
end