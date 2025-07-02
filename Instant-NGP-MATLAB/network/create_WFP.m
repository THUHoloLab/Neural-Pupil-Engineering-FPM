function net = create_WFP(args)
    arguments
        args.output_dims = 1;
        args.layers = 2;
        args.hash_level = 8;
        args.base_res = 16;
        args.high_res = 4096;
        args.bounding_box
    end

    layers = [
        featureInputLayer(2,"Name","xyz-input");
        Hash_EncodeingLayer( ...
            "base_res",     args.base_res,...
            "high_res",     args.high_res,...
            "device",       gpuDevice(),...
            "bounding_box", args.bounding_box,...
            "level",        args.hash_level);
    ];

    for con = 1:args.layers
        layers = [layers;
            FC_SimpleLayer(64,"Name","fc" + con);
            reluLayer();
        ];
    end

    layers = [layers;
        FC_SimpleLayer(args.output_dims,"Name",'fin');
    ];

    net = dlnetwork(layers);
end