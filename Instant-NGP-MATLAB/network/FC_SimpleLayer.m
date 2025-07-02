classdef FC_SimpleLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        outputSize;
        
    end

    properties(Learnable)
        weights;
        bias; 
    end

    methods
        function layer = FC_SimpleLayer(outputSize,NameValueArgs)
            arguments
                outputSize
                NameValueArgs.Name = '';
            end
            layer.Name = "Linear-" + NameValueArgs.Name;
            layer.Type = 'self-defined FC';
            layer.outputSize = outputSize;
        end

        function layer = initialize(layer,layout) 
            % Initialize fully connect weights.
            if isempty(layer.weights)
                % Find number of channels.
                idx = finddim(layout,"C");
                numChannels = layout.Size(idx);
    
                % Initialize using Glorot.
                sz = [prod(layer.outputSize) numChannels];
                numOut = prod(layer.outputSize);
                numIn = numChannels;
                layer.weights = initializeGlorot(sz,numOut,numIn);
            end

            % Initialize fully connect bias.
            if isempty(layer.bias)
                % Initialize with zeros.
                layer.bias = initializeZeros([prod(layer.outputSize) 1]);
            end
        end

        function Z = predict(layer, X)
            Z = fullyconnect(X,layer.weights,layer.bias);
            % X = stripdims(X);
            % Z = layer.weights * X;
            % Z = dlarray(Z,"CB");
        end
    end
end


function weights = initializeGlorot(sz,numOut,numIn,className)
arguments
    sz
    numOut
    numIn
    className = 'single'
end
Z = 2*rand(sz,className) - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);
end

function parameter = initializeZeros(sz,className)
arguments
    sz
    className = 'single'
end
parameter = zeros(sz,className);
parameter = dlarray(parameter);
end




