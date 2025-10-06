classdef autoDiff_cuFPM < deep.DifferentiableFunction
    %{
        Automatic differentiation module for Fourier ptychographic microscopy (FPM)
        Supports both CPU and CUDA implementations for forward/backward propagation
        Authors: Shuhe Zhang, Liangcai Cao
        {shuhe-zhang, clc}@tsinghua.edu.cn
    %}
    properties
        if_use_CUDA = false;  % Flag to enable CUDA acceleration
    end

    methods
        function self = autoDiff_cuFPM(use_CUDA)
            % Constructor for FPM auto-differentiation module
            % Input: use_CUDA - boolean flag to enable GPU acceleration
            
            output_num = 1; % important!!!

            self@deep.DifferentiableFunction(...
                        output_num,...
                        SaveInputsForBackward = true,...
                        SaveOutputsForBackward = false,...
                        NumMemoryValues = 2);

            self.if_use_CUDA = use_CUDA;
        end
        
        % forward function
        function [pred_Y,memo1,memo2] = forward(self,...
                                           wavefront1, ...
                                           wavefront2, ...
                                           dY_obs, ...
                                           ledIdx, ...
                                           pratio)

            % Forward pass for FPM reconstruction
            % Inputs:
            %   wavefront1, wavefront2 - complex wavefronts
            %   dY_obs - observed diffraction patterns  
            %   ledIdx - LED array indices for illumination
            %   pratio - pixel ratio parameter
            %
            % Outputs:
            %   pred_Y - predicted intensity measurement
            %   memo1, memo2 - cached values for backward pass
            
            if ~self.if_use_CUDA
                % CPU/gpuArray implementation: FFT-based propagation
                wavefront1 = fftshift(fft2(wavefront1));
                memo2 = dY_obs * 0;
                for idx = 1:size(ledIdx,2)
                    kt = ledIdx(2,idx);
                    kb = ledIdx(2,idx) + size(dY_obs,1) - 1;
                    kl = ledIdx(1,idx);
                    kr = ledIdx(1,idx) + size(dY_obs,1) - 1;
                    memo2(:,:,idx) = wavefront1(kt:kb,kl:kr);
                end
                memo1 = ifft2(ifftshift(ifftshift(memo2 .* ...
                                              wavefront2,1),2))./ pratio^2;
            else
                % CUDA implementation: fused kernel operation
                [memo1,memo2] = fullyfusedFPM_Fwd( ...
                    wavefront1,...
                    wavefront2,...
                    dY_obs,...
                    int32(ledIdx),...
                    pratio ...
                );
                memo1 = memo1 ./ pratio^2;
            end
            pred_Y = abs(memo1);  % Final output: intensity measurement
        end

        % backward function
        function [dldw1,...
                  dldw2,...
                  dldY,...
                  dldled,...
                  dldP] = backward(self,dl_dout,...
                                     ~,...
                                     wavefront1, ...
                                     wavefront2, ...
                                     dY_obs, ...
                                     ledIdx, ...
                                     pratio,...
                                     memo1,memo2)

            % Backward pass for gradient computation
            % Inputs:
            %   dl_dout - gradient from downstream layer
            %   wavefront1, wavefront2, dY_obs, ledIdx, pratio - forward inputs
            %   memo1, memo2 - cached values from forward pass
            %
            % Outputs:
            %   dldw1, dldw2 - gradients for wavefront parameters
            %   dldY, dldled, dldP - gradients for other parameters (empty here)

            dldY = [];
            dldled = [];
            dldP = [];
            
            if ~self.if_use_CUDA
                % CPU/gpuArray implementation: gradient computation via FFT
                dldw1 = wavefront1 .* 0;
                memo1 = dl_dout .* sign(memo1);
                memo1_record = fftshift(fftshift(fft2(memo1),1),2);
                memo1 = memo1_record .* conj(wavefront2);
                for idx = 1:size(ledIdx,2)
                    kt = ledIdx(2,idx);
                    kb = ledIdx(2,idx) + size(dY_obs,1) - 1;
                    kl = ledIdx(1,idx);
                    kr = ledIdx(1,idx) + size(dY_obs,1) - 1;

                    dldw1(kt:kb,kl:kr) = ...
                                       dldw1(kt:kb,kl:kr) + memo1(:,:,idx);
                end
                dldw1 = (ifft2(ifftshift(dldw1)));
                dldw2 = (sum(memo1_record .* conj(memo2),3));
            else
                % CUDA implementation: fused backward kernel, requires
                % gpu device
                [dldw1,dldw2] = fullyfusedFPM_Bwd( ...
                                                    dl_dout,...
                                                    wavefront1,...
                                                    wavefront2,...
                                                    memo1,...
                                                    memo2,...
                                                    int32(ledIdx) ...
                                                 );
            end

            dldw1 = conj(dldw1);  % Return complex conjugate gradients
            dldw2 = conj(dldw2);
        end
    end
end