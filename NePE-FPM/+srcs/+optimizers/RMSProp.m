classdef RMSProp < handle
    properties
        mom2
        beta1
        epsilon
        iteration
    end
    
    methods
        function self = RMSProp(beta1,epsilon)

            arguments
                beta1(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta1, 0), mustBeLessThan(beta1, 1)} = 0.9;
                epsilon(1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-8;
            end
            self.mom2       = [];
            self.beta1      = beta1;
            self.epsilon    = epsilon;
        end

        function para = step(self,para,grad,iteration,lr)
            [para, self.mom2] = update_core(para, grad, ...
                                                  self.mom2, ...
                                                  iteration, ...
                                                  lr, ...
                                                  self.beta1, ...
                                                  self.epsilon);

        end
    end
end




function [p, avg_gsq] = update_core(p, g, avg_gsq, t, lr, beta1, epsilon)
%update_core Update parameters via adaptive moment estimation


arguments
    p
    g
    avg_gsq
    t(1,1) {mustBeNumeric, mustBePositive, mustBeInteger}
    
    lr(1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 0.001;
    beta1(1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta1, 0), mustBeLessThan(beta1, 1)} = 0.999;
    epsilon(1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-8;
end

persistent func
if isempty(func)
    func = deep.internal.LearnableUpdateFunction( ...
        @iSingleStepValue, ...
        @iSingleStepParameter);
end

if isempty(avg_gsq)
    % Execute a first-step update with g_av and g_sq_av set to 0.  The step
    % will create arrays for these that are the correct size
    paramArgs = {g};
    fixedArgs = {0, t, lr, beta1, epsilon};
else
    % Execute the normal update
    paramArgs = {g, matlab.lang.internal.move(avg_gsq)};
    fixedArgs = {t, lr, beta1, epsilon};
end

[p, avg_gsq] = deep.internal.networkContainerFixedArgsFun(func, ...
    p, matlab.lang.internal.move(paramArgs), fixedArgs);
end


function [p, avg_gsq] = iSingleStepParameter(p, g, avg_gsq, t, lr, beta1, epsilon)
% Apply per-parameter learn-rate factor
lr = lr .* p.LearnRateFactor;

% Short-circuit for 0 learn rate.
if lr == 0
    return;
end

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate

[step, avg_gsq] = rmsprop_step(...
    g, avg_gsq, lr, beta1, epsilon);

v = p.Value;
p.Value = [];
v = v + step;
p.Value = v;
end


function [p, avg_gsq] = iSingleStepValue(p, g, avg_gsq, t, lr, beta1, epsilon)
% Short-circuit for 0 learn rate.
if lr == 0
    return;
end

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the learning rate
[step, avg_gsq] = rmsprop_step(...
    g, avg_gsq, lr, beta1, epsilon);
p = p + step;

end

function [step, avg_gsq] = rmsprop_step(g, avg_gsq, learnrate, beta1, epsilon)
avg_gsq = beta1 .* avg_gsq + (1 - beta1) .* (abs(g).^2);
step = -learnrate.*( g ./ (sqrt(avg_gsq) + epsilon) );
end
