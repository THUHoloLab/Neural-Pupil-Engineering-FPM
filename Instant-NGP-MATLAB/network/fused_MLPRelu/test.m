
clc
clear

A = single(gpuArray(randn(64,2,'single')));
B = single(gpuArray(randn(2,4096*4096,'single')));
C = single(gpuArray(randn(1,6,'single')));

C0 = gpuArray(single(zeros(64,1)));
% 
% tic
% for con = 1:30
% Y1 = fullyfused_MLPRelu(A,B,C);
% wait(gpuDevice());
% end
% toc

tic
for con = 1:300
Y2 = A*B;
wait(gpuDevice());
end
toc

% tic
% for con = 1:30
% Z = fullyconnect(dlarray(B,'CB'),A,C0);
% wait(gpuDevice());
% end
% toc
% % mean(abs(Y1 - Y2),'all')