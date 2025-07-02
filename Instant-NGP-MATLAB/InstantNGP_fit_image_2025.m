clc
clear
reset(gpuDevice());

foo = @(x) gpuArray(single(x));

img = imread("peppers.png");
img = foo(img)/255;
img = imresize(img,[2048,2048]);

target = dlarray(reshape(img,[],3),"BC");

x = linspace(0,1,size(img,2));
y = linspace(0,1,size(img,1));

[x,y] = meshgrid(x,y);

pos_batch = foo([x(:),y(:)]);

pos_batch = dlarray(pos_batch,"BC");

addpath(genpath('network\'))
rootFolder = '';

net = create_WFP(...
    "output_dims",3,...
    "layers",2,...
    "hash_level",12,...
    "bounding_box",[0,0;1,1]);


optimizer_E = optimizers.YoGi(0.9,0.99,1e-4);

learnRate = 0.01;

%begin training
tStart = tic;
for  iteration = 1:480
    
    start_timer = tic;
    [loss,dldw] = dlfeval(@model_loss, net, pos_batch, target);
    % toc
    this_loss = extractdata(loss);
    fprintf("loss = %.3f,  iter takes = %.3f \n",this_loss,toc(start_timer));    
    net = optimizer_E.step(net,dldw,iteration,learnRate);
    % 
    % if mod(iteration,50) == 0 
    %     [img_test, score] = validation(net, pos_batch, img, size(img));
    %     figure(121);
    %     imshow(img_test,[])
    %     drawnow;
    %     % fprintf("training takes: %.3f s, psnr = %.4f \n",toc(tStart),score);
    % end
    
    
    % 
    % if iteration > 300 && (mod(iteration,100) == 0)
    %     learnRate = learnRate * 0.7;
    % end
    % 
    % if learnRate < 1e-9
    %     break;
    % end
end



function [loss,dldw] = model_loss(net, xyzs, target)
start_timer = tic;
predict = net.forward(xyzs);
wait(gpuDevice());
fprintf("forward takes %.3f s \n",toc(start_timer));

start_timer = tic;
loss = l2loss(predict,target);
dldw = dlgradient(loss, net.Learnables);
wait(gpuDevice());
fprintf("backward takes %.3f s \n",toc(start_timer));

end

function [img, score] = validation(net,xyzs,target,img_sz)
predict = net.forward(xyzs);
img = extractdata(predict)';
img = reshape(img,[img_sz(1),img_sz(2),3]);

score = psnr(img,target);
end