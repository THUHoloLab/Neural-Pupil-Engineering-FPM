clear
clc
reset(gpuDevice());


led_num = [1,8,12,16,24,32];
led_total = sum(led_num(:));
rot_ang = 0 / 180 * pi;


name = 100;
learning_rate = [0.002,0.002,0.002];

for color_index = 1
% Load data
load(['raw_img_data_color=',num2str(color_index),'.mat']);
    
img_raw_rgb(:,:,color_index) = imRaw_new(:,:,1);

pix = size(imRaw_new,1);
[f_pos_set_true,pratio,Pupil0,fx_data,...
                              coor_xy] = srcs.init_environment_rgb(...
                                                      color_index,...
                                                      pix, ...
                                                      led_num, ...
                                                      rot_ang);




batchSize = 15; 

numEpochs = 60;
numIterationsPerEpoch  = size(imRaw_new,3) / batchSize;
numIterations = numEpochs * numIterationsPerEpoch;

epoch = 0;
iteration = 0;

%% The iterative recovery process for FP
disp('initializing parameters')
foo = @(x) complex(gpuArray(single(x)));

oI = (imresize(mean(imRaw_new(:,:,1),3),pratio)); 

wavefront1 = dlarray(foo(oI));  
Pupil0 = single(gpuArray(Pupil0));



error_bef = inf;

optimizer_w1 = srcs.optimizers.Adam(0.9,0.99,1e-7);
optimizer_w2 = srcs.optimizers.Adam(0.9,0.99,1e-15);
learning_rate = 0.01;

fx_CCD = fx_data.fx_CCD;
fy_CCD = fx_data.fy_CCD;
dff = fx_data.df;

pupil_net = network.NGP("base_res",32,...
                        "high_res",pix,...
                        "layers",2,...
                        "hash_level",12,...
                        "feature_dim",4,...
                        "log2_hashmap_size",19,...
                        "bounding_box",[min(fx_CCD(:)),min(fy_CCD(:));...
                                        max(fx_CCD(:)),max(fy_CCD(:))]);

pupil_coor = gpuArray(single([fx_CCD(:),fy_CCD(:)]));
pupil_coor = dlarray(pupil_coor,"BC");

%% initialize pupil function
disp('begin solving-----')
for con = 1:20
    [dldw1,ppp] = dlfeval(@model_loss,pupil_net,...
                              pupil_coor, ...
                              Pupil0);

    pupil_net = optimizer_w2.step(pupil_net,dldw1,con,0.002);
end

% done

cudaFPM = autoDiff_cuFPM(true);

while epoch < numEpochs
    epoch = epoch + 1;

    first = 1;
    last = 1;

    tic;
    mask = 0;
    while last < led_total
        iteration = iteration + 1;
        
        last = first + batchSize - 1;

        leds = f_pos_set_true(first:min(last,led_total),:);

        [loss,dldw1,dldw2,wavefront2] = dlfeval(@autodiff_fpm,...
                                           cudaFPM,...
                                           wavefront1, ...
                                           pupil_net, ...
                                           pupil_coor, ...
                                           Pupil0, ...
                                           imRaw_new(:,:,first:min(last,led_total)), ...
                                           int32(gpuArray([leds(:,1)';leds(:,3)'])), ...
                                           pratio);

        wait(gpuDevice());
 
    
        first = last + 1;
 
        % learning the parameters
        wavefront1 = optimizer_w1.step(wavefront1,dldw1,iteration,learning_rate);
        pupil_net = optimizer_w2.step(pupil_net,dldw2,iteration,learning_rate/2);
    end
    clc
    
    this_timer = toc;
    sprintf("at %d epoch, takes = %2f",epoch,this_timer)

    if mod(epoch,1) == 0 
        figure(7);

        w1 = gather(extractdata(wavefront1));
        w2 = gather(extractdata(wavefront2));

        subplot(121);imshow((abs(w1)),[]);
        subplot(122);imshow((abs(w2)),[]);
        title(['Iteration No. = ',int2str(epoch), '  \alpha = ',num2str(learning_rate)])
        drawnow;
    end

    if epoch > 150
        learning_rate = 0.5 * learning_rate;
    end
end

w1 = extractdata(wavefront1);
w2 = extractdata(wavefront2);
end


%% helper function1
function [loss,dldw1,dldw2,wavefront2] = autodiff_fpm(cudaFPM,...
                                                      wavefront1, ...
                                                      pupil_net, ...
                                                      pupil_path,...
                                                      pupil,...
                                                      obse_Y, ...
                                                      ledIdx, ...
                                                      pratio)


predict = real(stripdims(pupil_net.forward(pupil_path)));
predict = reshape(predict',[size(pupil,1),size(pupil,2),2]);

wavefront2 = 0.5*(sin(predict(:,:,1)) + 1) .* ...
                  exp(2 .* 1i .* pi .* sin(predict(:,:,2)));

pred_Y = cudaFPM(wavefront1, ...
                 wavefront2, ...
                 obse_Y, ...
                 single(ledIdx), ...
                 pratio);

loss = srcs.fd_loss(pred_Y, obse_Y, 'isotropic');

[dldw1,dldw2] = dlgradient(loss,wavefront1,pupil_net.Learnables);
end

%% helper function2 
function [dldw1,wavefront2] = model_loss(pupil_net,...
                                         pupil_path, ...
                                         Pupil0)

predict2 = stripdims(pupil_net.forward(pupil_path));
predict2 = real(reshape(predict2',[size(Pupil0,1),size(Pupil0,2),2]));
wavefront2 = 0.5*(sin(predict2(:,:,1)) + 1) .* exp(2 .* 1i .* pi .* sin(predict2(:,:,2)));

loss = mean(abs(wavefront2 - Pupil0).^2,'all');

[dldw1] = dlgradient(loss,pupil_net.Learnables);

end

