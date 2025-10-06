% 13*13
%% Simulate the forward imaging process of Fourier ptychography
clear
clc
reset(gpuDevice());


led_num = [1,8,12,16,24,32];
led_total = sum(led_num(:));
rot_ang = 0 / 180 * pi;

learning_rate = [0.002,0.002,0.002];

for color_index = 1:3

load(['raw_img_data_color=',num2str(color_index),'.mat']);

pix = size(imRaw_new,1);

[f_pos_set_true,...
 pratio,...
 Pupil0,~,~] = srcs.init_environment_rgb(color_index,...
                                         pix, ...
                                         led_num, ...
                                         rot_ang);

batchSize = 15; 

numEpochs = 50;
numIterationsPerEpoch  = size(imRaw_new,3) / batchSize;
numIterations = numEpochs * numIterationsPerEpoch;

epoch = 0;
iteration = 0;

%% The iterative recovery process for FP
disp('initializing parameters')
foo = @(x) complex(gpuArray(single(x)));

oI = (imresize(mean(imRaw_new(:,:,1),3),pratio)); 
wavefront1 = dlarray(foo(oI));
wavefront2 = dlarray(foo(Pupil0));   

disp('begin solving-----')

optimizer_w1 = srcs.optimizers.Adam();
optimizer_w2 = srcs.optimizers.Adam();

lr = 0.01;

cudaForward_foo = autoDiff_cuFPM(true); % init differentiable function
while epoch < numEpochs
    epoch = epoch + 1;

    first = 1;
    last = 1;

    start_tic = tic;
    mask = 0;
    while last < led_total
        iteration = iteration + 1;
        
        last = first + batchSize - 1;
        leds = f_pos_set_true(first:min(last,led_total),:);
        kt = leds(:,3);
        kl = leds(:,1);

        ledIdx = int32(gpuArray([kl';kt']));
       

        [loss,dldw1,dldw2] = dlfeval(@autodiff_fpm,...
                                           cudaForward_foo,...
                                           wavefront1, ...
                                           wavefront2, ...
                                           imRaw_new(:,:,first:min(last,led_total)), ...
                                           ledIdx, ...
                                           pratio);

        wait(gpuDevice());
 
    
        first = last + 1;
  
        wavefront1 = optimizer_w1.step(wavefront1,dldw1,iteration,lr);
        wavefront2 = optimizer_w2.step(wavefront2,dldw2,iteration,lr);
    end
    clc
    
    this_timer = toc(start_tic);
    sprintf("at %d epoch, takes = %2f",epoch,this_timer)

    if mod(epoch,1) == 0 
        w2 = gather(extractdata(wavefront1));

        c1 = fftshift(fft2(w2));

        figure(7);
        w1 = gather(extractdata(wavefront2));
        subplot(121);imshow((abs(w2)),[]);
        subplot(122);imshow((abs(w1)),[]);
        title(['Iteration No. = ',int2str(epoch), '  \alpha = ',num2str(lr)])
        drawnow;
    end

end

w1 = extractdata(wavefront2);
w2 = extractdata(wavefront1);
end

%% helpers
function [loss,dldw1,dldw2] = autodiff_fpm(foo_fpm,...
                                           wavefront1, ...
                                           wavefront2, ...
                                           obse_Y, ...
                                           ledIdx, ...
                                           pratio)

pred_Y = foo_fpm(wavefront1, ...
                 wavefront2, ...
                 obse_Y, ...
                 single(ledIdx), ...
                 pratio);

loss = srcs.fd_loss(pred_Y, obse_Y, 'isotropic');

[dldw1, dldw2] = dlgradient(loss, wavefront1, wavefront2);
end