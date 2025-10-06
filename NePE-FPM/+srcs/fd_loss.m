function loss = fd_loss(x, y, type)
dW = x - y;
RxdW = [dW(:,2:end,:) - dW(:,1:end-1,:), dW(:,1,:) - dW(:,end,:)];
RydW = [dW(2:end,:,:) - dW(1:end-1,:,:); dW(1,:,:) - dW(end,:,:)];

switch type
    case 'isotropic'
        loss = sqrt(RxdW.^2 + RydW.^2  + 1e-5);
    case 'anisotropic' 
        loss = abs(RxdW) + abs(RydW);
    otherwise
    error("parameter #3 should be a string either 'isotropic', or 'anisotropic'")
end
loss = sum(loss(:));
end