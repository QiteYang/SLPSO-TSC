function x6 = testNet(x, net)

    N=size(x,1);
    W=net.W;B=net.B;
    %% forward
    x1 = x;
    x2=x1*W{1}+repmat(B{1},N,1);%N*neuronN
    x3=max(0,x2);%ReLU
    x4=x3*W{2}+repmat(B{2},N,1);
    x5=(exp(x4)-exp(-x4))./(exp(x4)+exp(-x4));%Sigmoid%N*neuronN
    x6=x5*W{3}+repmat(B{3},N,1);%N*M
end