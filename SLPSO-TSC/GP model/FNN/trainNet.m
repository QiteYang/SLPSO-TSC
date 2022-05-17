function net = trainNet(x, y, Params, net)

    N=size(x,1);
    learnR=Params.learnR;
    W=net.W;B=net.B;

    %% forward
    x1 = x;
    x2=x1*W{1}+repmat(B{1},N,1);%N*neuronN
    x3=max(0,x2);%ReLU
    x4=x3*W{2}+repmat(B{2},N,1);
    x5=(exp(x4)-exp(-x4))./(exp(x4)+exp(-x4));%Sigmoid%N*neuronN
    x6=x5*W{3}+repmat(B{3},N,1);%N*M

    %% backward
    %cost_loss=sum(sum(0.5*(x7-y).^2));
    %fc x6-x7
    e=x6-y;%N*M
    dW{3}=x5'*e;%neuronN*M
    dB{3}=sum(e);%1*M
    dx2=e*W{3}';%N*neuronN
    %sigmoid x5-x6
    e2=dx2.*(1-x5.^2);%N*neuronN   
    %fc x4-x5
    dW{2}=x3'*e2;%neuronN*neuronN
    dB{2}=sum(e2);%1*neuronN
    dx1=e2*W{2}';%N*neuronN
    dx1(find(x3<=0))=0;%N*neuronN
    %fc x1-x2
    dW{1}=x1'*dx1;%V*neuronN
    dB{1}=sum(dx1);%1*neuronN

    decay=1e-05;
    for k=1:3
        W{k}=W{k}-(decay*W{k}+dW{k})/N*learnR;
        B{k}=B{k}-dB{k}/N*learnR;
    end
    net.W=W;net.B=B;
end