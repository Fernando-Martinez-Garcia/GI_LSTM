function [Xn,Tn,RX,RY,nX,nY] = LSTM_Preprocessing_X_Y_1c_eco(X,Y,flag_mM_sd)
% This function linearly maps X to the hypercube [-1,1]^nX(2), as well as T to 
% [-1,1]^nT(2)

nX=size(X);
nY=size(Y);
Xn=zeros(nX(1),nX(2));
Tn=zeros(nY(1),nY(2));
RX=zeros(nX(2),2);
RY=zeros(nY(2),2);

if flag_mM_sd==0
    for i=1:nX(2)
        r1s=(max(X(:,i))+min(X(:,i)))/2;
        r1d=(max(X(:,i))-min(X(:,i)))/2;
        RX(i,1)=r1s;
        RX(i,2)=r1d;
        Xn(:,i)=(X(:,i)-r1s)/r1d;
    end
    
    for i=1:nY(2)
        r1sT=(max(Y(:,i))+min(Y(:,i)))/2;
        r1dT=(max(Y(:,i))-min(Y(:,i)))/2;
        RY(i,1)=r1sT;
        RY(i,2)=r1dT;
        Tn(:,i)=(Y(:,i)-r1sT)/r1dT;
    end
else
    for i=1:nX(2)
        r1s=mean(X(:,i));
        r1d=sqrt(var(X(:,i)));
        RX(i,1)=r1s;
        RX(i,2)=r1d;
        Xn(:,i)=(X(:,i)-r1s)/r1d;
    end
    
    for i=1:nY(2)
        r1sT=mean(Y(:,i));
        r1dT=sqrt(var(Y(:,i)));
        RY(i,1)=r1sT;
        RY(i,2)=r1dT;
        Tn(:,i)=(Y(:,i)-r1sT)/r1dT;
    end
end

end