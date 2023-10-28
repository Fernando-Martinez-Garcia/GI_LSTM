function [Xn,Yn,RX,RY,Cluster_V,Cluster_incV,Cluster_Vy,Cluster_incVy] = GLSTM_Initialization_M(X,Y,Flag_mM_sd,number_hidden_layers,number_hidden_units,number_M_parameters,Flag_u_g)
epsilon=0.1;

%This function maps (normalizes) the values of input and output matrix;

[Xn,Yn,RX,RY,nX,nY] = GI_LSTM_Preprocessing_X_Y(X,Y,Flag_mM_sd);
%vector of dimensions in the different layers(including input and ouput) with size "HL+2"
V_D=[[nX(2),number_hidden_units(1:end-1)]+number_hidden_units+1;number_hidden_units];

%initial_weigths and biases
Cluster_V=cell(number_hidden_layers,4+2*length(number_M_parameters));
Cluster_Vy=cell(1,1);
Cluster_incV=cell(number_hidden_layers,4+2*length(number_M_parameters));
Cluster_incVy=cell(1,1);

Gates_Matrix_indexes=[1:4,4+2*(1:length(number_M_parameters))-1];
Momentum_Matrix_indexes=4+2*(1:length(number_M_parameters));

if Flag_u_g==0%uniform distribution
    for i=1:number_hidden_layers
        for j=Gates_Matrix_indexes
            aux_V=2*rand(V_D(2,i),V_D(1,i))-1;
%             aux_V(:,V_D(1,i)-number_hidden_units(i))=zeros(size(aux_V,1),1)            
            if (j==3)||(j==5)||(j==7)
                aux_V=epsilon*aux_V/norm(aux_V);
                aux_V(:,V_D(1,i)-number_hidden_units(i))=1*ones(size(aux_V,1),1);
                Cluster_V{i,j}=aux_V;
            else
                Cluster_V{i,j}=epsilon*aux_V/norm(aux_V);
            end
            Cluster_incV{i,j}=zeros(V_D(2,i),V_D(1,i));
        end
        for jm=1:length(number_M_parameters)
            aux_V=2*rand(number_hidden_units(i),number_M_parameters(jm))-1;
            Cluster_V{i,Momentum_Matrix_indexes(jm)}=aux_V./sum(abs(aux_V),2);
            Cluster_incV{i,Momentum_Matrix_indexes(jm)}=zeros(number_hidden_units(i),number_M_parameters(jm));
        end
    end
    aux_Vy=epsilon*(2*rand(nY(2),V_D(2,end)+1)-1);
%     aux_Vy(:,end)=zeros(size(aux_Vy,1),1);
    Cluster_Vy{1}=aux_Vy/norm(aux_Vy);
    Cluster_incVy{1}=zeros(nY(2),V_D(2,end)+1);
else%gaussian distribution
    for i=1:number_hidden_layers
        for j=Gates_Matrix_indexes
            aux_V=randn(V_D(2,i),V_D(1,i));
%             aux_V(:,V_D(1,i)-number_hidden_units(i))=zeros(size(aux_V,1),1);
            if j==3
                Cluster_V{i,j}=epsilon*aux_V/norm(aux_V);
            else
                Cluster_V{i,j}=epsilon*aux_V/norm(aux_V);
            end
            Cluster_incV{i,j}=zeros(V_D(2,i),V_D(1,i));
        end
        for jm=1:length(number_M_parameters)
            aux_V=randn(number_hidden_units(i),number_M_parameters(jm));
            Cluster_V{i,number_M_parameters(jm)}=aux_V/sum(abs(aux_V),2);
            Cluster_incV{i,number_M_parameters(jm)}=zeros(number_hidden_units(i),number_M_parameters(jm));
        end
    end
    aux_Vy=epsilon*randn(nY(2),V_D(2,end)+1);
%     aux_Vy(:,end)=zeros(size(aux_Vy,1),1);
    Cluster_Vy{1}=aux_Vy/norm(aux_Vy);
    Cluster_incVy{1}=zeros(nY(2),V_D(2,end)+1);
end

end

