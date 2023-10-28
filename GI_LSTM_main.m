%-----------User-defined hyperparameters-----------%
network_size_type=1;%predefined network size type.
sub_seq_length=400;%length of the subsequences to be used.
M_q_dependence=[120,0];%Memory-group max dependencies [q1,q2]
Learning_rate=0.001;%Learning rate for the ADAM method.
plot_on=10;%Visualization of thee validation set fitting process.
% Maximum number of iterations in the training proccess.
number_training_iterations=20000;
% Number_val_fails is the number of semi-consecutive increasing erorrs
%allowed before stopping the training.
number_val_fails=20000;
%-------------------------------------------------%

%Array containing the number of units for different predefined network sizes
number_units_array=2.^[0:8];

Flag_mM_sd=1;
Flag_u_g=0;
plot_heat_on=0;
for kkkk=network_size_type
number_hidden_units=number_units_array(network_size_type);
number_hidden_layers=size(number_hidden_units,2);
Backward_window=sub_seq_length;
Fordward_window=Backward_window;
Step_window=round(Fordward_window/1);
disp(['number hidden units:',num2str(number_hidden_units)]);
for jjjj=2:2
    jjjj
meta_iterations=1;
Cluster_Ye=cell(meta_iterations,1);
Cluster_Ye1=cell(meta_iterations,1);
E_Meta_train=zeros(meta_iterations,1);
E_Meta_val=zeros(meta_iterations,1);
E_Meta_testing=zeros(meta_iterations,1);
Iterations_optimal=zeros(meta_iterations,1);
Window_FF=zeros(meta_iterations,2);

for i_Meta=1:meta_iterations    
    i_Meta
    
    if M_q_dependence(2)>0
        number_M_parameters=M_q_dependence;
    else
        number_M_parameters=[M_q_dependence(1)];
    end
    
    [Xn,Yn,RX,RY,Cluster_V,Cluster_incV,Cluster_Vy,Cluster_incVy] = ...
    GI_LSTM_Initialization(X_train,Y_train,Flag_mM_sd,number_hidden_layers,number_hidden_units,number_M_parameters,Flag_u_g);
Cluster_incV_momentum=Cluster_incV;
Cluster_incV_momentumsqr=Cluster_incV_momentum;
Cluster_incVy_momentum=Cluster_incVy;
Cluster_incVy_momentumsqr=Cluster_incVy;
% //////////////////////////////////Training phase of the LSTM-NN//////////////////////////////////

%--------------------------------------------------------------------------%
% alpha is the learning rate of the LSTM-NN.
alpha_base=[Learning_rate,Learning_rate];
betha1=0.9;
betha2=0.999;
%--------------------------------------------------------------------------%
% lambda is used for L2 regularization
lambda=0.000./alpha_base(1);
number_parameters=(size(Xn,2)+number_hidden_units(1)+1+1/5+size(Yn,2)/5)*number_hidden_units(1)*5;
norm_gradient_minimum=0.001*sqrt(number_parameters)*Fordward_window;
%--------------------------------------------------------------------------%
% if Flag_val_enable is 1 the training stops based on performance of the 
%validation set (X_val,Y_val).
Flag_val_enable=1;
%--------------------------------------------------------------------------%

Scaling_DelC_th=100;
Scaling_DelC_prev=Scaling_DelC_th*ones(number_hidden_layers,ceil(size(Xn,1)/sub_seq_length));
Scaling_DelC_prev_aux=Scaling_DelC_th*ones(number_hidden_layers,ceil(size(Xn,1)/sub_seq_length));
%--------------------------------------------------------------------------%
%The output of the LSTM-NN will be used as part of input X
Flag_output_feedback_training=0;
Flag_output_feedback_validation=0;
Flag_output_feedback_testing=0;
Flag_MA=0;
Flag_Gradient_weigth_saturation=0;
Gradient_threshold=1.0;
Columns_replaced=[1];

%auxiliary variables initialization
cont_val_fails=0;
error_val_old=inf;
error_train_old=inf;
Cluster_V_val=cell(size(Cluster_V));
Cluster_Vy_val=cell(1,1);
%--------------------------------------------------------------------------%
%forward-passes-information storing through one or more hidden layers
Cluster_H=cell(number_hidden_layers,2);%nodes values
Cluster_H_sub_seq=zeros(number_hidden_units(end),sub_seq_length);%nodes values
Cluster_A=cell(number_hidden_layers,2);%simple activation values
Cluster_I=cell(number_hidden_layers,2);%input gate values
Cluster_F=cell(number_hidden_layers,2);%forget gate values
Cluster_O=cell(number_hidden_layers,2);%output gate values
Cluster_F1=cell(number_hidden_layers,2);%forget gate values
Cluster_F2=cell(number_hidden_layers,2);%forget gate values
Cluster_F1_temporal=cell(number_hidden_layers,size(Xn,1));
Cluster_F2_temporal=cell(number_hidden_layers,size(Xn,1));
Array_w_epoch=zeros(number_M_parameters(1),number_training_iterations,number_hidden_layers);
if size(number_M_parameters,2)>1
    Array_w_epoch2=zeros(number_M_parameters(2)+1,number_training_iterations,number_hidden_layers);
end

if ~isempty(number_M_parameters)
    Matrix_layers_C_prev_sub_seq=zeros(sum(number_hidden_units),number_M_parameters(1));
    if length(number_M_parameters)>1
        max_prev_dependency=(number_M_parameters(2)+1)*number_M_parameters(1);
        Matrix_layers_M1_prev_sub_seq=zeros(sum(number_hidden_units),max_prev_dependency);%cell values (internal state of LSTM)
    else
        max_prev_dependency=number_M_parameters(1);
        Matrix_layers_M1_prev_sub_seq=zeros(sum(number_hidden_units),max_prev_dependency);%cell values (internal state of LSTM)
    end
else
    Matrix_layers_C_prev_sub_seq=zeros(sum(number_hidden_units),1);
    max_prev_dependency=1;
end

Matrix_layers_C=zeros(sum(number_hidden_units),sub_seq_length);%cell values (internal state of LSTM)
Matrix_layers_M1=zeros(sum(number_hidden_units),sub_seq_length);
Matrix_layers_M2=zeros(sum(number_hidden_units),sub_seq_length);
Matrix_layers_M2_prev_sub_seq=zeros(sum(number_hidden_units),1);%cell values (internal state of LSTM)

rows_M_L_i=size(Xn,2)+number_hidden_layers+number_hidden_units(end)+2*sum(number_hidden_units(1:end-1));

Matrix_LSTM_input=zeros(rows_M_L_i,sub_seq_length);
rows_LSTM_input=zeros(number_hidden_layers,2);
for i_C_LSTM_input=1:number_hidden_layers
    if i_C_LSTM_input==1
        rows_LSTM_input(i_C_LSTM_input,2)=size(Xn,2)+1+number_hidden_units(1);
    else
        rows_LSTM_input(i_C_LSTM_input,2)=number_hidden_units(i_C_LSTM_input-1)...
            +1+number_hidden_units(i_C_LSTM_input)+rows_LSTM_input(i_C_LSTM_input-1,2);
    end
end
rows_LSTM_input(1,1)=1;
rows_LSTM_input(2:end,1)=1+rows_LSTM_input(1:end-1,2);

Matrix_Ye_sub_seq_training=zeros(size(Yn,2),sub_seq_length);
Matrix_error=zeros(size(Matrix_Ye_sub_seq_training));

Cluster_DelH=cell(number_hidden_layers,2);%gradient of the cost function with respect tp nodes
Cluster_DelInputs=cell(number_hidden_layers,sub_seq_length);%gradient of the cost function with respect to inputs to the LSTM units
Cluster_DelA=cell(number_hidden_layers,2);%gradient of the cost function with respect to activation values
Cluster_DelI=cell(number_hidden_layers,2);%gradient of the cost function with respect to input gate values
Cluster_DelF=cell(number_hidden_layers,2);%gradient of the cost function with respect to forget gate values
Cluster_DelF1=cell(number_hidden_layers,2);%gradient of the cost function with respect to forget gate values
Cluster_DelF2=cell(number_hidden_layers,2);%gradient of the cost function with respect to forget gate values
Cluster_DelO=cell(number_hidden_layers,2);%gradient of the cost function with respect to output gate values
Matrix_DelC=zeros(sum(number_hidden_units),sub_seq_length);%gradient of the cost function with respect to cell values
Matrix_DelM1=zeros(sum(number_hidden_units),sub_seq_length);%Matrix storing previous gradients of the cell values
Matrix_DelM2=zeros(sum(number_hidden_units),sub_seq_length);%Matrix storing previous gradients of the cell values
if ~isempty(number_M_parameters)
    Matrix_DelW1=zeros(sum(number_hidden_units),number_M_parameters(1));%Matrix storing previous gradients of the cell values
    if length(number_M_parameters)>1
        Matrix_DelW2=zeros(sum(number_hidden_units),number_M_parameters(2));%Matrix storing previous gradients of the cell values
    end
end
row_CM_del=zeros(number_hidden_layers,2);

for i_CM=1:number_hidden_layers
    if i_CM==1
        row_CM_del(:,2)=number_hidden_units';
        row_CM_del(1,1)=1;
    else
        row_CM_del(i_CM,2)=row_CM_del(i_CM-1,2)+row_CM_del(i_CM,2);
    end
end
row_CM_del(2:end,1)=1+row_CM_del(1:end-1,2);
row_CM=row_CM_del;

tic
betha1_acum=1;
betha2_acum=1;
for i_t_i=1:number_training_iterations
    
    alpha = ones(number_hidden_layers,1)*alpha_base(1);
    cont_sub_seq=Fordward_window;
    Flag_stop=1;
    i_sub_seq=0;
    Scaling_DelC_prev_aux=Scaling_DelC_th*ones(number_hidden_layers,ceil(size(Xn,1)/sub_seq_length));
    while(Flag_stop)
        betha1_acum=betha1*betha1_acum;
        betha2_acum=betha2*betha2_acum;
        if (cont_sub_seq>=size(Xn,1))
            cont_sub_seq=size(Xn,1);
            Flag_stop=0;
        end
        %////////////////////////////////////////////Forward pass////////////////////////////////////////////%
        sub_seq_Xn=Xn(cont_sub_seq-Fordward_window+1:cont_sub_seq,:)';
        sub_seq_Yn=Yn(cont_sub_seq-Fordward_window+1:cont_sub_seq,:)';
        for i_b_s=1:sub_seq_length
            for i_h_l=1:number_hidden_layers
                
                if (cont_sub_seq-Fordward_window>0)&&(i_b_s==1)
                    index_prev_sub_seq=max(number_M_parameters(1),1);
                    for i_p_bc=1:index_prev_sub_seq
                        Matrix_layers_C_prev_sub_seq(row_CM(i_h_l,1):row_CM(i_h_l,2),i_p_bc)=...
                            Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),end-index_prev_sub_seq+i_p_bc);
                    end
                    if ~isempty(number_M_parameters)                        
                        if length(number_M_parameters)>1
                            index_prev_sub_seq2=max(number_M_parameters(2)*number_M_parameters(1),1);
                            Matrix_layers_M2_prev_sub_seq(row_CM(i_h_l,1):row_CM(i_h_l,2),end)=...
                                Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),end);
                        else
                            index_prev_sub_seq2=max(number_M_parameters(1),1);
                        end
                        for i_p_bM1=1:index_prev_sub_seq2
                            Matrix_layers_M1_prev_sub_seq(row_CM(i_h_l,1):row_CM(i_h_l,2),i_p_bM1)=...
                                Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),end-index_prev_sub_seq2+i_p_bM1);
                        end                         
                    end
                end
                
                %-----------------selection of the Input values to the LSTM units-------------------%
                if cont_sub_seq-Fordward_window+i_b_s==1
%                 if i_b_s==1
                    if i_h_l==1
                        Vector_LSTM_input=[sub_seq_Xn(:,i_b_s);1;zeros(number_hidden_units(1),1)];
                    else
                        Vector_LSTM_input=[Cluster_H{i_h_l-1,1};1;zeros(number_hidden_units(i_h_l),1)];
                    end
                else
                    if i_h_l==1
                        if Flag_output_feedback_training==1
                            sub_seq_Xn_modified=X_modification_output_feedback_1c_eco(sub_seq_Xn(:,i_b_s),...
                                Matrix_Ye_sub_seq_training(:,i_b_s-1),Columns_replaced);
                        else
                            sub_seq_Xn_modified=sub_seq_Xn(:,i_b_s);
                        end
                        if Flag_MA==1                            
                            sub_seq_Xn_modified=X_modification_output_MA_1c_eco(sub_seq_Xn(:,i_b_s),...
                                Flag_MA,sub_seq_Yn(:,i_b_s-1)-Matrix_Ye_sub_seq_training(:,i_b_s-1));
                        end
                        Vector_LSTM_input=[sub_seq_Xn_modified;1;Cluster_H{1,2}];
                    else
                        Vector_LSTM_input=[Cluster_H{i_h_l-1,1};1;Cluster_H{i_h_l,2}];
                    end
                end
                %-----------------------------------------------------------------------------------%
                
                %-------------------------Acessing weights------------------------%
                Matrix_Va_auxiliar=Cluster_V{i_h_l,1};
                Matrix_Vi_auxiliar=Cluster_V{i_h_l,2};
                Matrix_Vf_auxiliar=Cluster_V{i_h_l,3};
                Matrix_Vo_auxiliar=Cluster_V{i_h_l,4};
                if ~isempty(number_M_parameters)
                    Matrix_Vf1_auxiliar=Cluster_V{i_h_l,5};
                    den_WM1_auxiliar=sum(abs(Cluster_V{i_h_l,6}),2);
                    Matrix_WM1_auxiliar=Cluster_V{i_h_l,6}./den_WM1_auxiliar;
                    if length(number_M_parameters)>1
                        Matrix_Vf2_auxiliar=Cluster_V{i_h_l,7};
                        den_WM2_auxiliar=sum(abs(Cluster_V{i_h_l,8}),2);
                        Matrix_WM2_auxiliar=Cluster_V{i_h_l,8}./den_WM2_auxiliar;
                    else
                         Matrix_Vf2_auxiliar=Cluster_V{i_h_l,3}*0;
                         Matrix_WM2_auxiliar=Cluster_V{i_h_l,3}*0;
                    end
                else
                    Matrix_Vf1_auxiliar=Cluster_V{i_h_l,3}*0;
                    Matrix_Vf2_auxiliar=Cluster_V{i_h_l,3}*0;
                    Matrix_WM1_auxiliar=Cluster_V{i_h_l,3}*0;
                    Matrix_WM2_auxiliar=Cluster_V{i_h_l,3}*0;
                end
                %-----------------------------------------------------------------%
                
                %-------------------Non-linear transformations--------------------%
                Cluster_A{i_h_l,1}=tanh(Matrix_Va_auxiliar*Vector_LSTM_input);
                Cluster_I{i_h_l,1}=0.5*tanh(Matrix_Vi_auxiliar*Vector_LSTM_input/2)+0.5;
                Cluster_F{i_h_l,1}=0.5*tanh(Matrix_Vf_auxiliar*Vector_LSTM_input/2)+0.5;
                if ~isempty(number_M_parameters)
                    Cluster_F1{i_h_l,1}=0.5*tanh(Matrix_Vf1_auxiliar*Vector_LSTM_input/2)+0.5;
                else
                    Cluster_F1{i_h_l,1}=Cluster_I{i_h_l,1}*0;
                end
                if length(number_M_parameters)>1
                    Cluster_F2{i_h_l,1}=0.5*tanh(Matrix_Vf2_auxiliar*Vector_LSTM_input/2)+0.5;
                else
                    Cluster_F2{i_h_l,1}=Cluster_I{i_h_l,1}*0;
                end
                Cluster_O{i_h_l,1}=0.5*tanh(Matrix_Vo_auxiliar*Vector_LSTM_input/2)+0.5;
                %-----------------------------------------------------------------%
                Cluster_F1_temporal{i_h_l,cont_sub_seq-Fordward_window+i_b_s}=...
                    Cluster_F1{i_h_l,1};
                Cluster_F2_temporal{i_h_l,cont_sub_seq-Fordward_window+i_b_s}=...
                    Cluster_F2{i_h_l,1};
                %-----------------------LSTM input update-------------------------%
                Matrix_LSTM_input(rows_LSTM_input(i_h_l,1):rows_LSTM_input(i_h_l,2),i_b_s)=...
                    Vector_LSTM_input;
                %-----------------------------------------------------------------%
                
                if cont_sub_seq-Fordward_window+i_b_s>1
                    if (~isempty(number_M_parameters))
                        if (length(number_M_parameters)>1)&&(i_b_s-number_M_parameters(1)>1)
                                final_indexM2=i_b_s-number_M_parameters(1);
                                initial_index_valueM2=max(1,...
                                    final_indexM2-number_M_parameters(1)*...
                                    min(floor(final_indexM2/number_M_parameters(1)),number_M_parameters(2)-1));
                                M2_index=fliplr(initial_index_valueM2:number_M_parameters(1):final_indexM2);
                                Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                                    sum(Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),M2_index).*Matrix_WM2_auxiliar(:,1:length(M2_index)),2);
                            
                                Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                                    sum(Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-(1:number_M_parameters(1))).*Matrix_WM1_auxiliar,2);
                                
                                c_f=Cluster_F1{i_h_l,1}+Cluster_F2{i_h_l,1}+10^-10;
                                a_f=Cluster_F1{i_h_l,1}./c_f;
                                b_f=Cluster_F2{i_h_l,1}./c_f;
                                Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...                                    
                                    Cluster_I{i_h_l,1}.*Cluster_A{i_h_l,1}+...
                                    a_f.*Cluster_F1{i_h_l,1}.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)+...
                                    b_f.*Cluster_F2{i_h_l,1}.*Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                        elseif(length(number_M_parameters)>1)&&(cont_sub_seq-Fordward_window-number_M_parameters(1)>1)%This assumes that sub_seq size is bigger than number_M_parameters(1)
                            final_indexM2=i_b_s+cont_sub_seq-number_M_parameters(1);
                                initial_index_valueM2=max(1,...
                                    final_indexM2-number_M_parameters(1)*...
                                    min(floor(final_indexM2/number_M_parameters(1)),number_M_parameters(2)-1));
                                M2_index=fliplr(initial_index_valueM2:number_M_parameters(1):final_indexM2);
                                aux_i_au_M1=M2_index-cont_sub_seq;
                                Matrix_layers_M1_increased=[Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),aux_i_au_M1(aux_i_au_M1>0)),...
                                    Matrix_layers_M1_prev_sub_seq(row_CM(i_h_l,1):row_CM(i_h_l,2),end+aux_i_au_M1(aux_i_au_M1<=0))];
                                Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                                    sum(Matrix_layers_M1_increased.*Matrix_WM2_auxiliar(:,1:length(M2_index)),2);
                                
                                
                                aux_i_aug_C=i_b_s-(1:number_M_parameters(1));
                                Matrix_layers_C_increased=[Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),aux_i_aug_C(aux_i_aug_C>0)),...
                                    Matrix_layers_C_prev_sub_seq(row_CM(i_h_l,1):row_CM(i_h_l,2),end+aux_i_aug_C(aux_i_aug_C<=0))];
                                Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                                    sum(Matrix_layers_C_increased.*Matrix_WM1_auxiliar,2);
                                
                                c_f=Cluster_F1{i_h_l,1}+Cluster_F2{i_h_l,1}+10^-10;
                                a_f=Cluster_F1{i_h_l,1}./c_f;
                                b_f=Cluster_F2{i_h_l,1}./c_f;
                                Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...                                    
                                    Cluster_I{i_h_l,1}.*Cluster_A{i_h_l,1}+...
                                    a_f.*Cluster_F1{i_h_l,1}.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)+...
                                    b_f.*Cluster_F2{i_h_l,1}.*Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                        elseif(cont_sub_seq-Fordward_window>1)%This assumes that sub_seq size is bigger than number_M_parameters(1)
                            aux_i_aug_C=i_b_s-(1:number_M_parameters(1));
                            Matrix_layers_C_increased=[Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),aux_i_aug_C(aux_i_aug_C>0)),...
                                    Matrix_layers_C_prev_sub_seq(row_CM(i_h_l,1):row_CM(i_h_l,2),end+aux_i_aug_C(aux_i_aug_C<=0))];
                            Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                                sum(Matrix_layers_C_increased.*Matrix_WM1_auxiliar,2);
                            Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...                                
                                Cluster_I{i_h_l,1}.*Cluster_A{i_h_l,1}+...
                                Cluster_F1{i_h_l,1}.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                            
                        else
                            Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                                sum(Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-(1:min(number_M_parameters(1),i_b_s-1))).*...
                                Matrix_WM1_auxiliar(:,1:min(number_M_parameters(1),i_b_s-1)),2);
                            Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...                                
                                Cluster_I{i_h_l,1}.*Cluster_A{i_h_l,1}+...
                                Cluster_F1{i_h_l,1}.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                        end
                    else
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Cluster_F{i_h_l,1}.*Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-1)+...
                            Cluster_I{i_h_l,1}.*Cluster_A{i_h_l,1};
                    end
                else
                    Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=Cluster_I{i_h_l,1}.*Cluster_A{i_h_l,1};
                end
                
                Cluster_H{i_h_l,1}=Cluster_O{i_h_l,1}.*tanh(Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s));
                Cluster_H{i_h_l,2}=Cluster_H{i_h_l,1};%REDUNDACY MIGHT BE PRESENT HERE IN TERMS OF STORING
                
            end
            
            Cluster_H_sub_seq(:,i_b_s)=Cluster_H{end,1};
            Matrix_Ye_sub_seq_training(:,i_b_s)=Cluster_Vy{1,1}*[Cluster_H{end,1};1];
            
        end
        Matrix_error=sub_seq_Yn-Matrix_Ye_sub_seq_training;
        
        %----------------------------Gradients for the output weights----------------------------%
        Cluster_incVy{1,1}=Cluster_incVy{1,1}+...
            Matrix_error*[Cluster_H_sub_seq;ones(1,size(Cluster_H_sub_seq,2))]';
        %----------------------------------------------------------------------------------------%
        
        Cluster_incVy_momentum{1,1}=(1-betha1)*Cluster_incVy{1,1}+betha1*Cluster_incVy_momentum{1,1};
        Cluster_incVy_momentumsqr{1,1}=(1-betha1)*Cluster_incVy{1,1}.^2+betha1*abs(Cluster_incVy_momentum{1,1});
        
        %///////////////////////////////////////////////////////////////////////////////////////////////////////%
        %//////////////////////////////////////////////Backward Pass////////////////////////////////////////////%
        for i_h_l=1:number_hidden_layers
            
            rows_i_h_l_del=row_CM(end+1-i_h_l,1):row_CM(end+1-i_h_l,2);
            
            %-------------------------Acessing weights------------------------%
            Matrix_Va_auxiliar=Cluster_V{end+1-i_h_l,1};
            Matrix_Vi_auxiliar=Cluster_V{end+1-i_h_l,2};
            Matrix_Vf_auxiliar=Cluster_V{end+1-i_h_l,3};
            Matrix_Vo_auxiliar=Cluster_V{end+1-i_h_l,4};
            if ~isempty(number_M_parameters)
                Matrix_Vf1_auxiliar=Cluster_V{end+1-i_h_l,5};
                den_WM1_auxiliar=sum(abs(Cluster_V{end+1-i_h_l,6}),2);
                Matrix_WM1_auxiliar=Cluster_V{end+1-i_h_l,6}./den_WM1_auxiliar;
                if length(number_M_parameters)>1
                    Matrix_Vf2_auxiliar=Cluster_V{end+1-i_h_l,7};
                    den_WM2_auxiliar=sum(abs(Cluster_V{end+1-i_h_l,8}),2);
                    Matrix_WM2_auxiliar=Cluster_V{end+1-i_h_l,8}./den_WM2_auxiliar;
                else
                    Matrix_Vf2_auxiliar=Cluster_V{end+1-i_h_l,3}*0;
                    Matrix_WM2_auxiliar=Cluster_V{end+1-i_h_l,3}*0;
                end
            else
                Matrix_Vf1_auxiliar=Cluster_V{end+1-i_h_l,3}*0;
                Matrix_WM1_auxiliar=Cluster_V{end+1-i_h_l,3}*0;
                Matrix_Vf2_auxiliar=Cluster_V{end+1-i_h_l,3}*0;
                Matrix_WM2_auxiliar=Cluster_V{end+1-i_h_l,3}*0;
            end
            
            %-----------------------------------------------------------------%
            
            %----------------------------Gradients for the LSTM units weights-----------------------------%
            for i_b_s=1:sub_seq_length
                
                %--------------------selection of the Input values to the LSTM units----------------------%
                Vector_LSTM_input=Matrix_LSTM_input(rows_LSTM_input(end+1-i_h_l,1):rows_LSTM_input(end+1-i_h_l,2),...
                    end+1-i_b_s);
                %-----------------------------------------------------------------------------------------%
                
                %----------------------Gradient with respect to the hidden units--------------------------%
                if i_h_l==1
                    aux_Cluster_Vy=Cluster_Vy{1}';
                    if i_b_s==1
                        Cluster_DelH{end+1-i_h_l,1}=aux_Cluster_Vy(1:end-1,:)*Matrix_error(:,end+1-i_b_s);
                    else
                        Vector_X_next_H=Cluster_DelInputs{end+1-i_h_l,end+1-i_b_s+1};
                        Cluster_DelH{end+1-i_h_l,1}=aux_Cluster_Vy(1:end-1,:)*Matrix_error(:,end+1-i_b_s)+...
                            Vector_X_next_H(end+1-number_hidden_units(end+1-i_h_l):end);
                    end
                else
                    if i_b_s==1
                        Vector__next_layer_Input=Cluster_DelInputs{end+1-i_h_l+1,end+1-i_b_s};
                        Cluster_DelH{end+1-i_h_l,1}=...
                            Vector__next_layer_Input(1:number_hidden_units(end+1-i_h_l));
                    else
                        Vector__next_layer_Input=Cluster_DelInputs{end+1-i_h_l+1,end+1-i_b_s};
                        Vector_X_next_H=Cluster_DelInputs{end+1-i_h_l,end+1-i_b_s+1};
                        Cluster_DelH{end+1-i_h_l,1}=...
                            Vector__next_layer_Input(1:number_hidden_units(end+1-i_h_l))+...
                            Vector_X_next_H(end+1-number_hidden_units(end+1-i_h_l):end);
                    end                    
                end
                
                %-----------------------------------------------------------------------------------------%
                
                %---------------------Gradient with respect to output gate values-------------------------%                
                tanh_C_auxiliar=tanh(Matrix_layers_C(rows_i_h_l_del,end+1-i_b_s));
                Cluster_DelO{end+1-i_h_l,1}=Cluster_DelH{end+1-i_h_l,1}.*...
                    tanh_C_auxiliar;
                %-----------------------------------------------------------------------------------------%
                
                %-------------------------Gradient with respect to cell values----------------------------%
                Matrix_F=0.5*tanh(Matrix_Vf_auxiliar*Vector_LSTM_input/2)+0.5;
                if ~isempty(number_M_parameters)
                    if (sub_seq_length-i_b_s>0)
                        Matrix_F1=0.5*tanh(Matrix_Vf1_auxiliar*Vector_LSTM_input/2)+0.5;
                        if (length(number_M_parameters)>1)&&(sub_seq_length-i_b_s>number_M_parameters(1))
                            Matrix_F2=0.5*tanh(Matrix_Vf2_auxiliar*Vector_LSTM_input/2)+0.5;
                            c_f_del=Matrix_F1+Matrix_F2+10^-10;
                            a_f_del=Matrix_F1./c_f_del;
                            b_f_del=Matrix_F2./c_f_del;
                        else
                            Matrix_F2=Matrix_F*0;
                            c_f_del=Matrix_F1*0;
                            a_f_del=ones(size(Matrix_F1));
                            b_f_del=Matrix_F2*0;
                        end
                    else
                        Matrix_F2=Matrix_F*0;
                        Matrix_F1=Matrix_F*0;
                        c_f_del=Matrix_F1*0;
                        a_f_del=Matrix_F1*0;
                        b_f_del=Matrix_F2*0;                                                
                    end
                else
                    Matrix_F2=Matrix_F*0;
                    Matrix_F1=Matrix_F*0;
                    c_f_del=Matrix_F1*0;
                    a_f_del=Matrix_F1*0;
                    b_f_del=Matrix_F2*0;
                end
                
                Matrix_O=0.5*tanh(Matrix_Vo_auxiliar*Vector_LSTM_input/2)+0.5;
                
                Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)=...
                    Cluster_DelH{end+1-i_h_l,1}.*Matrix_O.*(1-tanh_C_auxiliar.^2);
                
                if ~isempty(number_M_parameters)
                    if (i_b_s>1)
                        Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)=...
                            Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)+...
                            sum(Matrix_DelM1(rows_i_h_l_del,end+1+1-i_b_s:end+1-max(i_b_s-number_M_parameters(1),1)).*...
                            Matrix_WM1_auxiliar(:,1:min(i_b_s-1,number_M_parameters(1))),2);%Rightmost element in the array associated to fartest future
                    end
                else
                    if i_b_s==1
                        F_DelC=zeros(size(Matrix_O));
                    else
                        Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)=...
                            Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)+...
                            F_DelC;
                    end
                end
                
                %-----------------------------------Gradient Saturation-----------------------------------%
                Scaling_DelC_prev_aux(end+1-i_h_l,i_sub_seq+1)=...
                    max(Scaling_DelC_prev_aux(end+1-i_h_l,i_sub_seq+1),...
                    max(max(abs(Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)),[],[1]),...
                    Scaling_DelC_th));

                n_factor_SDC=max(Scaling_DelC_prev(end+1-i_h_l,i_sub_seq+1),...
                    Scaling_DelC_prev_aux(end+1-i_h_l,i_sub_seq+1))./...
                    Scaling_DelC_th;

                index_th=sum(Scaling_DelC_prev(end+1-i_h_l,i_sub_seq+1)>Scaling_DelC_th,[1]);
                index_th=index_th+...
                    sum(Scaling_DelC_prev_aux(end+1-i_h_l,i_sub_seq+1)>Scaling_DelC_th,[1]);

                if sum(index_th)>0
                    Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)=...
                        Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)./...
                        n_factor_SDC;
                end
            
                %-----------------------------------------------------------------------------------------%
                               
                if (~isempty(number_M_parameters))&&(sub_seq_length-i_b_s>0)
                    Matrix_DelM1(rows_i_h_l_del,end+1-i_b_s)=a_f_del.*Matrix_F1.*Matrix_DelC(rows_i_h_l_del,end+1-i_b_s);
                    if (length(number_M_parameters)>1)&&(i_b_s>number_M_parameters(1))
                        final_del_M2_index=min(number_M_parameters(1)*(number_M_parameters(2)-1),i_b_s-1-number_M_parameters(1));
                        M2_index_del=sub_seq_length+1-i_b_s+number_M_parameters(1):number_M_parameters(1):sub_seq_length+1-i_b_s+number_M_parameters(1)+final_del_M2_index;%%dependent on sub_seq size
                        Matrix_DelM1(rows_i_h_l_del,end+1-i_b_s)=...
                            Matrix_DelM1(rows_i_h_l_del,end+1-i_b_s)+...
                            sum(Matrix_DelM2(rows_i_h_l_del,M2_index_del).*Matrix_WM2_auxiliar(:,1:length(M2_index_del)),2);%Rightmost element in the array associated to fartest future
                    end
                end   
                
                if (length(number_M_parameters)>1)&&(sub_seq_length-i_b_s>number_M_parameters(1))
                    Matrix_DelM2(rows_i_h_l_del,end+1-i_b_s)=b_f_del.*Matrix_F2.*Matrix_DelC(rows_i_h_l_del,end+1-i_b_s);
                end
                
                F_DelC=...
                    Matrix_DelC(rows_i_h_l_del,end+1-i_b_s).*Matrix_F;%Product's value of the cell and the forget gate                       
                %-----------------------------------------------------------------------------------------%
                
                %----------------------Gradient with respect to input gate values-------------------------%
                Matrix_A=tanh(Matrix_Va_auxiliar*Vector_LSTM_input);
                Cluster_DelI{end+1-i_h_l,1}=...
                    Matrix_DelC(rows_i_h_l_del,end+1-i_b_s).*Matrix_A;
                %-----------------------------------------------------------------------------------------%
                
                %----------------------Gradient with respect to forgate gate values-----------------------%
                if (~isempty(number_M_parameters))
                    Cluster_DelF{end+1-i_h_l,1}=Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)*0;
                    if (sub_seq_length-i_b_s>0)
                        Cluster_DelF1{end+1-i_h_l,1}=...
                            Matrix_DelC(rows_i_h_l_del,end+1-i_b_s).*...
                            Matrix_layers_M1(rows_i_h_l_del,end+1-i_b_s).*(2*a_f_del-a_f_del.^2);
                        if (length(number_M_parameters)>1)&&(sub_seq_length-i_b_s>number_M_parameters(1))
                            Cluster_DelF1{end+1-i_h_l,1}=...
                                Matrix_DelC(rows_i_h_l_del,end+1-i_b_s).*...
                                Matrix_layers_M2(rows_i_h_l_del,end+1-i_b_s).*(-b_f_del.^2);
                            
                            Cluster_DelF2{end+1-i_h_l,1}=...
                                Matrix_DelC(rows_i_h_l_del,end+1-i_b_s).*...
                                (Matrix_layers_M2(rows_i_h_l_del,end+1-i_b_s).*(2*b_f_del-b_f_del.^2)+...
                                Matrix_layers_M1(rows_i_h_l_del,end+1-i_b_s).*(-a_f_del.^2));
                        else
                            Cluster_DelF2{end+1-i_h_l,1}=Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)*0;
                        end
                    else
                        Cluster_DelF1{end+1-i_h_l,1}=Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)*0;
                        Cluster_DelF2{end+1-i_h_l,1}=Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)*0;
                    end
                else
                    if (sub_seq_length-i_b_s>0)
                        Cluster_DelF{end+1-i_h_l,1}=...
                            Matrix_DelC(rows_i_h_l_del,end+1-i_b_s).*...
                            Matrix_layers_C(rows_i_h_l_del,end+1-i_b_s-1);
                    else
                        Cluster_DelF{end+1-i_h_l,1}=Matrix_DelC(rows_i_h_l_del,end+1-i_b_s)*0;
                    end
                end
                %-----------------------------------------------------------------------------------------%
                
                %--------------------Gradient with respect to simple activation values--------------------%
                Matrix_I=0.5*tanh(Matrix_Vi_auxiliar*Vector_LSTM_input/2)+0.5;
                Cluster_DelA{end+1-i_h_l,1}=Matrix_DelC(rows_i_h_l_del,end+1-i_b_s).*...
                    Matrix_I;
                %-----------------------------------------------------------------------------------------%
                
                Vector_ZA_auxiliar=Cluster_DelA{end+1-i_h_l,1}.*(1-Matrix_A.^2);
                Vector_ZI_auxiliar=Cluster_DelI{end+1-i_h_l,1}.*(1-Matrix_I).*...
                    Matrix_I;
                if ~isempty(number_M_parameters)
                    Vector_ZWM1_auxiliar=zeros(length(rows_i_h_l_del),number_M_parameters(1));
                    if (sub_seq_length-i_b_s>0)
                        Vector_ZWM1_auxiliar(:,1:min(number_M_parameters(1),sub_seq_length-i_b_s))=...
                            Matrix_DelM1(rows_i_h_l_del,end+1-i_b_s).*...
                            Matrix_layers_C(rows_i_h_l_del,end+1-i_b_s-(1:min(number_M_parameters(1),sub_seq_length-i_b_s)));
                            %dependent on sub_seq size            
                    end
                    if (length(number_M_parameters)>1)
                        Vector_ZWM2_auxiliar=zeros(length(rows_i_h_l_del),number_M_parameters(2));
                    else
                        Vector_ZWM2_auxiliar=0;
                    end
                    if (sub_seq_length-i_b_s>number_M_parameters(1))&&(length(number_M_parameters)>1)
                        auxiliary_indexZWM2=sub_seq_length-i_b_s;%i_b_s is decreasing and initially bigger than number_M_parameters(1)
                                final_index_valueZWM2=i_b_s+number_M_parameters(1)*...
                                    min(floor(auxiliary_indexZWM2/number_M_parameters(1)),number_M_parameters(2));
                                M2_index=i_b_s+number_M_parameters(1):number_M_parameters(1):final_index_valueZWM2;%dependent on sub_seq size
                        Vector_ZWM2_auxiliar(:,1:min(number_M_parameters(2),length(M2_index)))=...
                            Matrix_DelM2(rows_i_h_l_del,end+1-i_b_s).*...
                            Matrix_layers_M1(rows_i_h_l_del,end+1-M2_index);
                            %dependent on sub_seq size            
                    end
                else
                    Vector_ZWM1_auxiliar=0;
                    Vector_ZWM2_auxiliar=0;
                end
                Vector_ZF_auxiliar=Cluster_DelF{end+1-i_h_l,1}.*(1-Matrix_F).*...
                    Matrix_F;
                if ~isempty(number_M_parameters)
                    Vector_ZF1_auxiliar=Cluster_DelF1{end+1-i_h_l,1}.*(1-Matrix_F1).*...
                        Matrix_F1;                    
                    if length(number_M_parameters)>1
                        Vector_ZF2_auxiliar=Cluster_DelF2{end+1-i_h_l,1}.*(1-Matrix_F2).*...
                        Matrix_F2;
                    else
                        Vector_ZF2_auxiliar=Cluster_DelF{end+1-i_h_l,1}*0;
                    end
                else
                    Vector_ZF1_auxiliar=Cluster_DelF{end+1-i_h_l,1}*0;
                end
                Vector_ZO_auxiliar=Cluster_DelO{end+1-i_h_l,1}.*(1-Matrix_O).*...
                    Matrix_O;
                
                %------------gradient of the cost function with respect to inputs to the LSTM units-------%
                Cluster_DelInputs{end+1-i_h_l,end+1-i_b_s}=Cluster_V{end+1-i_h_l,1}'*Vector_ZA_auxiliar+...
                    Cluster_V{end+1-i_h_l,2}'*Vector_ZI_auxiliar+...
                    Cluster_V{end+1-i_h_l,3}'*Vector_ZF_auxiliar+...
                    Cluster_V{end+1-i_h_l,4}'*Vector_ZO_auxiliar;  
                if ~isempty(number_M_parameters)
                   Cluster_DelInputs{end+1-i_h_l,end+1-i_b_s}=Cluster_DelInputs{end+1-i_h_l,end+1-i_b_s}+...
                       Cluster_V{end+1-i_h_l,5}'*Vector_ZF1_auxiliar;
                   if length(number_M_parameters)>1
                      Cluster_DelInputs{end+1-i_h_l,end+1-i_b_s}=Cluster_DelInputs{end+1-i_h_l,end+1-i_b_s}+...
                          Cluster_V{end+1-i_h_l,7}'*Vector_ZF2_auxiliar;
                   end
                end
                
                %--------------------------Gradient with respect to Momentum weights----------------------%
                if (~isempty(number_M_parameters))&&(sub_seq_length-i_b_s>0)
                    Matrix_WM1_M=Cluster_V{end+1-i_h_l,6};
                    if length(number_M_parameters)>1
                        Matrix_WM2_M=Cluster_V{end+1-i_h_l,8};
                    end

                    if (~isempty(number_M_parameters))&&(sub_seq_length-i_b_s>0)
                        Matrix_DelW1(rows_i_h_l_del,:)=((sum(Vector_ZWM1_auxiliar.*Matrix_WM1_auxiliar,2).*...
                            (-sign(Matrix_WM1_M)))+Vector_ZWM1_auxiliar)./den_WM1_auxiliar;
                    end
                    if (length(number_M_parameters)>1)&&(sub_seq_length-i_b_s>number_M_parameters(1))
                        Matrix_DelW2(rows_i_h_l_del,:)=((sum(Vector_ZWM2_auxiliar.*Matrix_WM2_auxiliar,2).*...
                            (-sign(Matrix_WM2_M)))+Vector_ZWM2_auxiliar)./den_WM2_auxiliar;
                    end 
                end
               
                %-----------------------------------------------------------------------------------------%
                
                %--------------------Gradient with respect to LSTM weights matrices ----------------------%
                
                Cluster_incV{end+1-i_h_l,1}=Cluster_incV{end+1-i_h_l,1}+Vector_ZA_auxiliar*Vector_LSTM_input';
                Cluster_incV{end+1-i_h_l,2}=Cluster_incV{end+1-i_h_l,2}+Vector_ZI_auxiliar*Vector_LSTM_input';
                Cluster_incV{end+1-i_h_l,3}=Cluster_incV{end+1-i_h_l,3}+Vector_ZF_auxiliar*Vector_LSTM_input';
                Cluster_incV{end+1-i_h_l,4}=Cluster_incV{end+1-i_h_l,4}+Vector_ZO_auxiliar*Vector_LSTM_input';
                if (~isempty(number_M_parameters))&&(sub_seq_length-i_b_s>0)
                    Cluster_incV{end+1-i_h_l,5}=Cluster_incV{end+1-i_h_l,5}+Vector_ZF1_auxiliar*Vector_LSTM_input';
                    Cluster_incV{end+1-i_h_l,6}=Cluster_incV{end+1-i_h_l,6}+Matrix_DelW1(rows_i_h_l_del,:);
                    
                end
                if (length(number_M_parameters)>1)&&(sub_seq_length-i_b_s>number_M_parameters(1))
                    Cluster_incV{end+1-i_h_l,7}=Cluster_incV{end+1-i_h_l,7}+Vector_ZF2_auxiliar*Vector_LSTM_input';
                    Cluster_incV{end+1-i_h_l,8}=Cluster_incV{end+1-i_h_l,8}+Matrix_DelW2(rows_i_h_l_del,:);
                end
                %-----------------------------------------------------------------------------------------%
            end

            Scaling_DelC_prev(end+1-i_h_l,i_sub_seq+1)=...
                Scaling_DelC_prev_aux(end+1-i_h_l,i_sub_seq+1);
            if (mod(i_t_i,10)==0)&&(i_sub_seq+1==ceil(size(Xn,1)/sub_seq_length))
                if mean(Scaling_DelC_prev,'all')>Scaling_DelC_th
                    %                 'Scaling_DelC_prev:'
                    [mean(Scaling_DelC_prev,'all'),max(Scaling_DelC_prev,[],'all')]
                end
            end

            for i_aux1=1:4+2*length(number_M_parameters)
                Cluster_incV_momentum{end+1-i_h_l,i_aux1}=(1-betha1)*Cluster_incV{end+1-i_h_l,i_aux1}+betha1*Cluster_incV_momentum{end+1-i_h_l,i_aux1};
                Cluster_incV_momentumsqr{end+1-i_h_l,i_aux1}=(1-betha2)*Cluster_incV{end+1-i_h_l,i_aux1}.^2+betha2*abs(Cluster_incV_momentumsqr{end+1-i_h_l,i_aux1});
                
                aux_Momentum_V=(Cluster_incV_momentum{end+1-i_h_l,i_aux1}/(1-betha1_acum))./...
                    sqrt(Cluster_incV_momentumsqr{end+1-i_h_l,i_aux1}/(1-betha2_acum)+10^-8);
                %------------------------------Saturation of the gradient---------------------------------%
                if Flag_Gradient_weigth_saturation==0
                    aux_L2=Cluster_V{end+1-i_h_l,i_aux1};
                    if (i_aux1<5)
                        Cluster_V{end+1-i_h_l,i_aux1}=aux_L2+alpha(end+1-i_h_l)*aux_Momentum_V;
                    else
                        aux_L2=aux_L2*(1-lambda*alpha(end+1-i_h_l));
                        Cluster_V{end+1-i_h_l,i_aux1}=aux_L2+alpha(end+1-i_h_l)*aux_Momentum_V;
                    end
                elseif Flag_Gradient_weigth_saturation==7
                    aux_L2=Cluster_V{end+1-i_h_l,i_aux1};
                    index_aux_L2=[1:size(Xn,2),size(Xn,2)+2:size(aux_L2,2)];
                    aux_L2(:,index_aux_L2)=aux_L2(:,index_aux_L2)*(1-lambda*alpha(end+1-i_h_l));
                    Cluster_V{end+1-i_h_l,i_aux1}=aux_L2+alpha(end+1-i_h_l)*...
                        Gradient_saturation(Gradient_threshold,aux_Momentum_V);
                end
                %-----------------------------------------------------------------------------------------%
            end
%             end
            %---------------------------------------------------------------------------------------------%
        end

        for i_h_l=1:number_hidden_layers
            if (~isempty(number_M_parameters))
                Cluster_V{end+1-i_h_l,6}=Cluster_V{end+1-i_h_l,6}./...
                    sum(abs(Cluster_V{end+1-i_h_l,6}),2);
                if (length(number_M_parameters)>1)
                    Cluster_V{end+1-i_h_l,8}=Cluster_V{end+1-i_h_l,8}./...
                        sum(abs(Cluster_V{end+1-i_h_l,8}),2);
                end
            end
        end

        aux_momentum_Vy=(Cluster_incVy_momentum{1,1}/(1-betha1_acum))./...
            sqrt(Cluster_incVy_momentumsqr{1,1}/(1-betha2_acum)+10^-8);
        
        %------------------------------Saturation of the gradient---------------------------------%
        if Flag_Gradient_weigth_saturation==0
            aux_L2=Cluster_Vy{1,1};
            aux_L2(:,1:end-1)=aux_L2(:,1:end-1)*(1-lambda*alpha(end+1-i_h_l));
            Cluster_Vy{1,1}=aux_L2+alpha(end+1-i_h_l)*aux_momentum_Vy;
        else
            aux_L2=Cluster_Vy{1,1};
            aux_L2(:,1:end-1)=aux_L2(:,1:end-1)*(1-lambda*alpha(end+1-i_h_l));
            Cluster_Vy{1,1}=aux_L2+alpha(end+1-i_h_l)*Gradient_saturation(Gradient_threshold,aux_momentum_Vy);
        end
        %-----------------------------------------------------------------------------------------%
        
        
        for i_h_l=1:number_hidden_layers
            for i=1:4+2*length(number_M_parameters)
                Cluster_incV{i_h_l,i}=Cluster_incV{i_h_l,i}*0;
            end
            Cluster_incVy{1,1}=Cluster_incVy{1,1}*0;
        end
        
        cont_sub_seq=cont_sub_seq+Step_window;
        i_sub_seq=i_sub_seq+1;
    end
    f_mean_max = Averega_Forgetting_Factor_v2(Cluster_F1_temporal,Cluster_F2_temporal,0.05,1);
    
    for i_aux1=1:4+2*length(number_M_parameters)
        for i_aux2=1:number_hidden_layers
            Cluster_incV_momentum{i_aux2,i_aux1}=Cluster_incV_momentum{i_aux2,i_aux1};
        end
    end
    Cluster_incVy_momentum{1,1}=Cluster_incVy_momentum{1,1};
    
    if Flag_val_enable==1
        for i_h_ia=1:number_hidden_layers
            Array_w_epoch(:,i_t_i,i_h_ia)=...
                mean(f_mean_max(:,1,i_h_ia).*abs(Cluster_V{1,6})./sum(abs(Cluster_V{1,6}),2),1);
            if length(number_M_parameters)>1
                Array_w_epoch2(:,i_t_i,i_h_ia)=[mean(f_mean_max(:,1,i_h_ia)),...
                    mean(f_mean_max(:,2,i_h_ia).*abs(Cluster_V{1,8})./sum(abs(Cluster_V{1,8}),2),1)];
            end
        end
        [~,Matrix_error_train,C_train_final,M1_train_final,H_train_final]=GI_LSTM_evaluation(X_train,Y_train,RX,RY,Cluster_V,Cluster_Vy,number_hidden_units,number_hidden_layers,0,Columns_replaced,0,[],[],[],number_M_parameters,Flag_MA);
        [Ye_val,Matrix_error_val,C_val_final,M1_val_final,H_val_final]=GI_LSTM_evaluation(X_val,Y_val,RX,RY,Cluster_V,Cluster_Vy,number_hidden_units,number_hidden_layers,Flag_output_feedback_validation,Columns_replaced,1,C_train_final,M1_train_final,H_train_final,number_M_parameters,Flag_MA);
        [~,Matrix_error_testing,~,~,~]=                                GI_LSTM_evaluation(X_testing,Y_testing,RX,RY,Cluster_V,Cluster_Vy,number_hidden_units,number_hidden_layers,Flag_output_feedback_testing,Columns_replaced,1,C_val_final,M1_val_final,H_val_final,number_M_parameters,Flag_MA);
        error_train=norm(Matrix_error_train)/sqrt(length(Matrix_error_train));
        error_val=norm(Matrix_error_val)/sqrt(length(Matrix_error_val));
        error_test=norm(Matrix_error_testing)/sqrt(length(Matrix_error_testing));
        if mod(i_t_i,10)==0
            
            if plot_heat_on==1
                figure(4)
                h1=heatmap(Array_w_epoch(:,1:i_t_i)./...
                    max(Array_w_epoch(:,1:i_t_i),[],1),'CellLabelColor','none');
                h1.GridVisible=0;
                XLabels = 1:i_t_i;
                CustomXLabels = string(XLabels);
                CustomXLabels(:) = " ";
                h1.XDisplayLabels = CustomXLabels;
                if length(number_M_parameters)>1
                    figure(5)
                    h2=heatmap(Array_w_epoch2(:,1:i_t_i)./...
                        max(Array_w_epoch2(:,1:i_t_i),[],1),'CellLabelColor','none');
                    h2.GridVisible=0;
                    h2.XDisplayLabels = CustomXLabels;
                end
                pause(0.1);
            end

            if (plot_on==1)
                figure(1);
                plot(Y_val(:,1));
                hold on
                plot(Ye_val(:,1))
                hold off
                if (~isempty(number_M_parameters))
                    figure(2)
                    plot(mean(f_mean_max(:,1,1).*abs(Cluster_V{1,6})./sum(abs(Cluster_V{1,6}),2),1));
                    if length(number_M_parameters)>1
                        figure(3)
                        plot([mean(f_mean_max(:,1,1),1),...
                            mean(f_mean_max(:,2,1).*abs(Cluster_V{1,8})./sum(abs(Cluster_V{1,8}),2),1)]);
                    end
                end
                pause(0.1)
            end
            disp(['Train_error = ', num2str(error_train),',Val_error = ', num2str(error_val), ',Test_error = ', num2str(error_test)]);
        end        
        if (error_val_old>=error_val)&&(error_train_old>error_train)
            error_train_old=error_train;
            error_val_old=error_val;
            cont_val_fails=0*max(cont_val_fails-1,0);
            Cluster_V_val=Cluster_V;
            Cluster_Vy_val=Cluster_Vy;
            Iterations_optimal(i_Meta,1)=i_t_i;
        else
            cont_val_fails=cont_val_fails+1;
        end     
        
        if cont_val_fails>number_val_fails
            'Maximum semi-consecutive fails reached'
            Cluster_V=Cluster_V_val;
            Cluster_Vy=Cluster_Vy_val;
            Iterations_optimal(i_Meta,1)=i_t_i;
            break;
        end
    else
        Cluster_V_val=Cluster_V;
        Cluster_Vy_val=Cluster_Vy;
        Iterations_optimal(i_Meta,1)=i_t_i;
    end

    %///////////////////////////////////////////////////////////////////////////////////////////////////////%
end

Cluster_V=Cluster_V_val;
Cluster_Vy=Cluster_Vy_val;

[Ye_train,~,C_train_final,M1_train_final,H_train_final]=GI_LSTM_evaluation(X_train,Y_train,RX,RY,Cluster_V_val,Cluster_Vy_val,number_hidden_units,number_hidden_layers,0,Columns_replaced,0,[],[],[],number_M_parameters,Flag_MA);
[Ye_val,~,C_val_final,M1_val_final,H_val_final]=GI_LSTM_evaluation(X_val,Y_val,RX,RY,Cluster_V_val,Cluster_Vy_val,number_hidden_units,number_hidden_layers,0,Columns_replaced,1,C_train_final,M1_train_final,H_train_final,number_M_parameters,Flag_MA);
[Ye,~,~,~,~]=GI_LSTM_evaluation(X_testing,Y_testing,RX,RY,Cluster_V_val,Cluster_Vy_val,number_hidden_units,number_hidden_layers,Flag_output_feedback_testing,Columns_replaced,1,C_val_final,M1_val_final,H_val_final,number_M_parameters,Flag_MA);

E_Meta_train(i_Meta,1)=norm(Ye_train-Y_train)/sqrt(length(Y_train));
E_Meta_val(i_Meta,1)=norm(Ye_val-Y_val)/sqrt(length(Y_val));
E_Meta_testing(i_Meta,1)=norm(Ye-Y_testing)/sqrt(length(Y_testing));
Cluster_Ye{i_Meta,1}=Ye;
i_Meta
'done'
toc
end

E_Meta_neurons_train(kkkk)=mean(E_Meta_train)
std_Meta_neurons_train(kkkk)=std(E_Meta_val)
E_Meta_neurons_val(kkkk)=mean(E_Meta_val)
std_Meta_neurons_val(kkkk)=std(E_Meta_val)
E_Meta_neurons_testing(kkkk)=mean(E_Meta_testing)
std_Meta_neurons_testing(kkkk)=std(E_Meta_testing)
Iterations_meta_neurons(kkkk)=mean(Iterations_optimal)

end
end
