function [Ye,Matrix_error,C_final,M1_final,H_final]=GLSTM_evaluation_M(X_testing,Y_testing,RX,RY,Cluster_V,Cluster_Vy,number_hidden_units,number_hidden_layers,Flag_output_feedback,replacing_columns,Flag_initial_conditions,C_initial,M1_initial,H_initial,number_M_parameters,Flag_MA)

Xn=zeros(size(X_testing))';
Cluster_H=cell(number_hidden_layers,2);%current and previous values of the hidden neurons
Matrix_layers_C=zeros(sum(number_hidden_units),size(Xn,2));%cell values (internal state of LSTM)
if ~isempty(number_M_parameters)
    Matrix_layers_M1=zeros(sum(number_hidden_units),size(Xn,2));
    if length(number_M_parameters)>1
        Matrix_layers_M2=zeros(sum(number_hidden_units),size(Xn,2));
    end
end

for i=1:size(X_testing,2)
    Xn(i,:)=(X_testing(:,i)-RX(i,1))/RX(i,2);
end
Yn=zeros(size(Y_testing));


row_CM=zeros(number_hidden_layers,2);
for i_CM=1:number_hidden_layers
    if i_CM==1
        row_CM(:,2)=number_hidden_units';
        row_CM(1,1)=1;
    else
        row_CM(i_CM,2)=row_CM(i_CM-1,2)+row_CM(i_CM,2);
    end
end
row_CM(2:end,1)=1+row_CM(1:end-1,2);

den_WM1_auxiliar=cell(length(number_hidden_units),1);
% Matrix_WM1_auxiliar=cell(size(Cluster_V{:,6}));

for i_b_s=1:size(Xn,2)
    
    for i_h_l=1:number_hidden_layers

        %-----------------selection of the Input values to the LSTM units-------------------%
        if i_b_s>1
            Cluster_H{i_h_l,1}=Cluster_H{i_h_l,2};
        end
        
        if i_b_s==1
            if Flag_initial_conditions==1
                H_initial_Matrix=H_initial{i_h_l,1};
                if i_h_l==1
                    Vector_LSTM_input=[Xn(:,i_b_s);1;H_initial_Matrix];
                else
                    Vector_LSTM_input=[Cluster_H{i_h_l-1,2};1;H_initial_Matrix];
                end
            else
                if i_h_l==1
                    Vector_LSTM_input=[Xn(:,i_b_s);1;zeros(number_hidden_units(1),1)];
                else
                    Vector_LSTM_input=[Cluster_H{i_h_l-1,2};1;zeros(number_hidden_units(i_h_l),1)];
                end
            end
        else
            if i_h_l==1
                if Flag_output_feedback==1
                    Xn_modified=X_modification_output_feedback_1c_eco(Xn(:,i_b_s),...
                        Yn(i_b_s-1,:)',replacing_columns);
                else
                    Xn_modified=Xn(:,i_b_s);
                end
                if Flag_MA==1
                    Xn_modified=X_modification_output_MA_1c_eco(Xn(:,i_b_s),...
                        Flag_MA,Y_testing(i_b_s-1,:)'-Yn(i_b_s-1,:)');
                end
                Vector_LSTM_input=[Xn_modified;1;Cluster_H{i_h_l,1}];
            else
                Vector_LSTM_input=[Cluster_H{i_h_l-1,2};1;Cluster_H{i_h_l,1}];
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
            if i_b_s==1
                den_WM1_auxiliar{i_h_l,1}=sum(abs(Cluster_V{i_h_l,6}),2);
                Matrix_WM1_auxiliar{i_h_l,1}=Cluster_V{i_h_l,6}./den_WM1_auxiliar{i_h_l,1};
            end
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
            Matrix_WM1_auxiliar{i_h_l,1}=Cluster_V{i_h_l,3}*0;
            Matrix_WM2_auxiliar=Cluster_V{i_h_l,3}*0;
        end
        %-----------------------------------------------------------------%
        
        %-------------------Non-linear transformations--------------------%        
%         Matrix_LtH_auxiliar=Matrix_Va_auxiliar*Vector_LSTM_input;
        Matrix_A=tanh(Matrix_Va_auxiliar*Vector_LSTM_input);
%         Matrix_LtH_auxiliar=Matrix_Vi_auxiliar*Vector_LSTM_input;
        Matrix_I=0.5*tanh(Matrix_Vi_auxiliar*Vector_LSTM_input/2)+0.5;
%         Matrix_LtH_auxiliar=Matrix_Vf_auxiliar*Vector_LSTM_input;
        Matrix_F=0.5*tanh(Matrix_Vf_auxiliar*Vector_LSTM_input/2)+0.5;
%         Matrix_LtH_auxiliar=Matrix_Vf1_auxiliar*Vector_LSTM_input;
        if ~isempty(number_M_parameters)
            Matrix_F1=0.5*tanh(Matrix_Vf1_auxiliar*Vector_LSTM_input/2)+0.5;
        else
            Matrix_F1=Matrix_I*0;
        end
%         Matrix_LtH_auxiliar=Matrix_Vf2_auxiliar*Vector_LSTM_input;
        if length(number_M_parameters)>1
            Matrix_F2=0.5*tanh(Matrix_Vf2_auxiliar*Vector_LSTM_input/2)+0.5;
        else
            Matrix_F2=Matrix_I*0;
        end
%         Matrix_LtH_auxiliar=Matrix_Vo_auxiliar*Vector_LSTM_input;
        Matrix_O=0.5*tanh(Matrix_Vo_auxiliar*Vector_LSTM_input/2)+0.5;
        %-----------------------------------------------------------------%
        
        if Flag_initial_conditions==1
            if (~isempty(number_M_parameters))
                if (length(number_M_parameters)>1)                    
                    c_f=Matrix_F1+Matrix_F2+10^-10;
                    a_f=Matrix_F1./c_f;
                    b_f=Matrix_F2./c_f;
                    
                    if (i_b_s>number_M_parameters(1)*number_M_parameters(2))
                        M2_index=i_b_s-number_M_parameters(1):-number_M_parameters(1):i_b_s-number_M_parameters(1)*number_M_parameters(2);
                        Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),M2_index).*...
                            Matrix_WM2_auxiliar,2);
                        
                        Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-(1:number_M_parameters(1))).*Matrix_WM1_auxiliar{i_h_l,1},2);
                        
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Matrix_I.*Matrix_A+...
                            a_f.*Matrix_F1.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)+...
                            b_f.*Matrix_F2.*Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                    elseif (i_b_s>number_M_parameters(1))
                        M2_index=i_b_s-number_M_parameters(1):-number_M_parameters(1):1;
                        index_residual=number_M_parameters(1)-M2_index(end);
                        Combined_M1=[Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),M2_index),...
                            M1_initial(row_CM(i_h_l,1):row_CM(i_h_l,2),end-index_residual:-number_M_parameters(1):end-index_residual-number_M_parameters(1)*(number_M_parameters(2)-1-length(M2_index)))];
                        
                        Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(Combined_M1.*Matrix_WM2_auxiliar,2);
                        Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-(1:number_M_parameters(1))).*Matrix_WM1_auxiliar{i_h_l,1},2);
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Matrix_I.*Matrix_A+...
                            a_f.*Matrix_F1.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)+...
                            b_f.*Matrix_F2.*Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                    elseif(i_b_s==1)
                         Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                             sum(M1_initial(row_CM(i_h_l,1):row_CM(i_h_l,2),end+1-number_M_parameters(1):-number_M_parameters(1):end+1-number_M_parameters(1)*number_M_parameters(2)).*...
                             Matrix_WM2_auxiliar,2);
                         Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(C_initial(row_CM(i_h_l,1):row_CM(i_h_l,2),end+1-(1:number_M_parameters(1))).*...
                            Matrix_WM1_auxiliar{i_h_l,1},2);
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Matrix_I.*Matrix_A+...
                            a_f.*Matrix_F1.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)+...
                            b_f.*Matrix_F2.*Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                    else
                        Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(M1_initial(row_CM(i_h_l,1):row_CM(i_h_l,2),end+i_b_s-number_M_parameters(1):-number_M_parameters(1):end+i_b_s-number_M_parameters(1)*number_M_parameters(2)).*...
                            Matrix_WM2_auxiliar,2);
                        Combined_C=[Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),(i_b_s-1):-1:1),...
                            C_initial(row_CM(i_h_l,1):row_CM(i_h_l,2),end:-1:end-(number_M_parameters(1)-i_b_s))];
                        Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(Combined_C.*...
                            Matrix_WM1_auxiliar{i_h_l,1},2);
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Matrix_I.*Matrix_A+...
                            a_f.*Matrix_F1.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)+...
                            b_f.*Matrix_F2.*Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                    end
                else
                    if (i_b_s>number_M_parameters(1))
                        Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-(1:number_M_parameters(1))).*...
                            Matrix_WM1_auxiliar{i_h_l,1},2);
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Matrix_I.*Matrix_A+...
                            Matrix_F1.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                    elseif (i_b_s==1)
                        Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(C_initial(row_CM(i_h_l,1):row_CM(i_h_l,2),end+1-(1:number_M_parameters(1))).*...
                            Matrix_WM1_auxiliar{i_h_l,1},2);
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Matrix_I.*Matrix_A+...
                            Matrix_F1.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                    else
                        Combined_C=[Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),(i_b_s-1):-1:1),...
                            C_initial(row_CM(i_h_l,1):row_CM(i_h_l,2),end:-1:end-(number_M_parameters(1)-i_b_s))];
                        Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(Combined_C.*...
                            Matrix_WM1_auxiliar{i_h_l,1},2);
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Matrix_I.*Matrix_A+...
                            Matrix_F1.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                    end
                end
            else
                if i_b_s>1
                    Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                        Matrix_F.*Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-1)+...
                        Matrix_I.*Matrix_A;
                else
                    Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                        Matrix_F.*C_initial(row_CM(i_h_l,1):row_CM(i_h_l,2),end)+...
                        Matrix_I.*Matrix_A;
                end
            end
        else
            if i_b_s>1
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
                            sum(Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-(1:number_M_parameters(1))).*Matrix_WM1_auxiliar{i_h_l,1},2);
                        
                        c_f=Matrix_F1+Matrix_F2+10^-10;
                        a_f=Matrix_F1./c_f;
                        b_f=Matrix_F2./c_f;
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Matrix_I.*Matrix_A+...
                            a_f.*Matrix_F1.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)+...
                            b_f.*Matrix_F2.*Matrix_layers_M2(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                    else
                        Matrix_WM1_auxiliar2=Matrix_WM1_auxiliar{i_h_l,1};
                        Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            sum(Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-(1:min(number_M_parameters(1),i_b_s-1))).*...
                            Matrix_WM1_auxiliar2(:,1:min(number_M_parameters(1),i_b_s-1)),2);
                        Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                            Matrix_I.*Matrix_A+...
                            Matrix_F1.*Matrix_layers_M1(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s);
                    end
                else
                    Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=...
                        Matrix_F.*Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s-1)+...
                        Matrix_I.*Matrix_A;
                end
            else
                Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s)=Matrix_I.*Matrix_A;
            end
        end
        
        Cluster_H{i_h_l,2}=Matrix_O.*tanh(Matrix_layers_C(row_CM(i_h_l,1):row_CM(i_h_l,2),i_b_s));
    end
    
    Yn(i_b_s,:)=(Cluster_Vy{1,1}*[Cluster_H{end,2};1])';
    
end

if ~isempty(number_M_parameters)
    if (size(Matrix_layers_C,2)>=number_M_parameters(1))
        C_final=...
            Matrix_layers_C(:,end+1-number_M_parameters(1):end);
    else
        C_final=[C_initial(:,end+1-number_M_parameters(1)+size(Matrix_layers_C,2):end),...
            Matrix_layers_C];
    end
    if length(number_M_parameters)>1
        if (size(Matrix_layers_M1,2)>=number_M_parameters(1)*(number_M_parameters(2)+1))
            M1_final=Matrix_layers_M1(:,end+1-number_M_parameters(1)*(number_M_parameters(2)+1):end);
        else
            M1_final=[M1_initial(:,end+1-number_M_parameters(1)*(number_M_parameters(2)+1)+size(Matrix_layers_M1,2):end),...
                Matrix_layers_M1(:,:)];
        end
    else
        M1_final=[];
    end
else
    M1_final=[];
    C_final=Matrix_layers_C(:,end);
end
H_final=Cluster_H(:,2);

Ye=zeros(size(Yn));
Yn_testing=zeros(size(Yn));
for i=1:size(Y_testing,2)
    Ye(:,i)=Yn(:,i)*RY(i,2)+RY(i,1);
    Yn_testing(:,i)=(Y_testing(:,i)-RY(i,1))/RY(i,2);
end

Matrix_error=Ye-Y_testing;

end