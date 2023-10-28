function [mean_max_w_ffp] = Averega_Forgetting_Factor_v2(Cluster_F_temporal,Cluster_Fp_temporal,threshold_f,Flag_units)
Training_size=size(Cluster_F_temporal(1,:),2);
if Flag_units==1
    mean_max_w_ffp=zeros(size(Cluster_F_temporal{1,1},1),2,size(Cluster_F_temporal,1));
else
    mean_max_w_ffp=zeros(size(Cluster_F_temporal{1,1},1),4,size(Cluster_F_temporal,1));
end

for i_h_i=1:size(Cluster_F_temporal,1)
    Window_time_length=zeros(size(Cluster_F_temporal{i_h_i,1}));
    Window_time_length2=zeros(size(Cluster_F_temporal{i_h_i,1}));
    for i_f=1:Training_size
        fn=(Cluster_F_temporal{i_h_i,i_f}.^2)./(Cluster_F_temporal{i_h_i,i_f}+Cluster_Fp_temporal{i_h_i,i_f});
        Window_time_length(:,1,:)=Window_time_length(:,1)+log(fn);
        fpn=(Cluster_Fp_temporal{i_h_i,i_f}.^2)./(Cluster_F_temporal{i_h_i,i_f}+Cluster_Fp_temporal{i_h_i,i_f});
        Window_time_length2(:,1,:)=Window_time_length2(:,1)+log(fpn);
    end
    Window_time_length=Window_time_length/Training_size;
    Window_time_length2=Window_time_length2/Training_size;

    if Flag_units==1
        mean_w=exp(Window_time_length);
        mean_w2=exp(Window_time_length2);
        mean_max_w_ffp(:,:,i_h_i)=[mean_w,mean_w2];
    else
        mwtl=mean(Window_time_length,'all');
        maxwtl=mean(max(Window_time_length,[],1));
        mwt2=mean(Window_time_length2,'all');
        maxwt2=mean(max(Window_time_length2,[],1));

        mean_w=max(log(threshold_f)/mwtl,.1);
        max_w=min(max(log(threshold_f)/maxwtl,.1),inf);
        mean_w2=max(log(threshold_f)/mwt2,.1);
        max_w2=min(max(log(threshold_f)/maxwt2,.1),inf);

        mean_max_w_ffp(:,:,i_h_i)=[mean_w,max_w,mean_w2,max_w2];
    end
end

end