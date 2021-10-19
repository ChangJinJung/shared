close all; clear all;

%% Load time-series of Facial action units extracted from iMotion software

load('/mnt/Depression/facialexpr/emodata200211.mat');

emoout_HC=emoout(1:84,:,:);
emoout_MDD=emoout(85:end,:,:);

fnlist_HC=fnlist(1:84)';
fnlist_MDD=fnlist(85:end)';


%% Apply loganism to raw action unit values and merge 17 units

MergedUnits_MDD=[];

for k=1:size(emoout_MDD_taskAll,1)
    
    AU_MDD=zeros(size(emoout_MDD_taskAll,3),17); 
    
    for p=1:17
        AU(:,p)=emoout_MDD_taskAll(k,p,:); %% use 17 action units for PCA
    end
    
    MergedUnits_MDD=[MergedUnits_MDD; log(1+AU_MDD)]; %% Apply loganism 
   
end

MergedUnits_HC=[];

for k=1:size(emoout_HC_taskAll,1)
    
    AU_HC=zeros(size(emoout_HC_taskAll,3),17); 
    
    for p=1:17
        AU_HC(:,p)=emoout_HC_taskAll(k,p,:); %% use 17 action units for PCA
    end
    
    MergedUnits_HC=[MergedUnits_HC; log(1+AU_HC)]; %% Apply loganism 
end

MergedUnits=[MergedUnits_MDD; MergedUnits_HC];

%% P erform Principal component analysis
x = MergedUnits;

[coeff,score,latent,tsquared,explained,mu] = pca(x); %% use pca function of MATLAB for Principal component analysis 

PCscores=(x-repmat(mu,size(x,1),1))*inv(coeff'); %% computes PC scores; 



