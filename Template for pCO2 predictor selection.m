%% stepffnn  random_num global 20230608
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CHL_type=1;%In regions that remote sensing data is not available, set CHL_type=0; if remote sensing data is available, set CHL_type=1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('local',12);
end
load('SOCATv2023_20230710.mat')
load('pCO2stx_zscore_20230608.mat')

X_chosinfo={'num of months',' Lat',' sLon',' cLon',' year',' month',' SST',' SSS','refTA','refDIC',' refDO',' refN',' refP',' Si',' SSTanom',' SSSanom',' MLD',' MLDanom',' SSH',' SSHanom','Wind speed',' SLP',' surfP',' Wvel_5m','Wvel_65m','Wvel_105m','Wvel_195m',' xCO_2',' xCO_2anom',' ONI','AOI','SOI','GLODEP',' CHL-a',' CHLanom'}';
X_chosinfo=[X_chosinfo;{'KD490','PAR','POCs','RRS412','RRS443','RRS469','RRS488','RRS531','RRS547','RRS555','RRS645','RRS667','RRS678','Ta412','Ta443','Ta469','Ta488','Ta531','Ta547','Ta555','Ta645','Ta667','Ta678','Tb412','Tb443','Tb469','Tb488','Tb531','Tb547','Tb555','Tb645','Tb667','Tb678'}'];
% put names of predictors in X_chosinfo as m*1 cells, m: total number of predictors

XX=[SOCAT(7,:);Sindex(3,:);SOCAT([1 2 2 3:6 15:30 32:78],:);rem(SOCAT(3,:),4)+1]; %rem(SOCAT(3,:),4)+1 is dividing groups of K-fold validation by years, hear SOCAT(3,:) is sample years

temp=XX(:,XX(7,:)>4&XX(7,:)<10);
XX=[XX temp temp];
%winter weighting correction,  use this only in the Southern Ocean south of 50S 

setdemorandstream(pi);
randIndex=randperm(size(XX,2));
XX=XX(:,randIndex);
XX=XX(:,randIndex);

XX(3,:)=sind(XX(3,:));
XX(4,:)=sind(XX(4,:));
XX(5,:)=cosd(XX(5,:));

XX(34,XX(34,:)>-200)=nan;
XX(34,:)=log10(-XX(34,:));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch CHL_type
    case 0
        runpath='...\withoutCHL'; %set save path
        if ~exist(runpath,'file')
            mkdir(runpath);
        end
        cd(runpath)
        disp('нчр╤блкь')
    case 1
        runpath='...\withCHL'; %set another save path
        if ~exist(runpath,'file')
            mkdir(runpath);
        end
        cd(runpath)
        disp('спр╤блкь')
        XX(:,XX(2,:)<128)=[];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp=mean(XX([1 8 9],:),1);
XX(:,isnan(temp))=[];
clear temp

K=XX(end,:);
XX(end,:)=[];

P=XX(end-2:end,:);% biogeochemical province number, if do not divide regions, set P=1 and for i=1 in Line 73
XX(end-2:end,:)=[];

time=datestr(now);
time([3 7 15 18])='_';

FFNNsize=[10 10 10 10];

Validation=4;

pCO2_P=cell(2,11);
for i=1:11 % biogeochemical province number, if do not divide regions,set for i=1 only

disp(['FFNN size = ',num2str(FFNNsize)])
disp(['Validation_Check = ',num2str(Validation)])
    lo=(sum(P==i,1)>0);
    X=XX(2:end,lo==1);
    Y=XX(1,lo==1);
    tempP=P(:,lo==1);
    tempP(1,tempP(1,:)~=i)=nan;
    for nn=1:length(X(:,1))
        X(nn,:)=(X(nn,:)-stxmu(nn,i))./stxsigma(nn,i);
    end
    
    if CHL_type==1
        tempXpara=stepwisebel(X,Y,tempP,K(:,lo==1),Validation,FFNNsize,X_chosinfo');
    else
        tempXpara=stepwisebel(X(1:end-35,:),Y,tempP,K(:,lo==1),Validation,FFNNsize,X_chosinfo(1:end-35,:)');
    end
    pCO2_P{1,i}=tempXpara{1,i};
    pCO2_P{2,i}=tempXpara{2,i};
    save(['temp_pCO2_P',time,'.mat'],'pCO2_P')
end
save(['pCO2_P.mat'],'pCO2_P')
