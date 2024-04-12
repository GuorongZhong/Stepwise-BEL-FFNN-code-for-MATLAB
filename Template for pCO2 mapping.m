%% Training

ctn=1:11;%biogeochemical province defined by SOM, if not defined, use ctn=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('opsz20230820.mat')

load('pCO2stx_zscore_20230608.mat') % standardization for each variable using zscore:  [Var_n,stxmu(n,i),stxsigma(n,i)]=zscore(Var_n); n:variable type, i:biogeochemical province

if ~exist('Xpredictor','var')
    load('Xpredictor_global.mat')
end
random_num=2:8:76;% seting different initial states
time=datestr(now);
time([3 7 15 18])='_';

load('SOCATv2023_20230710.mat')%SOCAT dataset as m*n matrix, each row is a variable type, each column is a SOCAT sample.

load('pCO2_P 20230627.mat')%pCO2 predictors selected by Stepwise BEL algorithm, as row number of variables in the SOCAT matrix, such as [2 34 1 13 ...]
Xparaindex=pCO2_P(1,:);
Xparaindex1=pCO2_P(2,:);

XX=[SOCAT(7,:);Sindex(3,:);SOCAT([1 2 2 3:6 15:30 32:78],:)];

setdemorandstream(pi);
randIndex=randperm(size(XX,2));
XX=XX(:,randIndex);
XX=XX(:,randIndex);

XX(3,:)=sind(XX(3,:));
XX(4,:)=sind(XX(4,:));
XX(5,:)=cosd(XX(5,:));

XX(34,XX(34,:)>-200&XX(end-2,:)<11)=nan;
XX(34,XX(34,:)>=0&XX(end-2,:)==11)=nan;
XX(34,:)=log10(-XX(34,:));

temp=mean(XX([1 8 9],:),1);
XX(:,isnan(temp))=[];

X=XX;

Y=X(1,:);
X(1,:)=[];
XY=[X;Y];
net1=cell(length(random_num),ctn(end),3);net2=net1;
disp('current step: training')

parfor_progress(time,length(ctn));%function for showing progress bar from Jeremy Scheff - jdscheff@gmail.com - http://www.jeremyscheff.com/
%(if not download, delete the parfor_progress content)

for i=ctn
    lo=(sum(XY(end-3:end-1,:)==i,1)>0);
    trainsamp=XY(:,lo==1);
    
    [s2,~]=size(trainsamp);
    [trainsamp(end,:),stxmu(end,i),stxsigma(end,i)]=zscore(trainsamp(end,:));

    for m=1:s2-3
        trainsamp(m,:)=(trainsamp(m,:)-stxmu(m,i))./stxsigma(m,i);
    end
    ptrainsamp=trainsamp([Xparaindex{1,i} end],:);
    temp=mean(ptrainsamp,1);
    ptrainsamp(:,isnan(temp))=[];
    superpara=ones(1,opsz(1,i)).*10;%set FFNN size as [10 10 ...]
    for nn=1:length(random_num)
        setdemorandstream(random_num(nn));
        constantnet=feedforwardnet(superpara);
        constantnet.trainParam.showWindow = 0;
        temp=constantnet;
        setdemorandstream(random_num(nn));
        temp=train(temp,ptrainsamp(1:end-1,:),ptrainsamp(end,:));
        net1{nn,i,1}=temp;
        
        temp2=temp(ptrainsamp(1:end-1,:));
        setdemorandstream(random_num(nn));
        constantnet=feedforwardnet(superpara);
        constantnet.trainParam.showWindow = 0;
        temp=constantnet;
        setdemorandstream(random_num(nn));
        temp=train(temp,[ptrainsamp(1:end-1,:);temp2],ptrainsamp(end,:));
        net1{nn,i,2}=temp;
        
        temp3=temp([ptrainsamp(1:end-1,:);temp2]);
        setdemorandstream(random_num(nn));
        constantnet=feedforwardnet(superpara);
        constantnet.trainParam.showWindow = 0;
        temp=constantnet;
        setdemorandstream(random_num(nn));
        net1{nn,i,3}=train(temp,[ptrainsamp(1:end-1,:);temp2;temp3],ptrainsamp(end,:));
    end
    if ~isequal(Xparaindex{1,i},Xparaindex1{1,i})
        ptrainsamp=trainsamp([Xparaindex1{1,i} end],:);
        temp=mean(ptrainsamp,1);
        ptrainsamp(:,isnan(temp))=[];
        superpara=ones(1,opsz(1,i)).*10;%set FFNN size as [10 10 ...]
        
        for nn=1:length(random_num)
            setdemorandstream(random_num(nn));
            constantnet=feedforwardnet(superpara);
            constantnet.trainParam.showWindow = 0;
            temp=constantnet;
            setdemorandstream(random_num(nn));
            temp=train(temp,ptrainsamp(1:end-1,:),ptrainsamp(end,:));
            net2{nn,i,1}=temp;
            
            temp2=temp(ptrainsamp(1:end-1,:));
            setdemorandstream(random_num(nn));
            constantnet=feedforwardnet(superpara);
            constantnet.trainParam.showWindow = 0;
            temp=constantnet;
            setdemorandstream(random_num(nn));
            temp=train(temp,[ptrainsamp(1:end-1,:);temp2],ptrainsamp(end,:));
            net2{nn,i,2}=temp;
            
            temp3=temp([ptrainsamp(1:end-1,:);temp2]);
            setdemorandstream(random_num(nn));
            constantnet=feedforwardnet(superpara);
            constantnet.trainParam.showWindow = 0;
            temp=constantnet;
            setdemorandstream(random_num(nn));
            net2{nn,i,3}=train(temp,[ptrainsamp(1:end-1,:);temp2;temp3],ptrainsamp(end,:));
        end
    else
        net2(:,i,:)=net1(:,i,:);
    end
    parfor_progress(time);
end
disp('training completed')
if isfile([time,'.txt'])
    delete([time,'.txt'])
end

% mapping part
ctn=1:11;

load('Xpredictor_global.mat')% Predictor producct as a m*64800*348 matrix, m is total number of predictors in the SOCAT matrix, 348 is the total number of months from 199201 to 202012
% for each predictor as a 180*360*348 martix,Predictor(n,:,time)=reshape(Var_n(:,:,time),[1 64800]);

random_num=2:8:76;
time=datestr(now);
time([3 7 15 18])='_';

disp('current step: interpolating')


pCO2=zeros(180,360,348).*nan;
parfor_progress(time,348);
for ct=1:348
    pXparaindex=Xparaindex;
    pXparaindex1=Xparaindex1;
    nnet1=net1;
    nnet2=net2;
    pstxmu=stxmu;
    pstxsigma=stxsigma;
    month=rem(ct,12);
    if month==0
        month=12;
    end
    X=Xpredictor(:,:,ct);
    
    for m=1:length(X(:,1))-1
        for nn=ctn
             X(m,X(end,:)==nn)=(X(m,X(end,:)==nn)-pstxmu(m,nn))./pstxsigma(m,nn);
        end
    end
    
    X_grid=X(1,:);
    temp3=zeros(1,64800).*nan;
    
    [~,j]=find(~isnan(X_grid)&~isnan(X(end-1,:)));
    if ~isempty(j)
        [~,s1]=size(j);
        temp=X(:,j);
        temp2=zeros(length(random_num),s1).*nan;
        for i=ctn
            [~,m]=find(temp(end,:)==i);
            dtemp=temp(1:end-1,m);
            if ~isempty(m)
                dtemp=dtemp(pXparaindex{1,i},:);
                for nn=1:length(random_num)
                    net=nnet1{nn,i,1};
                    tempyy1=net(dtemp);
                    net=nnet1{nn,i,2};
                    tempyy2=net([dtemp;tempyy1]);
                    net=nnet1{nn,i,3};
                    temp2(nn,m)=net([dtemp;tempyy1;tempyy2]).*pstxsigma(end,i)+pstxmu(end,i);
                end
            end
        end
        temp3(:,j)=nanmean(temp2,1);
    end
    
    [~,j]=find(~isnan(X_grid)&isnan(X(end-1,:)));
    if ~isempty(j)
        [~,s1]=size(j);
        temp=X(:,j);
        temp2=zeros(length(random_num),s1).*nan;
        for i=ctn
            [~,m]=find(temp(end,:)==i);
            dtemp=temp(1:end-1,m);
            if ~isempty(m)
                dtemp=dtemp(pXparaindex1{1,i},:);
                for nn=1:length(random_num)
                    net=nnet2{nn,i,1};
                    tempyy1=net(dtemp);
                    net=nnet2{nn,i,2};
                    tempyy2=net([dtemp;tempyy1]);
                    net=nnet2{nn,i,3};
                    temp2(nn,m)=net([dtemp;tempyy1;tempyy2]).*pstxsigma(end,i)+pstxmu(end,i);
                end
            end
        end
        temp3(:,j)=nanmean(temp2,1);
    end
    

    pCO2(:,:,ct)=reshape(temp3,[180 360]);
    parfor_progress(time);
end

save('pCO2.mat','pCO2','net1','net2')

if isfile([time,'.txt'])
    delete([time,'.txt'])
end

%% winter correction
ctn=10:11;%biogeochemical province defined by SOM, if not defined, use ctn=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('opsz20230820.mat')
load('pCO2_P 20230627.mat')
Xparaindex=pCO2_P(1,:);
Xparaindex1=pCO2_P(3,:);

load('pCO2stx_zscore_20230608.mat')
load('pco2class11_0623.mat')
load('SOCATv2023_20230710.mat')

random_num=2:8:76;
time=datestr(now);
time([3 7 15 18])='_';

for iteration=1:6

XX=[SOCAT(7,:);Sindex(3,:);SOCAT([1 2 2 3:6 15:30 32:78],:)];

setdemorandstream(pi);
randIndex=randperm(size(XX,2));
XX=XX(:,randIndex);
XX=XX(:,randIndex);

switch iteration
    case 1 % Traing with samples from May to September
        XX(:,XX(7,:)<5)=[];
        XX(:,XX(7,:)>9)=[];
        info='Traing with samples from May to September';
    case 2 % Traing with samples from April to September
        XX(:,XX(7,:)<4)=[];
        XX(:,XX(7,:)>9)=[];
        info='Traing with samples from April to September';
    case 3 % Traing with samples from May to October
        XX(:,XX(7,:)<5)=[];
        XX(:,XX(7,:)>10)=[];
        info='Traing with samples from May to October';
    case 4 % Traing with samples from April to October, preferred
        XX(:,XX(7,:)<4)=[];
        XX(:,XX(7,:)>10)=[];
        info='Traing with samples from April to October, preferred';
    case 5 % Traing with samples from March to November
        XX(:,XX(7,:)<3)=[];
        XX(:,XX(7,:)>11)=[];
        info='Traing with samples from March to November';
    case 6 % Traing with samples of all month
        info='Traing with samples of all month';
end

XX(3,:)=sind(XX(3,:));
XX(4,:)=sind(XX(4,:));
XX(5,:)=cosd(XX(5,:));

XX(34,XX(34,:)>-200&XX(end-2,:)<11)=nan;
XX(34,XX(34,:)>=0&XX(end-2,:)==11)=nan;
XX(34,:)=log10(-XX(34,:));

temp=mean(XX([1 8 9],:),1);
XX(:,isnan(temp))=[];

X=XX;

Y=X(1,:);
X(1,:)=[];
XY=[X;Y];
net1=cell(length(random_num),ctn(end),3);net2=net1;
disp('current step: training')
% %%
parfor_progress(time,length(ctn));
for i=ctn%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    lo=(sum(XY(end-3:end-1,:)==i,1)>0);
    trainsamp=XY(:,lo==1);
    
    [s2,~]=size(trainsamp);
    [trainsamp(end,:),stxmu(end,i),stxsigma(end,i)]=zscore(trainsamp(end,:));


    for m=1:s2-3
        trainsamp(m,:)=(trainsamp(m,:)-stxmu(m,i))./stxsigma(m,i);
    end
        ptrainsamp=trainsamp([Xparaindex1{1,i} end],:);
        temp=mean(ptrainsamp,1);
        ptrainsamp(:,isnan(temp))=[];
        superpara=ones(1,opsz(2,i)).*10;
        
        for nn=1:length(random_num)
            setdemorandstream(random_num(nn));
        constantnet=feedforwardnet(superpara);
        constantnet.trainParam.showWindow = 0;
        temp=constantnet;
        setdemorandstream(random_num(nn));
        temp=train(temp,ptrainsamp(1:end-1,:),ptrainsamp(end,:));
        net2{nn,i,1}=temp;
        
        temp2=temp(ptrainsamp(1:end-1,:));
        setdemorandstream(random_num(nn));
        constantnet=feedforwardnet(superpara);
        constantnet.trainParam.showWindow = 0;
        temp=constantnet;
        setdemorandstream(random_num(nn));
        temp=train(temp,[ptrainsamp(1:end-1,:);temp2],ptrainsamp(end,:));
        net2{nn,i,2}=temp;
        
        temp3=temp([ptrainsamp(1:end-1,:);temp2]);
        setdemorandstream(random_num(nn));
        constantnet=feedforwardnet(superpara);
        constantnet.trainParam.showWindow = 0;
        temp=constantnet;
        setdemorandstream(random_num(nn));
        net2{nn,i,3}=train(temp,[ptrainsamp(1:end-1,:);temp2;temp3],ptrainsamp(end,:));
        end

    parfor_progress(time);
end
disp('training completed')
if isfile([time,'.txt'])
    delete([time,'.txt'])
end


pCO2cr=zeros(180,360,348).*nan;
parfor_progress(time,348);
for ct=1:348
    month=rem(ct,12);
    if month==0
        month=12;
    end
    if month<5||month>9
        continue
    end
    pXparaindex=Xparaindex;
    pXparaindex1=Xparaindex1;
    nnet1=net1;
    nnet2=net2;
    pstxmu=stxmu;
    pstxsigma=stxsigma;
    
    X=Xpredictor(:,:,ct);
    
    for m=1:length(X(:,1))-1
        for nn=ctn
             X(m,X(end,:)==nn)=(X(m,X(end,:)==nn)-pstxmu(m,nn))./pstxsigma(m,nn);
        end
    end
    
    X_grid=X(1,:);
    temp3=zeros(1,64800).*nan; 
    
    [~,j]=find(~isnan(X_grid));
    if ~isempty(j)
        [~,s1]=size(j);
        temp=X(:,j);
        temp2=zeros(length(random_num),s1).*nan;
        for i=ctn
            [~,m]=find(temp(end,:)==i);
            dtemp=temp(1:end-1,m);
            if ~isempty(m)
                dtemp=dtemp(pXparaindex1{1,i},:);
                for nn=1:length(random_num)
                    net=nnet2{nn,i,1};
                    tempyy1=net(dtemp);
                    net=nnet2{nn,i,2};
                    tempyy2=net([dtemp;tempyy1]);
                    net=nnet2{nn,i,3};
                    temp2(nn,m)=net([dtemp;tempyy1;tempyy2]).*pstxsigma(end,i)+pstxmu(end,i);
                end
            end
        end
        temp3(:,j)=nanmean(temp2,1);
    end
    
    pCO2cr(:,:,ct)=reshape(temp3,[180 360]);
    parfor_progress(time);
end
save(['pCO2_corrected_',num2str(iteration),'.mat'],'pCO2cr','net2')
if isfile([time,'.txt'])
    delete([time,'.txt'])
end
end
load('pCO2_corrected_4.mat')
pCO2(~isnan(pCO2cr))=pCO2cr(~isnan(pCO2cr));