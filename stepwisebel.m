function Xparaindex=stepwisebel(X,Y,P,K,validation_check,FFNNsize,varargin)

%% stepwise BEL FFNN algorithm for finding optimall predictors, version 2023.2

%citation:
%Zhong, G., Li, X., Song, J., Qu, B., Wang, F., Wang, Y., Zhang, B., Sun, X., Zhang, W., Wang, Z., Ma, J., Yuan, H., and Duan, L.: 
%Reconstruction of global surface ocean pCO2 using region-specific predictors based on a stepwise FFNN regression algorithm, Biogeosciences, 19, 845¨C859, https://doi.org/10.5194/bg-19-845-2022, 2022.


%[Xparaindex] = stepwisebel(X,Y,P,K,validation_check,FFNNsize)
%   example: [Xparaindex] = stepwisebel(X,Y,1,K,1,[25 25])
%   example: [Xparaindex] = stepwisebel(X,Y,1,K,1,25)
%   example: [Xparaindex] = stepwisebel(X,Y,1,K,1,0)
%[Xparaindex] = stepwisebel(X,Y,P,K,validation_check,FFNNsize,X_info)
%   example:
%          [Xparaindex] = stepwisebel(X,Y,1,K,1,[25 25],{'SST','SSS',...})
%          [Xparaindex] = stepwisebel(X,Y,1,K,1,0,{'SST','SSS',...})
%neccessry inputs: X, Y, P, K, validation_check,FFNNsize
%optional inputs: X_info
%Xparaindex:a cell with the row number of selected predictors in X,
%For example:the output of [1 34 2 13 15] means the variable in the first,34th,.., and 15th rows of X was selected as predictors of Y

%input information:
% X: nx*mx matrix of nx input vectors of training samples£¬each row was a type of predictors
%   nx: total number of predictors
%   mx: total number of samples

% Y: 1*mx matrix of mx target outputs of training samples, each column was one training sample

% P: np*mx matrix of mx Biogeochemical Province numbers
%   np:if there is any sample belongs to muti provinces, put the number of provinces in different rows of P,otherwise, np=1;
%   if the samples were not divided by provinces, set P=1;
%   if matrix X contains only one province with its edge grids, set the province number of edge grids as nan;
%       for example:   [1   1   1 nan 1  1  nan;
%                       2  nan  3  1  2 nan  1]

% K: 1*mx matrix of mx k-fold validation labels, including numbers from 1 to k.k is the number of groups that the samples X and Y were divided, with recommended value of 4-5.
%   example: Set K=ceil(k.*rand(size(Y))); to randomly divided X and Y into k groups.

% validation_check: recommended value was 2 - 3.
%   1:algorithm complete immediately when the MAE increases in the next step
%   2:algorithm complete when the MAE increases continuously in the next two step

% FFNNsize: the number of hidden layers and the number of neurons in each hidden layer;
% the larger size of FFNN may taking much more time for runing this algorithm.
%   example: 25 represents a single hidden layer with 25 neurons
%            [50 50] represents two hidden layers with 50 neurons in each layer

% X_info: 1*nx cell of nx predictor names, set this term will make the output easier to understand

%output information:
%  the output information will be showed in real time as:

%-- Present time
%-- MAE   predictor_name
%-- ......
%-- MAE   predictor_name
%-- Validaton check completed
%-- connected after step 'MAE  predictor_name'
%-- MAE   predictor_name removed
%-- ......
%-- Province 1

%......

%-- Present time
%-- MAE   predictor_name
%-- ......
%-- MAE   predictor_name
%-- Validaton check completed
%-- connected after step 'MAE  predictor_name'
%-- MAE   predictor_name removed
%-- ......
%-- Province N

%(predictor_name is what you set in the term 'X_info',default value is the number of rows of 'X')

%If there is any trouble in running this script, contact zhongguorong@qdio.ac.cn

%%  default values setting
running_info=[];
switch nargin
    case 6
        X_chosinfo=num2cell(1:length(X(:,1)));
    case 7
        c=string(class(varargin));
        [~,m]=find(contains(c,"cell")==1);
        if ~isempty(m)
            X_chosinfo=varargin{m};
        else
            error('X_info format is not correct!')
        end
    otherwise
        disp(nargin)
        error('number of inputs is not correct!')
end

paratotalnum=length(X(:,1));
if length(X_chosinfo)~=paratotalnum
    error('length of X_info is not matched with X!')
end
if P==1
    ctn=1;
    P=ones(1,length(X(1,:)));
    ct=1;
else
    temp=unique(P(1,:));
    temp(isnan(temp))=[];
    ct=temp;
    ctn=max(temp,[],'all');
end


random_num=2:8:76;
average_num=length(random_num);
p_num=length(P(:,1));
K_num=unique(K);
K_num(isnan(K_num))=[];
K_num=length(K_num);
Xparaindex=cell(2,ctn);

E=zeros(ctn,paratotalnum.*2).*nan;
loopsort=ct;

    
if FFNNsize~=0    
    trainalgorithm='trainlm';
    constantnet=cell(4,length(random_num));
    for i=1:length(random_num)
    setdemorandstream(random_num(i));
    temp=feedforwardnet(FFNNsize,trainalgorithm);
    temp.trainParam.showWindow = 0;
    temp.trainParam.epochs=1000;
    constantnet{1,i}=temp;
    end
    constantnet(2:4,:)=[constantnet(1,:);constantnet(1,:);constantnet(1,:)];
        disp(['trainFc=',trainalgorithm])
else
    constantnet=[];
end

disp('current running province =')
disp(loopsort)
%% stepwise initialization
for Province_num=loopsort
    if isfile(['running_info_province_',num2str(Province_num),'.mat'])
        disp(['running_info.mat of current province already exist, province ',num2str(Province_num),' skiped. If not willing to skip, delete all existing running info files in current path before running'])
        continue;
    end
    running_info=[running_info;string(['Province ',num2str(Province_num)])]; %#ok<AGROW>
    lo=(sum(P==Province_num,1)>0);
    X_cons=Y(:,lo==1);
    X_chos=X(:,lo==1);
    X_P=P(:,lo==1);
    X_K=K(:,lo==1);

    para_num=paratotalnum;
    temp=mean(X_chos,1);
    X_cons(:,isnan(temp))=[];
    X_chos(:,isnan(temp))=[];
    X_P(:,isnan(temp))=[];
    X_K(:,isnan(temp))=[];
    
    X_chos(:,isnan(X_cons))=[];
    X_P(:,isnan(X_cons))=[];
    X_K(:,isnan(X_cons))=[];
    X_cons(:,isnan(X_cons))=[];
    
    stx=zeros(1,2).*nan;
    [X_cons,stx(1),stx(2)]=zscore(X_cons);
    
    datestr(now)
    para_index=1:para_num;

%% stepwise start
    for NN1=1:para_num
        loop_size=length(X_chos(:,1));
        dE=ones(1,loop_size).*nan;
        npool=sort(reshape(repmat(1:loop_size,[K_num 1]),[1 loop_size.*K_num]));
        kpool=repmat(1:K_num,[1 loop_size]);
        Ety=cell(1,loop_size*K_num);Eyy=Ety;
        parfor re=1:loop_size*K_num
            n=npool(re);
            tempX_chos=X_chos;
            if mean(tempX_chos(n,:),'all')==0
                continue
            end
            prandom_num=random_num;
            X_para=[X_cons;tempX_chos(n,:);X_P];
            kfold=kpool(re);
            pstx=stx;
            tempX=X_para;
            
            TEST=tempX(:,X_K==kfold);
            tempX(:,X_K==kfold)=[];
            
            tempY=tempX(1,:);
            tempX(1,:)=[];
            testx=TEST;
            testy=testx(1,:);
            testx(1,:)=[];
            tptrain=[tempX(1:end-p_num,:);tempY];
            temp=mean(tptrain,1);
            tptrain(:,isnan(temp))=[];
            
            net=constantnet;
            
            for i=1:average_num
                if FFNNsize==0
                    setdemorandstream(prandom_num(i));
                    net{1,i}=newgrnn(tptrain(1:end-1,:),tptrain(end,:),1);
                else
                    setdemorandstream(prandom_num(i));
                    net{1,i}=train(net{1,i},tptrain(1:end-1,:),tptrain(end,:));
                    nnet=net{1,i};
                    temp1=nnet(tptrain(1:end-1,:));
                    setdemorandstream(prandom_num(i));
                    net{2,i}=train(net{2,i},[tptrain(1:end-1,:);temp1],tptrain(end,:));
                    nnet=net{2,i};
                    temp2=nnet([tptrain(1:end-1,:);temp1]);
                    setdemorandstream(prandom_num(i));
                    net{3,i}=train(net{3,i},[tptrain(1:end-1,:);temp1;temp2],tptrain(end,:));
                end
            end

            tempx=testx;
            Ety{1,re}=tempx(end-p_num+1:end,:);
            temp=tempx(1:end-p_num,:);
            
            temp3=zeros(average_num,length(temp(1,:))).*nan;
            for i=1:average_num
                tempnet=net{1,i};
                temp1=tempnet(temp);
                tempnet=net{2,i};
                temp2=tempnet([temp;temp1]);
                tempnet=net{3,i};
                temp3(i,:)=tempnet([temp;temp1;temp2]).*pstx(2)+pstx(1);
            end

            testy=testy.*pstx(2)+pstx(1);
            
            
            Ety{1,re}(p_num+1,:)=testy;
            Eyy{1,re}=temp3;
        end
        
        for n=1:loop_size
            yy=[];
            ty=[];
            for j=1:K_num
                    yy=[yy Eyy{1,(n-1).*K_num+j}];%#ok<AGROW>
                    ty=[ty Ety{1,(n-1).*K_num+j}];%#ok<AGROW>
            end
            if isempty(yy)
                continue
            end
            
            lo=(ty(1,:)==Province_num);
            pyy=yy(:,lo==1);
            pty=ty(p_num+1,lo==1);     
            
            pty(:,isnan(pyy(1,:)))=[];
            pyy(:,isnan(pyy(1,:)))=[];
            pyy(:,isnan(pty))=[];
            pty(:,isnan(pty))=[];
            tempdE=zeros(1,50).*nan;
            for i=1:average_num
                tempdE(i)=nanmean(abs(pyy(i,:)-pty),'all');
            end
            
            if NN1==1
                dE(1,n)=nanmean(tempdE,'all');
            else
                dE(1,n)=nanmean(tempdE,'all')-Estatus;
            end
        end
        if isnan(nanmean(dE,'all'))
            break
        end
        if NN1==1
            E0=min(dE(1,:),[],'all');
            [~,minindex]=find(dE(1,:)==E0);
            
            Estatus=E0;
            E(Province_num,NN1)=Estatus;
            
            Xparaindex{1,Province_num}=[Xparaindex{1,Province_num} para_index(minindex(1))];
            disp([num2str(Estatus),', ',X_chosinfo{1,Xparaindex{1,Province_num}(end)}])
            running_info=[running_info;string([num2str(Estatus),', ',X_chosinfo{1,Xparaindex{1,Province_num}(end)}])]; %#ok<AGROW>
            para_index(minindex(1))=[];
            X_cons=[X_cons;X_chos(minindex(1),:)]; %#ok<AGROW>
            X_chos(minindex(1),:)=[];
        else
            E0=min(dE(1,:),[],'all');
            [~,minindex]=find(dE(1,:)==E0);
            Estatus=E(Province_num,NN1-1)+E0;
            
            if NN1>4
                temp=[E(Province_num,NN1-validation_check:NN1-1) Estatus];
               if temp(1)<min(temp(2:end),[],'all')
                    Xparaindex{1,Province_num}(end)=[];
                    break
               end
            end
            
            E(Province_num,NN1)=Estatus;
            Xparaindex{1,Province_num}=[Xparaindex{1,Province_num} para_index(minindex(1))];
            para_index(minindex(1))=[];
            X_cons=[X_cons;X_chos(minindex(1),:)]; %#ok<AGROW>
            X_chos(minindex(1),:)=[];
            disp([num2str(Estatus),', ',X_chosinfo{1,Xparaindex{1,Province_num}(end)}])
            running_info=[running_info;string([num2str(Estatus),', ',X_chosinfo{1,Xparaindex{1,Province_num}(end)}])]; %#ok<AGROW>
        end
        save(['running_info_province_',num2str(Province_num),'.mat'],'running_info','Xparaindex')
        if E0<0
            for NN2=1:NN1
                loop_size=NN1;
                npool=sort(reshape(repmat(1:loop_size,[K_num 1]),[1 loop_size.*K_num]));
                kpool=repmat(1:K_num,[1 loop_size]);
                Ety=cell(1,loop_size*K_num);Eyy=Ety;
                dE=zeros(1,loop_size).*nan;
                parfor re=1:loop_size*K_num
                    n=npool(re);
                    prandom_num=random_num;
                    temp=X_cons;
                    if (mean(temp(n+1,:),'all'))==0
                        continue
                    else
                        temp(n+1,:)=[];
                    end
                    X_para=[temp;X_P];
                    
                    kfold=kpool(re);
                    pstx=stx;
                    tempX=X_para;
                    
                    TEST=tempX(:,X_K==kfold);
                    tempX(:,X_K==kfold)=[];
                    
                    tempY=tempX(1,:);
                    tempX(1,:)=[];
                    testx=TEST;
                    testy=testx(1,:);
                    testx(1,:)=[];
                    tptrain=[tempX(1:end-p_num,:);tempY];
                    temp=mean(tptrain,1);
                    tptrain(:,isnan(temp))=[];
                    net=constantnet;
                    for i=1:average_num
                        if FFNNsize==0
                            setdemorandstream(prandom_num(i));
                            net{1,i}=newgrnn(tptrain(1:end-1,:),tptrain(end,:),1);
                        else
                            setdemorandstream(prandom_num(i));
                            net{1,i}=train(net{1,i},tptrain(1:end-1,:),tptrain(end,:));
                            nnet=net{1,i};
                            temp1=nnet(tptrain(1:end-1,:));
                            setdemorandstream(prandom_num(i));
                            net{2,i}=train(net{2,i},[tptrain(1:end-1,:);temp1],tptrain(end,:));
                            nnet=net{2,i};
                            temp2=nnet([tptrain(1:end-1,:);temp1]);
                            setdemorandstream(prandom_num(i));
                            net{3,i}=train(net{3,i},[tptrain(1:end-1,:);temp1;temp2],tptrain(end,:));
                        end
                    end   
                    tempx=testx;
                    Ety{1,re}=tempx(end-p_num+1:end,:);
                    temp=tempx(1:end-p_num,:);
                    
                    temp3=zeros(average_num,length(temp(1,:))).*nan;
                    for i=1:average_num
                        tempnet=net{1,i};
                        temp1=tempnet(temp);
                        tempnet=net{2,i};
                        temp2=tempnet([temp;temp1]);
                        tempnet=net{3,i};
                        temp3(i,:)=tempnet([temp;temp1;temp2]).*pstx(2)+pstx(1);
                    end
                    testy=testy.*pstx(2)+pstx(1);
                    
                    Ety{1,re}(p_num+1,:)=testy;
                    Eyy{1,re}=temp3;
                end
                for n=1:loop_size
                    yy=[];
                    ty=[];
                    for j=1:K_num
                        
                        yy=[yy Eyy{1,(n-1).*K_num+j}];%#ok<AGROW>
                        ty=[ty Ety{1,(n-1).*K_num+j}];%#ok<AGROW>
                        
                    end
                    if isempty(yy)
                        continue
                    end
                    lo=(ty(1,:)==Province_num);
                    pyy=yy(:,lo==1);
                    pty=ty(p_num+1,lo==1);
                    pty(:,isnan(pyy(1,:)))=[];
                    pyy(:,isnan(pyy(1,:)))=[];
                    pyy(:,isnan(pty))=[];
                    pty(:,isnan(pty))=[];
                    
                    tempdE=zeros(1,50).*nan;
                    for i=1:average_num
                        tempdE(i)=nanmean(abs(pyy(i,:)-pty),'all');
                    end
                    
                    dE(1,n)=nanmean(tempdE,'all')-Estatus;

                    
                end
                E0=min(dE(1,:),[],'all');
                [~,minindex]=find(dE(1,:)==E0);
                
                if E0<0
                    Estatus=E(Province_num,NN1)+E0;
                    E(Province_num,NN1)=Estatus;
                    temp=minindex(1);
                    
                    disp([num2str(Estatus),', ',X_chosinfo{1,Xparaindex{1,Province_num}(temp)},' removed'])
                    running_info=[running_info;string([num2str(Estatus),', ',X_chosinfo{1,Xparaindex{1,Province_num}(temp)},' removed'])]; %#ok<AGROW>
                    
                    Xparaindex{1,Province_num}(temp)=nan;
                    X_cons(minindex(1)+1,:)=zeros(size(X_cons(minindex(1)+1,:)));
                else
                    break
                end
            end
        end
    end
        save(['running_info_province_',num2str(Province_num),'.mat'],'running_info','Xparaindex')
end
    temp=Xparaindex{1,Province_num};
    temp(isnan(temp))=[];
    Xparaindex{1,Province_num}=temp;
    save(['running_info_province_',num2str(Province_num),'.mat'],'running_info','Xparaindex')
    disp('Stepwise BEL FFNN selection finished')
end


