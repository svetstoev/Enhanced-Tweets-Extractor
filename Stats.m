clear all
close all
clc

%% Load the Data and set the variables
Data_2012_16 = readtable('tweets.db');

% non-standardized variables
% independent
NIP = table2array(Data_2012_16 (:,2));
MCQ = table2array(Data_2012_16 (:,3));
CSS = table2array(Data_2012_16 (:,4));
BEE = table2array(Data_2012_16 (:,5));
BER = table2array(Data_2012_16 (:,6));
AEV = table2array(Data_2012_16 (:,7));
AES = table2array(Data_2012_16 (:,8));
ESS = table2array(Data_2012_16 (:,9));
%Sentdex = table2array(Data_2012_16 (:,10));
% dependent
Price_Closing = table2array(Data_2012_16 (:,11));
Volume = table2array(Data_2012_16 (:,12));
Return = table2array(Data_2012_16 (:,13));

% standardized variables
% independent
NIP_st = table2array(Data_2012_16 (:,15));
MCQ_st = table2array(Data_2012_16 (:,16));
CSS_st = table2array(Data_2012_16 (:,17));
BEE_st = table2array(Data_2012_16 (:,18));
BER_st = table2array(Data_2012_16 (:,19));
AEV_st = table2array(Data_2012_16 (:,20));
AES_st = table2array(Data_2012_16 (:,21));
ESS_st = table2array(Data_2012_16 (:,22));
%Sentdex_st = table2array(Data_2012_16 (:,23));
% dependent
Price_Closing_st = table2array(Data_2012_16 (:,24));
Volume_st = table2array(Data_2012_16 (:,25));
Return_st = table2array(Data_2012_16 (:,26));

% standardized changes
NIP_ch = table2array(Data_2012_16 (2:end,52));
MCQ_ch = table2array(Data_2012_16 (2:end,53));
CSS_ch = table2array(Data_2012_16 (2:end,54));
BEE_ch = table2array(Data_2012_16 (2:end,55));
BER_ch = table2array(Data_2012_16 (2:end,56));
AEV_ch = table2array(Data_2012_16 (2:end,57));
AES_ch = table2array(Data_2012_16 (2:end,58));
ESS_ch = table2array(Data_2012_16 (2:end,59));
%Sentdex_ch = table2array(Data_2012_16 (2:end,72));
% dependent
Price_Closing_ch = table2array(Data_2012_16 (2:end,61));
Volume_ch = table2array(Data_2012_16 (2:end,62));

Data_non_st = [Price_Closing NIP MCQ CSS BEE BER AEV AES ESS];
Data_st = [Price_Closing_st NIP_st MCQ_st CSS_st BEE_st BER_st AEV_st AES_st ESS_st];
Data_ch_st = [Price_Closing_ch NIP_ch MCQ_ch CSS_ch BEE_ch BER_ch AEV_ch AES_ch ESS_ch];
Data_ch = [Price_Closing_ch NIP_ch MCQ_ch CSS_ch BEE_ch BER_ch AEV_ch AES_ch ESS_ch];

%% Choose the data that is going to be used
Data = Data_ch_st
%% Statistics

% Durbin-Watson Statistics
% H0 - no serial autocorrelation
for i=1:(size(Data,2)-1)
    stats(i)=regstats(Data(:,1), Data(:,i+1), 'linear');
    [p_dw(i),dw(i)]=dwtest(stats(i).r, [ones(length(Data),1) Data(:,i+1)]);
end

% Mean
for i=1:(size(Data,2))
    something(i)=mean(Data(:,i));
end

% Standard Deviation
for i=1:(size(Data,2))
    stdeviation(i)=std(Data(:,i));
end

% Kurtosis
for i=1:(size(Data,2))
    kurt2(i)=kurtosis(Data(:,i));
end

%Skew
for i=1:(size(Data,2))
    skew(i)=skewness(Data(:,i));
end

%% Step 1
% Augmented Dickey Fuller Test (ADF)

for i=1:(size(Data,2))
    [adf(i),p_adf(i)]=adftest(Data(:,i));
end

%H0 - there is unit root (i.e. non-stationary)
% 0 - non stationary
% 1 - stationary

[adf_try,p_adf_try]=adftest(Data(:,8));

%% Step 2: Find the maximum order of integration

m=1 

%% Step 3: Create the Lag Matrix
lags = 5 % set the initialization value for the number of lags
Y = Data(:,1);
count=1;
for i=1:size(Data,2)
    for j=1:lags
        Data_lags(:,count)= lagmatrix(Data(:,i),j);
        count=count+1;
    end
end

%% Step 3: Setting up the VAR model by finding the optimal number of lags for each variable

for l=1:lags
    NIP_model{l} = fitglm(Data_lags(:,(lags+1):((lags+1)+l-1)),Y,'linear');
    MCQ_model{l} = fitglm(Data_lags(:,(2*lags+1):((2*lags+1)+l-1)),Y,'linear');
    CSS_model{l} = fitglm(Data_lags(:,(3*lags+1):((3*lags+1)+l-1)),Y,'linear');
    BEE_model{l} = fitglm(Data_lags(:,(4*lags+1):((4*lags+1)+l-1)),Y,'linear');
    BER_model{l} = fitglm(Data_lags(:,(5*lags+1):((5*lags+1)+l-1)),Y,'linear');
    AEV_model{l} = fitglm(Data_lags(:,(6*lags+1):((6*lags+1)+l-1)),Y,'linear');
    AES_model{l} = fitglm(Data_lags(:,(7*lags+1):((7*lags+1)+l-1)),Y,'linear');
    ESS_model{l} = fitglm(Data_lags(:,(8*lags+1):((8*lags+1)+l-1)),Y,'linear');
    Sentdex_model{l} = fitglm(Data_lags(:,(9*lags+1):((9*lags+1)+l-1)),Y,'linear');
    
    AIC_NIP(:,l) = NIP_model{1,l}.ModelCriterion.BIC;
    AIC_MCQ(:,l) = MCQ_model{1,l}.ModelCriterion.BIC;
    AIC_CSS(:,l) = CSS_model{1,l}.ModelCriterion.BIC;
    AIC_BEE(:,l) = BEE_model{1,l}.ModelCriterion.BIC;
    AIC_BER(:,l) = BER_model{1,l}.ModelCriterion.BIC;
    AIC_AEV(:,l) = AEV_model{1,l}.ModelCriterion.BIC;
    AIC_AES(:,l) = AES_model{1,l}.ModelCriterion.BIC;
    AIC_ESS(:,l) = ESS_model{1,l}.ModelCriterion.BIC;
    AIC_Sentdex(:,l) = Sentdex_model{1,l}.ModelCriterion.BIC;
    
end

lags_number_NIP=find(AIC_NIP==min(AIC_NIP))
lags_number_MCQ=find(AIC_MCQ==min(AIC_MCQ))
lags_number_CSS=find(AIC_CSS==min(AIC_CSS))
lags_number_BEE=find(AIC_BEE==min(AIC_BEE))
lags_number_BER=find(AIC_BER==min(AIC_BER))
lags_number_AEV=find(AIC_AEV==min(AIC_AEV))
lags_number_AES=find(AIC_AES==min(AIC_AES))
lags_number_ESS=find(AIC_ESS==min(AIC_ESS))
lags_number_Sentdex=find(AIC_Sentdex==min(AIC_Sentdex))

%% MACHINE LEARNING

% Lasso

%NIP_ch_1 = lagmatrix(NIP_ch,1)
MCQ_ch_1 = lagmatrix(MCQ_ch,1)
MCQ_ch_2 = lagmatrix(MCQ_ch,2)
BER_ch_2 = lagmatrix(BER_ch,2)
AEV_ch_1 = lagmatrix(AEV_ch,1)
AES_ch_1 = lagmatrix(AES_ch,1)
AES_ch_2 = lagmatrix(AES_ch,2)
ESS_ch_1 = lagmatrix(ESS_ch,1)
ESS_ch_2 = lagmatrix(ESS_ch,2)
%Sentdex_ch_1 = lagmatrix(Sentdex_ch,1)
%Sentdex_ch_2 = lagmatrix(Sentdex_ch,2)
X = [MCQ_ch_1 MCQ_ch_2 BER_ch_2 AEV_ch_1 AES_ch_1 AES_ch_2 ESS_ch_1 ESS_ch_2];% clean_Sentiment(2:end,end)];

% Define the Variables

Y=Price_Closing_ch

%X = [ones(length(Price_Closing_ch),1) NIP_ch_1 MCQ_ch_1 AEV_ch_2 AES_ch_1 AES_ch_2 ESS_ch_2];% clean_Sentiment(2:end,end)];

X=X(3:end,:)
Y=Y(3:end,:)

[h,pValue,stat,cValue] = chowtest(X(:,1:end),Y,1684)
%% historical mean
% X= X(:,[1 3 6 7 8 9]);
w=12;
y_pred_bench=zeros(length(X),1);
for i =(1+w):length(X)
    y_pred_bench(i)=mean(Y((i-w):i-1)); 
    
end

errors_bench(:,1)=y_pred_bench-Y(:,1);
errors_bench(:,2)=errors_bench(:,1).*errors_bench(:,1);

cum_rmse_bench=zeros(length(X),1);

for i =(1+w):length(X)
    cum_rmse_bench(i)=sqrt(mean(errors_bench((i-w):i-1,2))); 
    
end

plot(cum_rmse_bench(w+1:end));
%hold on
%% GLS
beta_lg=inv(X'*X)*X'*Y;
% beta_lg=glmfit(X(:,2:end),Y)
y_pred_lg=X*beta_lg;

errors_lg(:,1)=y_pred_lg-Y(:,1);
errors_lg(:,2)=errors_lg(:,1).*errors_lg(:,1);

cum_rmse_lg=zeros(length(X),1);

for i =(1+w):length(X)
    cum_rmse_lg(i)=sqrt(mean(errors_lg((i-w):(i-1),2))); 
    
end

% plot(cum_rmse_lg(w+1:end));
% hold off
dif_bench_lg=cum_rmse_bench - cum_rmse_lg;
%plotyy(1,2);

 % plotting lg and bench
 hold on
 t=(1+w):length(X);
yyaxis left
plot(t,cum_rmse_lg(w+1:end),t,cum_rmse_bench(w+1:end));

yyaxis right
plot(t,dif_bench_lg(w+1:end));


xlabel('Time');
% plotyy(t,[cum_rmse_lg(w+1:end)cum_rmse_bench(w+1:end)],t,dif_bench_lg(w+1:end));
hold off


%% LASSO FULL MODEL

 
[B, FitInfos]=lasso(X(:,2:end),Y,'DFmax',8);
B=B(:,1); % Corresponding to the optimal Lambda /Tuning Lambda/
    
B=[FitInfos.Intercept(1);B]; %Adding the intercept

y_pred_lassofull=X*B;

errors_lassofull(:,1)=y_pred_lassofull-Y(:,1);
errors_lassofull(:,2)=errors_lassofull(:,1).*errors_lassofull(:,1);

cum_rmse_lassofull=zeros(length(X),1);

for i =(1+w):length(X)
    cum_rmse_lassofull(i)=sqrt(mean(errors_lassofull((i-w):(i-1),2))); 
    
end
sum(cum_rmse_lassofull)
mean(cum_rmse_lassofull)
% plot(cum_rmse_lg(w+1:end));
% hold off

dif_bench_lassofull=cum_rmse_bench - cum_rmse_lassofull;
%plotyy(1,2);

% plotting lasso and bench
 hold on
 t=(1+w):length(X);
yyaxis left
plot(t,cum_rmse_lassofull(w+1:end),t,cum_rmse_bench(w+1:end));

yyaxis right
plot(t,dif_bench_lassofull(w+1:end));


xlabel('Time');


% LASSO with a moving window

y_pred_lasso=zeros(length(X),1);

for i =(1+w):length(X)
    
    [B, FitInfos] = lasso(X((i-w):(i-1),2:end),Y((i-w):(i-1),1),'CV',10,'DFmax',8);
    B=B(:,1);
    B=[FitInfos.Intercept(1);B];
 
    x=X((i-w):(i-1),2:end);
    x=[ones(w,1) x];
    er=x*B;
    y_pred_lasso(i) = er(end);

    close all
end
errors_lasso(:,1)=y_pred_lasso-Y(:,1);
errors_lasso(:,2)=errors_lasso(:,1).*errors_lasso(:,1);

cum_rmse_lasso=zeros(length(X),1);

for i =(1+w):length(X)
    cum_rmse_lasso(i)=sqrt(mean(errors_lasso((i-w):(i-1),2))); 
    
end
mean(cum_rmse_lasso)
sum(cum_rmse_lasso)
% plot(cum_rmse_lg(w+1:end));
% hold off
dif_bench_lasso=cum_rmse_bench - cum_rmse_lasso;
%plotyy(1,2);

% plotting lasso and bench
 hold on
 t=(1+w):length(X);
yyaxis left
plot(t,cum_rmse_lasso(w+1:end),t,cum_rmse_bench(w+1:end));

yyaxis right
plot(t,dif_bench_lasso(w+1:end));


xlabel('Time');

%% TREE FULL DATA

Yfit = predict(fitrtree(X(:,2:end),Y,'OptimizeHyperparameters','auto'),X(:,2:end));
errors_treefull(:,1)=Yfit-Y;
errors_treefull(:,2)=errors_treefull(:,1).*errors_treefull(:,1);

cum_rmse_treefull=zeros(length(X),1);

for i =(1+w):length(X)
    cum_rmse_treefull(i)=sqrt(mean(errors_treefull((i-w):(i-1),2))); 
    
end
sum(cum_rmse_treefull)
mean(cum_rmse_treefull)

dif_bench_tree=cum_rmse_bench - cum_rmse_treefull;

%plotting lasso and bench
 hold on
 t=(1+w):length(X);
yyaxis left
plot(t,cum_rmse_treefull(w+1:end),t,cum_rmse_bench(w+1:end));

yyaxis right
plot(t,dif_bench_tree(w+1:end));


xlabel('Time');

%% TREE MOVING WINDOW
y_pred_tree=zeros(length(X),1);
for i =(1+w):length(X)
    Yfit = predict(fitrtree(X((i-w):(i-1),2:end),Y((i-w):(i-1),1),'MinLeafSize',6),X((i-w):(i-1),2:end));
    y_pred_tree(i) = Yfit(end);
    close all
end

errors_treemov(:,1)=y_pred_tree-Y(:,1);
errors_treemov(:,2)=errors_treemov(:,1).*errors_treefull(:,1);

cum_rmse_treemov=zeros(length(X),1);

for i =(1+w):length(X)
    cum_rmse_treemov(i)=sqrt(mean(errors_treemov((i-w):(i-1),2))); 
    
end

dif_bench_treemov=cum_rmse_bench - cum_rmse_treemov;
%plotting moving tree and bench
 hold on
 t=(1+w):length(X);
yyaxis left
plot(t,cum_rmse_treemov(w+1:end),t,cum_rmse_bench(w+1:end));

yyaxis right
plot(t,dif_bench_treemov(w+1:end));


xlabel('Time');

    


