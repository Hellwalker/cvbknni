
% the main program of idknni algo
% imputation ordering with mr only 
% imputation direction change from col by col to row by row
% Jianglin

function[ImpData, paraSetting] = IDknnimpute3(Data)

% Step 1 imputation ordering via Fisher score and missing percentage
% re-ordered data -- orderedData
% [orderedData, oriRowOrder] = impOrderingAsMR(Data);
[orderedData, ~, oriRowOrder, oriColOrder] = impOrdering(Data);
% add nature index
[R,C] = size(orderedData);
ImpData = [(1:R)', orderedData];

% find missing col index
isNaNData = isnan(ImpData);
[rmis, ~] = find(isNaNData);
missingSum = sum(sum(isNaNData));
paraSetting = ones(missingSum, 3);
rmis = unique(rmis);
m = 1;
for itemp = 1:size(rmis,1)
    px = rmis(itemp,1);
    % temperal mis col index for row with missingness
    tempIn = isnan(ImpData(px,:));
    tempMisColIndex = find(tempIn);
    
    for jtemp = 1:size(tempMisColIndex,2)
        % each time select one mis col index
        py = tempMisColIndex(1,jtemp);       
        display(itemp);
        temp = find(sum(repmat(tempIn, R, 1)==isnan(ImpData), 2)==C+1);
        
        pxpyAvailableFeatIndex = logical(1-tempIn);
        % RestData for train
        RestData = ImpData(:,pxpyAvailableFeatIndex);
        RestData(temp,:) = [];
        RestData = ListDel(RestData);
        Y = ImpData(RestData(:,1),py);
        
        % test cases with unknown ytest, generally one row or some
        xtest = ImpData(temp, pxpyAvailableFeatIndex);
        delta = size(xtest,1)-1;
        
        A = ~isnan(Y);
        
        % ImpKernel func to get the imputedValue for corresponding missingness
        [ImpData(temp, py),paraSetting(m:(m+delta),:)] = feval(@IDknniKernel, RestData(A,2:end), xtest(:,2:end), Y(A,:));
        tempIn(1,py) = 0;
        m = m+delta+1;
    end
end

% exclude nature index
[~,order4output] = sort(ImpData(:,1));
ImpData = ImpData(order4output,2:end);
[~,a] = sort(oriRowOrder);
[~,b] = sort(oriColOrder);
ImpData = ImpData(a,b);



% customly kenerl performs improved dynamic knn imputation
%
% options:  mean/mode knn imputation for numeric/nominal data
%
% inputs:   TrainlData & targetdata - last row is xtest for unknown y, rest is train
%           misCase - the known Y corresponding to known X
%           compareM - matrix of mean and median
%
% Missing values are marked as -1 instead of ? in original .mat file
%
% NOTE: different with system knnimpute.m method, this method takes account
% of categorical data, and use pairwise deletion in process of detecting
% neighbors
%
% outputs:  Pro_data - the prcessed the data

function[returnEstimate, getPara] = IDknniKernel(trainData, targetData, Y)

DISTFUN = {'euclidean','cityblock','grc'};
SIZEDISTFUN = size(DISTFUN, 2);
ADAPTATION_METHOD = {'local_mean','local_median','IDWM','IRWM','Dudani'};
ADA_MTH_SUM = size(ADAPTATION_METHOD,2);

R = size(trainData,1);
r = size(targetData,1);
returnEstimate = zeros(r,1);
getPara = ones(r,3);
minErr = [];

% potential k pool with all odd number, maximum is sqrt(total)
kpool = 2*(1:round(sqrt(R)))-1;
kpool = kpool(:,1:ceil(size(kpool,2)/2));
%kpool = 1:round(sqrt(R));
kpool = [kpool,floor(R*0.9)];
K_LENGTH = size(kpool,2);

% stratified cross validation scheme 10 or less depends on data
cp = cvpartition(categorical(round(trainData(:,end))),'KFold',min(R,10));

% numeric knn imputation for numeric missingness
temp_times = ADA_MTH_SUM * K_LENGTH;
iter = SIZEDISTFUN * temp_times;

for ii = 1:iter
    % distance function index
    distfun = DISTFUN{1, ceil(ii/temp_times)};
    % kpool index to select k
    K = kpool(1, mod_custom(ceil(ii/ADA_MTH_SUM), K_LENGTH));
    % adaptation method index
    adafun = ADAPTATION_METHOD{1,mod_custom(ii, ADA_MTH_SUM)};
    % performance measure
    fun = @(XTRAIN,ytrain,XTEST)(adaptation_numeric(XTRAIN, XTEST, K, distfun, ytrain, adafun));
    %     display(distfun,'dis');
    %     display(K,'k');
    %     display(adafun,'ada');
    %     display(ii,'iteration');
    minErr(end+1:end+1,:) = crossval('mse', trainData, Y, 'predfun', fun, 'partition', cp);
end

[~, min_minErr_ind] = min(minErr);

% selected k: kpool(1, mod_custom(ceil(min_minErr_ind/ADAPTATION_METHOD_SUM), K_LENGTH))
% selected distance measure: ceil(min_minErr_ind/temp_times);
% selected adaptation method: mod_custom(min_minErr_ind, ADAPTATION_METHOD_SUM);

for j = 1:r
    selDisInd = ceil(min_minErr_ind/temp_times);
    selK = kpool(1, mod_custom(ceil(min_minErr_ind/ADA_MTH_SUM), K_LENGTH));
    selAdaInd = mod_custom(min_minErr_ind, ADA_MTH_SUM);
    getPara(j,:) = [selDisInd,selK,selAdaInd];
    returnEstimate(j,:) = adaptation_numeric(trainData, targetData(j,:), selK, DISTFUN{1,selDisInd}, Y, ADAPTATION_METHOD{1,selAdaInd});
end



% adaptation mix for numeric values
% four adaptation mths, mean, median, IDWM an IRWM
% distance measure, mikowski and grg/grc
function[para_pred] = adaptation_numeric(Xtrain, Xtest, K, DistMeasureMethod, Ytrain, adaMth)

[trainCaseNum,c] = size(Xtrain);
MI = zeros(1,c);

for i = 1:c
    MI(1,i) = mutualinfo(discretization(Xtrain(:,i)),discretization(Ytrain));
end

w = (MI+0.00000001)./(sum(MI)+0.00000001);

% different dealing according to the datatype of variables need imputation
switch DistMeasureMethod
    case 'euclidean'
        Dis = pdist2(bsxfun(@times,Xtrain,sqrt(w)), bsxfun(@times,Xtest,sqrt(w)), 'euclidean');
        
    case 'cityblock'
        Dis = pdist2(bsxfun(@times,Xtrain,w), bsxfun(@times,Xtest,w), 'cityblock');
        
    case 'grc'
        
        % GRA based distance measure
        % X, Y and datatype
        % flag = [0,...,0] all numeric
        % GRG is similarity measure, NOT dissimilarity measure
        testCaseNum = size(Xtest,1);
        Dis = zeros(trainCaseNum,testCaseNum);
        for i = 1:testCaseNum
            Dis(:,i) = 1-GRG(Xtrain, Xtest(i,:), zeros(1,c), w);
        end
end

[distance, I] = sort(Dis);

K = (K>=trainCaseNum)*trainCaseNum + (K<trainCaseNum)*K;
returned_ind = I(1:K,:);
C = size(returned_ind,2);

t2 = reshape(Ytrain(reshape(returned_ind, K*C,1)), K, C);

switch adaMth
    case 'local_mean'
        para_pred = (mean(t2,1))';
        
    case 'local_median'
        para_pred = (median(t2,1))';
        
    case 'IDWM' % inverse weighted mean of numeric k analogies
        Simi = bsxfun(@rdivide, 1.000000001, distance(1:K,:)+0.000000001);
        para_pred = (sum(t2.*bsxfun(@rdivide, Simi, sum(Simi,1)),1))';
        
        % the following inverse rank weighted mean of numeric k analogies
        % are given up in this experiment due to massive computation
    case 'IRWM' % inverse rank weighted mean of numeric k analogies
        w2 = (K:-1:1)./((K+1)*K/2);
        para_pred = (sum(bsxfun(@times,t2,w2'),1))';
        
    case 'Dudani'
        denom = max(distance(1:K,:),[],1) - min(distance(1:K,:),[],1) + 0.0000000001;
        w2 =  bsxfun(@rdivide,bsxfun(@minus,max(distance(1:K,:),[],1),distance(1:K,:))+0.0000000001,denom);
        para_pred = (sum(bsxfun(@times, w2, t2),1)./sum(w2,1))';
end


