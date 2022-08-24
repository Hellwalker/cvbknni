
% the main program
% knnimpute custom: GRA, k = 3, dudani, no imputation ordering, with MI
% G3D
% Jianglin

function[ImpData] = G3D(Data)

% add nature index
[R,C] = size(Data);
ImpData = [(1:R)', Data];

% find missing col index
isNaNData = isnan(ImpData);
[~, cmis] = find(isNaNData);

cmis = unique(cmis);
m = 1;
for itemp = 1:size(cmis,1)
    px = cmis(itemp,1);
    % temperal mis col index for row with missingness
    tempIn = isnan(ImpData(:,px));
    tempMisColIndex = find(tempIn);
    
    for jtemp = 1:size(tempMisColIndex,1)
        % each time select one mis col index
        py = tempMisColIndex(jtemp,1);
        if ~isnan(ImpData(py,px))
            continue;
        end
        pyISNAN = isnan(ImpData(py,:));
        
        temp = find(sum(repmat(pyISNAN, R, 1)==isnan(ImpData), 2)==C+1);
        
        pxpyAvailableFeatIndex = logical(1-pyISNAN);
        % RestData for train
        RestData = ImpData(logical(1-tempIn),pxpyAvailableFeatIndex);
        RestData = ListDel(RestData);
        Y = ImpData(RestData(:,1),px);
        
        % test cases with unknown ytest, generally one row or some
        xtest = ImpData(temp, pxpyAvailableFeatIndex);
        delta = size(xtest,1)-1;
        
        % ImpKernel func to get the imputedValue for corresponding missingness
        ImpData(temp, px) = feval(@IDknniKernel, RestData(:,2:end), xtest(:,2:end), Y);
        tempIn(temp,1) = 0;
        m = m+delta+1;
    end
end

% exclude nature index
[~,order4output] = sort(ImpData(:,1));
ImpData = ImpData(order4output,2:end);


% customly kenerl performs improved dynamic knn imputation
% inputs:   TrainlData & targetdata - last row is xtest for unknown y, rest is train
% outputs:  Pro_data - the prcessed the data

function[returnEstimate] = IDknniKernel(trainData, targetData, Y)
r = size(targetData,1);
returnEstimate = zeros(r,1);

for j = 1:r
    returnEstimate(j,:) = adaptation_numeric(trainData, targetData(j,:),  Y);
end

% adaptation mix for numeric values
% four adaptation mths, mean, median, IDWM an IRWM
% distance measure, mikowski and grg/grc
function[para_pred] = adaptation_numeric(Xtrain, Xtest, Ytrain)

% DISTFUN = {'grc'};
% ADAPTATION_METHOD = {'Dudani'};

K = 3;
[trainCaseNum,c] = size(Xtrain);
MI = zeros(1,c);

for i = 1:c
    MI(1,i) = mutualinfo(discretization(Xtrain(:,i)),discretization(Ytrain));
end

w = (MI+0.00000001)./(sum(MI)+0.00000001);

% different dealing according to the datatype of variables need imputation
% GRA based distance measure
% X, Y and datatype
% flag = [0,...,0] all numeric
% GRG is similarity measure, NOT dissimilarity measure
testCaseNum = size(Xtest,1);
Dis = zeros(trainCaseNum,testCaseNum);
for i = 1:testCaseNum
    Dis(:,i) = 1-GRG(Xtrain, Xtest(i,:), zeros(1,c), w);
end


[distance, I] = sort(Dis);

K = (K>=trainCaseNum)*trainCaseNum + (K<trainCaseNum)*K;
returned_ind = I(1:K,:);
C = size(returned_ind,2);

t2 = reshape(Ytrain(reshape(returned_ind, K*C,1)), K, C);

denom = max(distance(1:K,:),[],1) - min(distance(1:K,:),[],1) + 0.0000000001;
w2 =  bsxfun(@rdivide,bsxfun(@minus,max(distance(1:K,:),[],1),distance(1:K,:))+0.0000000001,denom);
para_pred = (sum(bsxfun(@times, w2, t2),1)./sum(w2,1))';



