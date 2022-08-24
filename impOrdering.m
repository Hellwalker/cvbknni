% rank the imputation order according to the characteristics of missingness

% by Hellwalker

% Fisher score detemines the most relevant features for classification

% rowOrder - store the original row order
% colOrder - store the original column order
% newFeatW - the feature weight calculated

function [orderedData, newFeatW, rowOrder, colOrder] = impOrdering(data)


fp = data(:,end);
X = data(:,1:end-1);

% missing rate of an instance, sort instance order
rowMissingRate = sum(isnan(data),2)./sum(~isnan(data),2);

[~, rowOrder] = sort(rowMissingRate);

% using fisher score to sort the feature order
% the larger, the more relevant
% Output:
%     out: A struct containing the following fields
% W - The distribution at each data point.
% fList - The list of features that are deemed useful.
% prf - This means that the smaller the feature weight is, the more useful it will be to the user.
% 
% Input:
%     X: The features on current trunk, each column is a feature vector on all instances, and each row is a part of the instance.
%     Y: The label of instances, in single column form: 1 2 3 4 5 ...

% fp > fp+1 1-nonfp, 2-fp
outfs = fsFisher(X, round(fp+1));

% newRankXfromFeat the most relevant to the least relevant
newRankXfromFeat = X(:,outfs.fList);
newFeatW = outfs.W(1,outfs.fList);
newFeatW = newFeatW>=1;
orderedData = [newRankXfromFeat(rowOrder,:), fp(rowOrder,:)];

colOrder = [outfs.fList, size(data,2)];



