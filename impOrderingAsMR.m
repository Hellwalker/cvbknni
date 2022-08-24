% rank the imputation order according to missing rate only

% by Hellwalker

% rowOrder - store the original row order
% colOrder - store the original column order
% newFeatW - the feature weight calculated

function [orderedData, rowOrder] = impOrderingAsMR(data)


fp = data(:,end);
X = data(:,1:end-1);

% missing rate of an instance, sort instance order
rowMissingRate = sum(isnan(data),2)./sum(~isnan(data),2);

[~, rowOrder] = sort(rowMissingRate,'ascend');

orderedData = [X(rowOrder,:), fp(rowOrder,:)];





