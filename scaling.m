% perform scaling or standardization, the transformation approaches
% options:  1. [0, 1]
%           2. [-1, 1]
%           3. xi-xbar / stdev
%           4. xi-median(xi) / sum(xi - median(xi))
%
% inputs:   Pre_data - rough data set for processing
%           StdOp - option of type of transformation
%
% outputs:  Pro_data - the prcessed the data

function [Pro_data] = scaling(Pre_data,StdOp, minmax)

if nargin == 2
    MaxV = max(Pre_data);
    MinV = min(Pre_data);
elseif nargin ==3
    MaxV = minmax(2,:);
    MinV = minmax(1,:);
end

switch(StdOp)
    case 1 % scale to [0.0001, 0.9999]
        % get min and max
        Pro_data = 0.0001 + 0.9998 * bsxfun(@rdivide, bsxfun(@minus,Pre_data,MinV), (MaxV-MinV));
        
    case 2 % scale to [-0.9999, 0.9999]
        % get min and max
        Pro_data = 0.9999 * bsxfun(@rdivide, bsxfun(@minus, 2*Pre_data, (MaxV+MinV)), (MaxV-MinV));
        
    case 3 % null
        Pro_data = Pre_data;
        
    case 4 % xi-xbar / stdev
        Pro_data = bsxfun(@rdivide, bsxfun(@minus, Pre_data, mean(Pre_data)), std(Pre_data));
        
    case 5 % xi-median(xi) / sum(xi - median(xi))
        temp = bsxfun(@minus, Pre_data, median(Pre_data));
        Pro_data = 0.0001 + bsxfun(@rdivide, temp, sum(abs(temp)));
end

% modified to deal with the bug inside KnnImpKernel
% X = 2*(1:ceil(sqrt(size(Pre_data,2))-1))-1;
% Pro_data = bsxfun(@plus, Pro_data, (1: X)/100001000);





