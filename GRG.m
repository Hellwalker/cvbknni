% grey relational coefficient GRC()
% input x vec
%       y vec matrix
%       type numeric, nominal vec 0-numeric 1-nominal
%       w feature weight

function[grg] = GRG(x, y, dataTypeVec, w)

C = size(x,2);

if nargin == 3
   w = ones(1, C)./C;
end

% distinguish coefficient rho set to 0.5
rho = 0.5;

% distance
d = abs(bsxfun(@minus, x(:,dataTypeVec==0), y(:,dataTypeVec==0)));

lamdaMIN = min(min(d));

lamdaMAX = max(max(d));

% GRC measure is separated as numeric and nominal
GRC_numeric = sum(bsxfun(@times, w(:,dataTypeVec==0), (lamdaMIN + rho * lamdaMAX)./(d+rho*lamdaMAX)),2);

% GRC=1 if x0p and xip is the same; GRC=0 if not
if sum(dataTypeVec)~=0
    GRC_nominal = sum(bsxfun(@eq, x(:,dataTypeVec==1), y(:,dataTypeVec==1))).*w;
else
    GRC_nominal = 0;
end

% GRG grey relational grade is the mean of GRC
grg = GRC_numeric + GRC_nominal;

