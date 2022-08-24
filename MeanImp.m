% perform mean/median/mode imputation 
% ------------------------------------------------------------------------
% options:  mean/median/mode imputation
%      
% inputs:   Data - rough data set for processing
%           Flag - index for numerical features (1 numeric, 2 nominal)
%                  (mode for nominal, median for ordinal, mean for numeric)
%
% Missing values are marked as -1 instead of ? in original .mat file
% outputs:  Pro_data - the prcessed the data

function [mean_imputed_data] = MeanImp(Data, Flag)

for i = 1 : size(Data,2)
    if any(isnan(Data(:,i)))
        if Flag(:,i) == 1
            Data(isnan(Data(:,i)),i) = mean(Data(~isnan(Data(:,i)),i));
        elseif Flag(:,i) == 2
            Data(isnan(Data(:,i)),i) = mode(Data(isnan(Data(:,i)),i));
        end
    end  
end

mean_imputed_data = Data;




