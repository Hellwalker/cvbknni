function[LD_data] = ListDel(Data)

[row,~] = find(isnan(Data));
Data(unique(row),:) = [];
LD_data = Data;
end