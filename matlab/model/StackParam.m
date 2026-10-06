function [param_data, param_size] = StackParam(parameter)

arguments
    parameter (1,1) struct
end

field = {'Uppi', 'Babt', 'Bgfa', 'Bnfl', 'Btau'};
num_protein = numel(parameter.Uppi);
block = cell(numel(field), 1);
for k = 1:numel(field)
    value = parameter.(field{k});
    validateattributes(value, {'numeric'}, {'vector', 'real', 'finite', 'numel', num_protein}, 'StackParam', ['parameter.', field{k}]);
    block{k} = double(value(:));
end
param_data = vertcat(block{:});
param_size = repmat([num_protein, 1], numel(field), 1);

end
