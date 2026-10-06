function [problem, weight] = TrainProblem(dataset, parameter, gradient_type)

arguments
    dataset (1,1) struct
    parameter (1,1) struct
    gradient_type (1,1) string
end

[weight, param_size] = StackParam(parameter);
num_protein = param_size(1, 1);
num_target = size(param_size, 1) - 1;
validateattributes(dataset.Lppi, {'numeric'}, {'real', 'finite', 'size', [num_protein, num_protein]}, 'PPIxGPN', 'dataset.Lppi');
validateattributes(dataset.Xtrain, {'numeric'}, {'2d', 'real', 'finite', 'nonempty', 'nrows', num_protein}, 'PPIxGPN', 'dataset.Xtrain');
Ytrain = double(dataset.Ytrain);
validateattributes(Ytrain, {'double'}, {'real', '>=', 0, '<=', 1, 'size', [size(dataset.Xtrain, 2), num_target]}, 'PPIxGPN', 'dataset.Ytrain');
validateattributes(parameter.gamma, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'}, 'PPIxGPN', 'parameter.gamma');

problem = struct("Lppi", double(dataset.Lppi), "Xtrain", double(dataset.Xtrain), "Ytrain", Ytrain, ...
    "gamma", double(parameter.gamma), "legacy", strcmp(gradient_type, "legacy"), ...
    "block_size", [num_protein, 1; num_protein, num_target]);

end
