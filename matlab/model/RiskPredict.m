function [pred_risk, syn_effect] = RiskPredict(dataset, parameter, model_param)

arguments
    dataset (1,1) struct
    parameter (1,1) struct
    model_param (:,1) double {mustBeReal, mustBeFinite}
end

[~, param_size] = StackParam(parameter);
num_protein = param_size(1, 1);
validateattributes(dataset.Lppi, {'numeric'}, {'real', 'finite', 'size', [num_protein, num_protein]}, 'RiskPredict', 'dataset.Lppi');
validateattributes(dataset.Xtest, {'numeric'}, {'2d', 'real', 'finite', 'nrows', num_protein}, 'RiskPredict', 'dataset.Xtest');
Xtest = double(dataset.Xtest);

[Uppi, Bset] = ResizeParam(model_param, [num_protein, 1; num_protein, size(param_size, 1) - 1]);
[readout, ~, solver] = PropagationReadout(double(dataset.Lppi), Uppi, Bset);
pred_risk = 1 ./ (1 + exp(-(Xtest.' * readout)));
if nargout > 1
    syn_effect = solver \ (Uppi .* Xtest);
end

end
