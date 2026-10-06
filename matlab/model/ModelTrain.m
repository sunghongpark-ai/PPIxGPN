function [model_param, history] = ModelTrain(dataset, parameter, options)

arguments
    dataset (1,1) struct
    parameter (1,1) struct
    options.gradient (1,1) string {mustBeMember(options.gradient, ["exact", "legacy"])} = "exact"
    options.phi_min (1,1) double {mustBeReal, mustBeNonNan} = -Inf
end

[problem, weight] = TrainProblem(dataset, parameter, options.gradient);
num_protein = problem.block_size(1, 1);
num_target = problem.block_size(2, 2);
validateattributes(dataset.Xvalid, {'numeric'}, {'2d', 'real', 'finite', 'nonempty', 'nrows', num_protein}, 'ModelTrain', 'dataset.Xvalid');
Xvalid = double(dataset.Xvalid);
Yvalid = double(dataset.Yvalid);
validateattributes(Yvalid, {'double'}, {'real', '>=', 0, '<=', 1, 'size', [size(Xvalid, 2), num_target]}, 'ModelTrain', 'dataset.Yvalid');
validateattributes(parameter.epoch, {'numeric'}, {'scalar', 'integer', 'positive'}, 'ModelTrain', 'parameter.epoch');
validateattributes(parameter.rate, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, 'ModelTrain', 'parameter.rate');

num_epoch = double(parameter.epoch);
weight(1:num_protein) = max(weight(1:num_protein), options.phi_min);
adam_param = AdamInit(weight, double(parameter.rate));
history = struct("LossTrain", zeros(num_epoch, num_target), "LossValid", zeros(num_epoch, num_target), ...
    "BestEpoch", 0, "BestLoss", Inf);
model_param = weight;
grad_param = zeros(size(weight));

for idx_epoch = 1:num_epoch
    if idx_epoch > 1
        [weight, adam_param] = WeightUpdate(weight, grad_param, adam_param);
        weight(1:num_protein) = max(weight(1:num_protein), options.phi_min);
    end
    [loss_train, readout, grad_param] = ModelEvaluate(weight, problem);
    loss_valid = CrossEntropy(Xvalid.' * readout, Yvalid);
    if ~all(isfinite([loss_train, loss_valid])) || ~all(isfinite(grad_param))
        error("PPIxGPN:NonfiniteValue", "Training produced a nonfinite loss or gradient at epoch %d.", idx_epoch);
    end
    history.LossTrain(idx_epoch, :) = loss_train;
    history.LossValid(idx_epoch, :) = loss_valid;
    score = mean(loss_valid);
    if score < history.BestLoss
        history.BestEpoch = idx_epoch;
        history.BestLoss = score;
        model_param = weight;
    end
end

end
