function [grad_param, loss_train] = ModelGradient(dataset, parameter, model_param, options)

arguments
    dataset (1,1) struct
    parameter (1,1) struct
    model_param (:,1) double {mustBeReal, mustBeFinite}
    options.gradient (1,1) string {mustBeMember(options.gradient, ["exact", "legacy"])} = "exact"
end

problem = TrainProblem(dataset, parameter, options.gradient);
[loss_train, ~, grad_param] = ModelEvaluate(model_param, problem);

end
