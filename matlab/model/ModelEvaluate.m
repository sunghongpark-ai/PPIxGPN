function [loss_train, readout, grad_param] = ModelEvaluate(weight, problem)

arguments
    weight (:,1) double
    problem (1,1) struct
end

[Uppi, Bset] = ResizeParam(weight, problem.block_size);
[readout, adjoint, solver] = PropagationReadout(problem.Lppi, Uppi, Bset);
logit = problem.Xtrain.' * readout;
loss_train = CrossEntropy(logit, problem.Ytrain);

if nargout > 2
    num_train = size(problem.Xtrain, 2);
    x_residual = problem.Xtrain * (1 ./ (1 + exp(-logit)) - problem.Ytrain);
    grad_bset = (solver \ (Uppi .* x_residual)) / num_train;
    if problem.legacy
        grad_uppi = sum(Bset .* (problem.Lppi.' * ((x_residual.' / solver) / solver).'), 2) / num_train;
    else
        grad_uppi = sum(adjoint .* (solver \ (problem.Lppi * x_residual)), 2) / num_train;
    end
    grad_param = [grad_uppi; grad_bset(:)] + 2 * problem.gamma * weight;
end

end
