function param = AdamInit(param_set, learn_rate, beta1, beta2, epsilon)

arguments
    param_set (:,1) double
    learn_rate (1,1) double {mustBeReal, mustBeFinite, mustBePositive} = 1e-4
    beta1 (1,1) double {mustBeInRange(beta1, 0, 1, "exclude-upper")} = 0.9
    beta2 (1,1) double {mustBeInRange(beta2, 0, 1, "exclude-upper")} = 0.999
    epsilon (1,1) double {mustBeReal, mustBeFinite, mustBePositive} = 1e-8
end

param = struct("alpha", learn_rate, "beta1", beta1, "beta2", beta2, "epsilon", epsilon, ...
    "t", 0, "m", zeros(size(param_set)), "v", zeros(size(param_set)));

end
