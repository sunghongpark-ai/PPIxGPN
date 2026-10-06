function [w, param] = WeightUpdate(w, g, param)

arguments
    w (:,1) double
    g (:,1) double
    param (1,1) struct
end

if ~isequal(size(g), size(w), size(param.m), size(param.v))
    error("PPIxGPN:SizeMismatch", "The weights, gradient, and Adam moments must have the same length.");
end

param.t = param.t + 1;
param.m = param.beta1 * param.m + (1 - param.beta1) * g;
param.v = param.beta2 * param.v + (1 - param.beta2) * g .^ 2;
m_hat = param.m / (1 - param.beta1 ^ param.t);
v_hat = param.v / (1 - param.beta2 ^ param.t);
w = w - param.alpha * m_hat ./ (sqrt(v_hat) + param.epsilon);

end
