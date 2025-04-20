function [w, param] = WeightUpdate(w, g, param)

alpha = param.alpha;
beta1 = param.beta1;
beta2 = param.beta2;
epsilon = param.epsilon;
t = param.t;
m = param.m;
v = param.v;

t = t + 1;
m = beta1 * m + (1-beta1) * g;
v = beta2 * v + (1-beta2) * (g.^2);
m_hat = m / (1 - beta1^t);
v_hat = v / (1 - beta2^t);
w = w - alpha * m_hat ./ (sqrt(v_hat) + epsilon);

param.alpha = alpha;
param.beta1 = beta1;
param.beta2 = beta2;
param.epsilon = epsilon;
param.t = t;
param.m = m;
param.v = v;
