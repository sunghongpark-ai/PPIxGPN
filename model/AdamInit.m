function param = AdamInit(param_set, alpha, beta1, beta2, epsilon)

num_var = length(param_set);
if exist('alpha','var'); param.alpha = alpha; else param.alpha = 1e-4; end
if exist('beta1','var'); param.beta1 = beta1; else param.beta1 = 0.9; end
if exist('beta2','var'); param.beta2 = beta2; else param.beta2 = 0.999; end
if exist('epsilon','var'); param.epsilon = epsilon; else param.epsilon = 1e-8; end

param.t = 0;
param.m = zeros(num_var,1);
param.v = zeros(num_var,1);

