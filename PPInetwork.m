function L = PPInetwork(network_data)

W = network_data;
num_protein = size(W,1);
W = W.*(W>=0.4);
W(W>0) = 1./(1+exp((-1)*(W(W>0)-mean(W(W>0)))./std(W(W>0))));
diagW = 1./sqrt(sum(W,2)); diagW(isinf(diagW)==1) = 0;
L = eye(num_protein)-(diag(diagW)*W*diag(diagW));