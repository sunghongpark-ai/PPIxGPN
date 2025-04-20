function B = ParamInit(num_protein,mu)

if (mu==0)
    B = zeros(num_protein,1); szB = size(B);
    B = (2*rand(szB)-1)*sqrt(6/sum(szB));
else
    B = mu*ones(num_protein,1);
end