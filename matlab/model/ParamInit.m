function param = ParamInit(num_protein, mu, options)

arguments
    num_protein (1,1) double {mustBeInteger, mustBePositive}
    mu (1,1) double {mustBeReal, mustBeFinite}
    options.stream (1,1) RandStream = RandStream.getGlobalStream
end

if mu == 0
    param = (2 * rand(options.stream, num_protein, 1) - 1) * sqrt(6 / (num_protein + 1));
else
    param = mu * ones(num_protein, 1);
end

end
