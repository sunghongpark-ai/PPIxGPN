function Lppi = PPInetwork(network_data, options)

arguments
    network_data (:,:) double {mustBeReal, mustBeFinite}
    options.threshold (1,1) double {mustBeReal, mustBeFinite, mustBePositive} = 0.4
end

num_protein = size(network_data, 1);
if size(network_data, 2) ~= num_protein
    error("PPIxGPN:InvalidNetwork", "network_data must be a square protein-by-protein matrix.");
end

linked = network_data >= options.threshold;
score = full(network_data(linked));
if isempty(score) || all(score == score(1))
    standardized = zeros(size(score));
else
    standardized = (score - mean(score)) ./ std(score);
end
weight = zeros(size(network_data), 'like', network_data);
weight(linked) = 1 ./ (1 + exp(-standardized));

scale = 1 ./ sqrt(full(sum(weight, 2)));
scale(isinf(scale)) = 0;
normalizer = spdiags(scale, 0, num_protein, num_protein);
Lppi = speye(num_protein) - normalizer * weight * normalizer;

end
