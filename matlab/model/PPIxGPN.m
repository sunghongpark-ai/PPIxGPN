function [pred_risk, model_param, history, dataset, parameter] = PPIxGPN(Xdata, Ydata, ppi_data, idx_train, idx_valid, idx_test, options)

arguments
    Xdata (:,:) double {mustBeReal, mustBeFinite}
    Ydata (:,4) double {mustBeReal, mustBeInRange(Ydata, 0, 1)}
    ppi_data (:,:) double {mustBeReal, mustBeFinite}
    idx_train {mustBeNumericOrLogical}
    idx_valid {mustBeNumericOrLogical}
    idx_test {mustBeNumericOrLogical}
    options.epoch (1,1) double {mustBeInteger, mustBePositive} = 1000
    options.rate (1,1) double {mustBeReal, mustBeFinite, mustBePositive} = 0.001
    options.gamma (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 0.01
    options.threshold (1,1) double {mustBeReal, mustBeFinite, mustBePositive} = 0.4
    options.seed double {mustBeScalarOrEmpty, mustBeInteger, mustBeNonnegative, mustBeLessThan(options.seed, 4294967296)} = []
    options.gradient (1,1) string {mustBeMember(options.gradient, ["exact", "legacy"])} = "exact"
    options.phi_min (1,1) double {mustBeReal, mustBeNonNan} = -Inf
end

[num_protein, num_participant] = size(Xdata);
if size(Ydata, 1) ~= num_participant
    error("PPIxGPN:SizeMismatch", "Ydata must have one row per participant: expected %d rows, found %d.", num_participant, size(Ydata, 1));
end
if ~isequal(size(ppi_data), [num_protein, num_protein])
    error("PPIxGPN:SizeMismatch", "ppi_data must be %d-by-%d to match the proteins in Xdata.", num_protein, num_protein);
end

idx_train = SplitIndex(idx_train, num_participant, 'idx_train', true);
idx_valid = SplitIndex(idx_valid, num_participant, 'idx_valid', true);
idx_test = SplitIndex(idx_test, num_participant, 'idx_test', false);
if ~isempty(intersect(idx_train, idx_valid)) || ~isempty(intersect(idx_train, idx_test)) || ~isempty(intersect(idx_valid, idx_test))
    error("PPIxGPN:OverlappingSplit", "idx_train, idx_valid, and idx_test must select disjoint participants.");
end

dataset.Xtrain = Xdata(:, idx_train);
dataset.Ytrain = Ydata(idx_train, :);
dataset.Xvalid = Xdata(:, idx_valid);
dataset.Yvalid = Ydata(idx_valid, :);
dataset.Xtest = Xdata(:, idx_test);
dataset.Ytest = Ydata(idx_test, :);
dataset.Lppi = PPInetwork(ppi_data, "threshold", options.threshold);

if isempty(options.seed)
    stream = RandStream.getGlobalStream;
else
    stream = RandStream("mt19937ar", "Seed", options.seed);
end
parameter.Uppi = ParamInit(num_protein, 1);
parameter.Babt = ParamInit(num_protein, 0, "stream", stream);
parameter.Bgfa = ParamInit(num_protein, 0, "stream", stream);
parameter.Bnfl = ParamInit(num_protein, 0, "stream", stream);
parameter.Btau = ParamInit(num_protein, 0, "stream", stream);
parameter.epoch = options.epoch;
parameter.rate = options.rate;
parameter.gamma = options.gamma;

[model_param, history] = ModelTrain(dataset, parameter, "gradient", options.gradient, "phi_min", options.phi_min);
pred_risk = RiskPredict(dataset, parameter, model_param);

end


function idx = SplitIndex(idx, num_participant, name, required)

if islogical(idx)
    if ~isvector(idx) || numel(idx) ~= num_participant
        error("PPIxGPN:InvalidIndex", "Logical %s must contain one element per participant (%d).", name, num_participant);
    end
    idx = find(idx(:));
elseif isempty(idx)
    idx = zeros(0, 1);
else
    validateattributes(idx, {'numeric'}, {'vector', 'integer', 'positive', '<=', num_participant}, 'PPIxGPN', name);
    idx = double(idx(:));
end
if required && isempty(idx)
    error("PPIxGPN:EmptySplit", "%s must select at least one participant.", name);
end

end
