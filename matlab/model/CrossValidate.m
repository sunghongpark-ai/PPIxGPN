function cv = CrossValidate(data, options)

arguments
    data (1,1) struct
    options.repeat (1,1) double {mustBeInteger, mustBePositive} = 1
    options.fold (1,1) double {mustBeInteger, mustBeGreaterThan(options.fold, 1)} = 5
    options.valid_ratio (1,1) double {mustBeInRange(options.valid_ratio, 0, 1, "exclusive")} = 0.2
    options.seed (1,1) double {mustBeInteger, mustBeNonnegative} = 1
    options.train (1,1) struct = struct()
    options.cutoff (1,1) double {mustBeInRange(options.cutoff, 0, 1)} = 0.5
    options.verbose (1,1) logical = false
end

if options.seed + options.repeat - 1 >= 2^32
    error("PPIxGPN:InvalidOption", "seed + repeat - 1 must be below 2^32.");
end
if isfield(options.train, "seed")
    error("PPIxGPN:InvalidOption", "Initialization seeds are drawn from the cross-validation stream; remove seed from train.");
end
[~, num_participant] = size(data.Xdata);
num_target = size(data.Ydata, 2);
if num_participant < options.fold
    error("PPIxGPN:InvalidOption", "fold cannot exceed the number of participants.");
end
if isfield(data, "target")
    target = string(data.target);
else
    target = "Y" + (1:num_target);
end
train_args = namedargs2cell(options.train);

cv.Assignment = zeros(num_participant, options.repeat);
cv.Risk = nan(num_participant, num_target, options.repeat);
cv.BestEpoch = zeros(options.repeat, options.fold);
row_count = options.repeat * options.fold * (num_target + 1);
repeat_column = zeros(row_count, 1);
fold_column = zeros(row_count, 1);
target_column = strings(row_count, 1);
metric_value = zeros(row_count, 4);
row = 0;
for r = 1:options.repeat
    stream = RandStream("mt19937ar", "Seed", options.seed + r - 1);
    [~, order] = sort(rand(stream, num_participant, 1));
    assignment = zeros(num_participant, 1);
    assignment(order) = mod((0:num_participant - 1).', options.fold) + 1;
    cv.Assignment(:, r) = assignment;
    for k = 1:options.fold
        idx_test = find(assignment == k);
        rest = find(assignment ~= k);
        [~, permutation] = sort(rand(stream, numel(rest), 1));
        num_valid = round(options.valid_ratio * numel(rest));
        if num_valid < 1 || num_valid >= numel(rest)
            error("PPIxGPN:InvalidOption", "valid_ratio leaves an empty training or validation set.");
        end
        idx_valid = sort(rest(permutation(1:num_valid)));
        idx_train = sort(rest(permutation(num_valid + 1:end)));
        init_seed = floor(rand(stream) * 2^31);
        [risk, ~, history] = PPIxGPN(data.Xdata, data.Ydata, data.ppi_data, idx_train, idx_valid, idx_test, ...
            train_args{:}, "seed", init_seed);
        cv.Risk(idx_test, :, r) = risk;
        cv.BestEpoch(r, k) = history.BestEpoch;
        metrics = EvaluateRisk(risk, data.Ydata(idx_test, :), "cutoff", options.cutoff, "target", target);
        span = row + (1:num_target + 1);
        repeat_column(span) = r;
        fold_column(span) = k;
        target_column(span) = [target, "Mean"];
        metric_value(span, :) = metrics{:, :};
        row = row + num_target + 1;
        if options.verbose
            fprintf("Repeat %d/%d, fold %d/%d: best epoch %d, mean test AUROC %.4f\n", r, options.repeat, ...
                k, options.fold, history.BestEpoch, metrics{"Mean", "AUROC"});
        end
    end
end

cv.FoldMetrics = [table(repeat_column, fold_column, target_column, 'VariableNames', {'Repeat', 'Fold', 'Target'}), ...
    array2table(metric_value, 'VariableNames', {'AUROC', 'AUPRC', 'Accuracy', 'F1'})];
name = [target, "Mean"];
summary_value = zeros(numel(name), 8);
for t = 1:numel(name)
    selected = metric_value(target_column == name(t), :);
    summary_value(t, :) = reshape([mean(selected, 1); std(selected, 0, 1)], 1, []);
end
cv.Summary = array2table(summary_value, 'VariableNames', ...
    {'AUROC_mean', 'AUROC_sd', 'AUPRC_mean', 'AUPRC_sd', 'Accuracy_mean', 'Accuracy_sd', 'F1_mean', 'F1_sd'}, ...
    'RowNames', cellstr(name));
cv.Options = options;

end
