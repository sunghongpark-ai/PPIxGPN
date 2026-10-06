function result = RunSample(options)

arguments
    options.file string {mustBeScalarOrEmpty} = string.empty
    options.gamma (1,:) double {mustBeNonempty, mustBeReal, mustBeFinite, mustBeNonnegative} = [0.01, 0.001, 0.0001]
    options.phi_min (1,:) double {mustBeNonempty, mustBeReal, mustBeNonNan} = 0.01
    options.epoch (1,1) double {mustBeInteger, mustBePositive} = 3000
    options.rate (1,1) double {mustBeReal, mustBeFinite, mustBePositive} = 0.001
    options.gradient (1,1) string {mustBeMember(options.gradient, ["exact", "legacy"])} = "exact"
    options.seed (1,1) double {mustBeInteger, mustBeNonnegative, mustBeLessThan(options.seed, 4294967296)} = 1
    options.cutoff (1,1) double {mustBeInRange(options.cutoff, 0, 1)} = 0.5
    options.cv_repeat (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.cv_fold (1,1) double {mustBeInteger, mustBeGreaterThan(options.cv_fold, 1)} = 5
    options.output string {mustBeScalarOrEmpty} = string.empty
    options.verbose (1,1) logical = true
end

timer = tic;
data = LoadDataset(options.file);
num_protein = size(data.Xdata, 1);
num_target = numel(data.target);
[gamma_grid, phi_grid] = ndgrid(options.gamma, options.phi_min);
candidate = [gamma_grid(:), phi_grid(:)];
num_candidate = size(candidate, 1);
valid_loss = nan(num_candidate, 1);
best_epoch = zeros(num_candidate, 1);
status = strings(num_candidate, 1);
chosen = 0;
fit = struct();
for c = 1:num_candidate
    try
        [risk, model_param, history, dataset, parameter] = PPIxGPN(data.Xdata, data.Ydata, data.ppi_data, ...
            data.idx_train, data.idx_valid, data.idx_test, "epoch", options.epoch, "rate", options.rate, ...
            "gamma", candidate(c, 1), "phi_min", candidate(c, 2), "gradient", options.gradient, "seed", options.seed);
        valid_loss(c) = history.BestLoss;
        best_epoch(c) = history.BestEpoch;
        status(c) = "ok";
        if chosen == 0 || valid_loss(c) < valid_loss(chosen)
            chosen = c;
            fit = struct("risk", risk, "model_param", model_param, "history", history, ...
                "dataset", dataset, "parameter", parameter);
        end
    catch exception
        if ~startsWith(exception.identifier, "PPIxGPN:")
            rethrow(exception);
        end
        status(c) = string(exception.identifier) + ": " + string(exception.message);
    end
    if options.verbose
        fprintf("Candidate %d/%d: gamma %g, phi_min %g, best epoch %d, validation loss %.6f (%s)\n", ...
            c, num_candidate, candidate(c, 1), candidate(c, 2), best_epoch(c), valid_loss(c), status(c));
    end
end
if chosen == 0
    error("PPIxGPN:NoCandidate", "Every candidate configuration failed; see the status messages.");
end

result.Selection = table(candidate(:, 1), candidate(:, 2), best_epoch, valid_loss, (1:num_candidate).' == chosen, status, ...
    'VariableNames', {'gamma', 'phi_min', 'BestEpoch', 'ValidLoss', 'Selected', 'Status'});
result.Selected = struct("gamma", candidate(chosen, 1), "phi_min", candidate(chosen, 2), ...
    "BestEpoch", best_epoch(chosen), "ValidLoss", valid_loss(chosen));
result.ModelParam = fit.model_param;

scoring = fit.dataset;
scoring.Xtest = data.Xdata;
[all_risk, all_effect] = RiskPredict(scoring, fit.parameter, fit.model_param);
[Uppi, Bset] = ResizeParam(fit.model_param, [num_protein, 1; num_protein, num_target]);

split_name = ["train"; "valid"; "test"];
split_column = repelem(split_name, num_target + 1);
target_column = repmat([data.target, "Mean"].', numel(split_name), 1);
metric_value = zeros(numel(split_column), 4);
for s = 1:numel(split_name)
    member = data.split == split_name(s);
    metrics = EvaluateRisk(all_risk(member, :), data.Ydata(member, :), "cutoff", options.cutoff, "target", data.target);
    metric_value((s - 1) * (num_target + 1) + (1:num_target + 1), :) = metrics{:, :};
    if split_name(s) == "test"
        result.TestMetrics = metrics;
    end
end
result.Metrics = [table(split_column, target_column, 'VariableNames', {'Split', 'Target'}), ...
    array2table(metric_value, 'VariableNames', {'AUROC', 'AUPRC', 'Accuracy', 'F1'})];

result.Parameters = [table(data.protein.', Uppi, 'VariableNames', {'Protein', 'phi'}), ...
    array2table(Bset, 'VariableNames', cellstr("theta_" + data.target)), ...
    table(mean(data.Xdata, 2), mean(all_effect, 2), 'VariableNames', {'IndependentMean', 'SynergeticMean'})];
result.Predictions = [table(data.participant, data.split, 'VariableNames', {'Participant', 'Split'}), ...
    array2table(data.Ydata, 'VariableNames', cellstr("Y_" + data.target)), ...
    array2table(all_risk, 'VariableNames', cellstr("Risk_" + data.target))];
result.History = [table((1:size(fit.history.LossTrain, 1)).', 'VariableNames', {'Epoch'}), ...
    array2table(fit.history.LossTrain, 'VariableNames', cellstr("LossTrain_" + data.target)), ...
    array2table(fit.history.LossValid, 'VariableNames', cellstr("LossValid_" + data.target))];

if options.cv_repeat > 0
    train = struct("epoch", options.epoch, "rate", options.rate, "gamma", result.Selected.gamma, ...
        "phi_min", result.Selected.phi_min, "gradient", options.gradient);
    result.CV = CrossValidate(data, "repeat", options.cv_repeat, "fold", options.cv_fold, "seed", options.seed, ...
        "train", train, "cutoff", options.cutoff, "verbose", options.verbose);
end
result.Options = options;
result.Source = data.source;
result.ElapsedSeconds = toc(timer);

if options.verbose
    fprintf("Selected gamma %g and phi_min %g (best epoch %d, validation loss %.6f).\n", result.Selected.gamma, ...
        result.Selected.phi_min, result.Selected.BestEpoch, result.Selected.ValidLoss);
    disp(result.TestMetrics);
end

if ~isempty(options.output)
    WriteOutputs(result, options.output);
end

end


function WriteOutputs(result, folder)

if ~isfolder(folder)
    mkdir(folder);
end
writetable(result.Selection, fullfile(folder, "selection.csv"));
writetable(result.Metrics, fullfile(folder, "metrics.csv"));
writetable(result.Parameters, fullfile(folder, "parameters.csv"));
writetable(result.Predictions, fullfile(folder, "predictions.csv"));
writetable(result.History, fullfile(folder, "history.csv"));
test_metrics = result.TestMetrics;
test_metrics.Target = string(test_metrics.Properties.RowNames);
summary = struct("Source", result.Source, "Selected", result.Selected, ...
    "TestMetrics", table2struct(test_metrics(:, ["Target", "AUROC", "AUPRC", "Accuracy", "F1"])), ...
    "Epoch", result.Options.epoch, "Rate", result.Options.rate, "Gradient", result.Options.gradient, ...
    "Seed", result.Options.seed, "Cutoff", result.Options.cutoff, "MATLABVersion", string(version), ...
    "ElapsedSeconds", result.ElapsedSeconds);
if isfield(result, "CV")
    writetable(result.CV.FoldMetrics, fullfile(folder, "cv_fold_metrics.csv"));
    writetable(result.CV.Summary, fullfile(folder, "cv_summary.csv"), "WriteRowNames", true);
    summary.CVRepeat = result.Options.cv_repeat;
    summary.CVFold = result.Options.cv_fold;
end
handle = fopen(fullfile(folder, "summary.json"), "w", "n", "UTF-8");
closer = onCleanup(@() fclose(handle));
fprintf(handle, "%s", jsonencode(summary, "PrettyPrint", true));

end
