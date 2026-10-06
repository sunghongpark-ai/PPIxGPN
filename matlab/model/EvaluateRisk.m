function metrics = EvaluateRisk(pred_risk, label, options)

arguments
    pred_risk (:,:) double {mustBeReal, mustBeNonNan}
    label (:,:) double {mustBeReal}
    options.cutoff (1,1) double {mustBeInRange(options.cutoff, 0, 1)} = 0.5
    options.target (1,:) string = ["Abeta", "GFAP", "NfL", "pTau"]
end

if ~isequal(size(pred_risk), size(label))
    error("PPIxGPN:SizeMismatch", "pred_risk and label must have the same size.");
end
if ~all(label == 0 | label == 1, "all")
    error("PPIxGPN:InvalidLabel", "Labels must be 0 or 1.");
end
if numel(options.target) ~= size(label, 2)
    error("PPIxGPN:SizeMismatch", "Provide one target name per label column.");
end

num_target = size(label, 2);
value = zeros(num_target, 4);
for k = 1:num_target
    score = pred_risk(:, k);
    truth = label(:, k) == 1;
    predicted = score >= options.cutoff;
    true_positive = nnz(predicted & truth);
    false_positive = nnz(predicted & ~truth);
    false_negative = nnz(~predicted & truth);
    f1_denominator = 2 * true_positive + false_positive + false_negative;
    value(k, :) = [RankArea(score, truth), AveragePrecision(score, truth), ...
        mean(predicted == truth), (2 * true_positive) / f1_denominator];
end
metrics = array2table([value; mean(value, 1)], "VariableNames", ["AUROC", "AUPRC", "Accuracy", "F1"], ...
    "RowNames", cellstr([options.target, "Mean"]));

end


function area = RankArea(score, truth)

num_positive = nnz(truth);
num_negative = numel(truth) - num_positive;
if num_positive == 0 || num_negative == 0
    area = NaN;
    return
end
[sorted, order] = sort(score);
first = find([true; diff(sorted) ~= 0]);
last = [first(2:end) - 1; numel(sorted)];
group = cumsum([true; diff(sorted) ~= 0]);
rank = zeros(numel(score), 1);
rank(order) = (first(group) + last(group)) / 2;
area = (sum(rank(truth)) - num_positive * (num_positive + 1) / 2) / (num_positive * num_negative);

end


function precision_area = AveragePrecision(score, truth)

num_positive = nnz(truth);
if num_positive == 0
    precision_area = NaN;
    return
end
[sorted, order] = sort(score, "descend");
hit = truth(order);
boundary = [diff(sorted) ~= 0; true];
true_positive = cumsum(hit);
false_positive = cumsum(~hit);
true_positive = true_positive(boundary);
false_positive = false_positive(boundary);
precision = true_positive ./ (true_positive + false_positive);
recall = true_positive / num_positive;
precision_area = sum(diff([0; recall]) .* precision);

end
