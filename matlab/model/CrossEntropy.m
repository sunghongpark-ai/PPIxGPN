function loss = CrossEntropy(logit, label)

arguments
    logit (:,:) double
    label (:,:) double
end

loss = mean(max(logit, 0) - label .* logit + log1p(exp(-abs(logit))), 1);

end
