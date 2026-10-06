function [readout, adjoint, solver] = PropagationReadout(Lppi, Uppi, Bset)

arguments
    Lppi (:,:) double
    Uppi (:,1) double
    Bset (:,:) double
end

num_protein = numel(Uppi);
solver = decomposition(Lppi + spdiags(Uppi, 0, num_protein, num_protein));
if isIllConditioned(solver)
    error("PPIxGPN:IllConditionedSystem", "The propagation system Lppi + diag(Uppi) is singular to working precision.");
end
adjoint = (Bset.' / solver).';
readout = Uppi .* adjoint;

end
