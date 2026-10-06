function varargout = ResizeParam(param_data, param_size)

arguments
    param_data (:,1) double
    param_size (:,2) double {mustBeInteger, mustBeNonnegative}
end

num_element = prod(param_size, 2);
if sum(num_element) ~= numel(param_data)
    error("PPIxGPN:SizeMismatch", "param_size describes %d elements, but param_data has %d.", sum(num_element), numel(param_data));
end
if nargout > numel(num_element)
    error("PPIxGPN:SizeMismatch", "param_size defines %d parameter blocks, but %d outputs were requested.", numel(num_element), nargout);
end

offset = [0; cumsum(num_element)];
varargout = cell(1, max(nargout, min(1, numel(num_element))));
for k = 1:numel(varargout)
    varargout{k} = reshape(param_data(offset(k) + 1:offset(k + 1)), param_size(k, :));
end

end
