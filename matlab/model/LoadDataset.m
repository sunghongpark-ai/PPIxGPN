function data = LoadDataset(file)

arguments
    file string {mustBeScalarOrEmpty} = string.empty
end

if isempty(file)
    file = fullfile(fileparts(fileparts(fileparts(mfilename("fullpath")))), "dataset", "sample.csv");
end
if ~isfile(file)
    error("PPIxGPN:MissingFile", "Dataset file does not exist: %s", file);
end

target = ["Abeta", "GFAP", "NfL", "pTau"];
fixed = ["record", "id", "split", "Y_" + target];
handle = fopen(file, "r", "n", "UTF-8");
if handle < 0
    error("PPIxGPN:InvalidDataset", "Cannot open dataset file: %s", file);
end
header_line = fgetl(handle);
fclose(handle);
if ~ischar(header_line)
    error("PPIxGPN:InvalidDataset", "Dataset file is empty: %s", file);
end
header = strip(strip(split(erase(string(header_line), char(65279)), ",")), "both", '"').';
if numel(header) <= numel(fixed) || ~isequal(header(1:numel(fixed)), fixed)
    error("PPIxGPN:InvalidDataset", "The header must start with %s and continue with protein columns.", strjoin(fixed, ","));
end
protein = header(numel(fixed) + 1:end);
if numel(unique(protein)) ~= numel(protein) || any(protein == "")
    error("PPIxGPN:InvalidDataset", "Protein column names must be unique and nonempty.");
end

options = delimitedTextImportOptions("NumVariables", numel(header), "Delimiter", ",", "Encoding", "UTF-8", ...
    "DataLines", [2, Inf], "VariableNamingRule", "preserve", "ExtraColumnsRule", "error", ...
    "EmptyLineRule", "skip");
options.VariableNames = cellstr(header);
options = setvartype(options, 1:3, "string");
options = setvartype(options, 4:numel(header), "double");
try
    content = readtable(file, options);
catch cause
    exception = MException("PPIxGPN:InvalidDataset", "Cannot read %s: %s", file, cause.message);
    throw(addCause(exception, cause));
end

record = content{:, 1};
is_participant = record == "participant";
is_ppi = record == "ppi";
if ~all(is_participant | is_ppi)
    error("PPIxGPN:InvalidDataset", "The record column may contain only participant and ppi.");
end
if ~any(is_participant)
    error("PPIxGPN:InvalidDataset", "The dataset contains no participant rows.");
end

participant = content{is_participant, 2};
if any(ismissing(participant) | participant == "") || numel(unique(participant)) ~= numel(participant)
    error("PPIxGPN:InvalidDataset", "Participant identifiers must be unique and nonempty.");
end
partition = lower(content{is_participant, 3});
if ~all(ismember(partition, ["train", "valid", "test"]))
    error("PPIxGPN:InvalidDataset", "Each participant split must be train, valid, or test.");
end
Ydata = content{is_participant, 4:numel(fixed)};
if ~all(Ydata == 0 | Ydata == 1, "all")
    error("PPIxGPN:InvalidDataset", "Diagnosis labels must be 0 or 1 for every participant.");
end
Xdata = content{is_participant, numel(fixed) + 1:end}.';
if ~all(isfinite(Xdata), "all")
    error("PPIxGPN:InvalidDataset", "Protein expression must be finite for every participant.");
end

network_id = content{is_ppi, 2};
[found, location] = ismember(protein, network_id);
if numel(network_id) ~= numel(protein) || ~all(found) || numel(unique(network_id)) ~= numel(network_id)
    error("PPIxGPN:InvalidDataset", "Provide exactly one ppi row for each protein column.");
end
ppi_data = content{is_ppi, numel(fixed) + 1:end};
ppi_data = ppi_data(location, :);
if ~all(isfinite(ppi_data), "all")
    error("PPIxGPN:InvalidDataset", "PPI scores must be finite.");
end

data = struct("Xdata", Xdata, "Ydata", Ydata, "ppi_data", ppi_data, ...
    "protein", protein, "participant", participant, "split", partition, "target", target, ...
    "idx_train", find(partition == "train"), "idx_valid", find(partition == "valid"), ...
    "idx_test", find(partition == "test"), "source", string(file));

end
