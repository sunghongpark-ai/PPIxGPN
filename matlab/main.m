function result = main(varargin)

originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));
addpath(fullfile(fileparts(mfilename('fullpath')), 'model'));
result = RunSample(varargin{:});

end
