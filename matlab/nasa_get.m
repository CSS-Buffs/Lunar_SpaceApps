function file = nasa_get(subdir, base, params, ext, opts)
%NASA_GET Download a NASA web-service response once and cache it under data/<subdir>.
%   file = nasa_get(subdir, base, params, ext)
%   params : cell array {'NAME1', value1, 'NAME2', value2, ...} (query string) or {}
%   ext    : file extension for the cache file, e.g. '.json', '.tif', '.jpg'
%   opts   : optional weboptions
%   Returns the path of the local file.

if nargin < 5, opts = weboptions('Timeout', 600); end
if isempty(params)
    url = base;
else
    url = char(matlab.net.URI(base, matlab.net.QueryParameter(params{:})));
end
md  = java.security.MessageDigest.getInstance('SHA-1');
h   = typecast(md.digest(uint8(url)), 'uint8');
key = lower(reshape(dec2hex(h(1:8)).', 1, []));

dataDir = fullfile(fileparts(mfilename('fullpath')), '..', 'data', subdir);
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
file = fullfile(dataDir, [key ext]);
info = dir(file);
if ~isempty(info) && info.bytes > 0
    return
end
tmp = [file '.part'];
for attempt = 1:3
    try
        websave(tmp, url, opts);
        movefile(tmp, file, 'f');
        return
    catch err
        if attempt == 3
            error('nasa_get:failed', 'Download failed: %s\n%s', url, err.message);
        end
        pause(2*attempt);
    end
end
end
