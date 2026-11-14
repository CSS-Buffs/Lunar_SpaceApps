function dem = lola_read(ppd)
%LOLA_READ Download (if needed) and load a NASA LRO/LOLA global DEM.
%   dem = lola_read(ppd)   ppd = 4 (2 MB), 16 (33 MB) or 64 (531 MB) pixels/degree
%
%   Source: NASA PDS Geosciences Node, LOLA Gridded Data Records (LDEM_<ppd>.IMG)
%   Encoding: int16 little-endian, HEIGHT [m] = DN*0.5,
%             RADIUS [m] = DN*0.5 + 1737400. Row 1 = 90N, column 1 starts at 0E.
%
%   Returns struct with fields
%     height  (lines x samples) height above 1737.4 km sphere [m]
%     lat     (lines x 1)  pixel-centre latitude  [deg], north -> south
%     lon     (1 x samples) pixel-centre east longitude [deg], 0 -> 360
%     R_ref   1737400 [m]
%     ppd     pixels per degree
%     interp  @(lat,lon) bilinear height lookup [m] (handles longitude wrap)

if nargin < 1, ppd = 4; end
assert(any(ppd == [4 16 64]), 'ppd must be 4, 16 or 64');

base    = 'https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/';
dataDir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
imgFile = fullfile(dataDir, sprintf('ldem_%d.img', ppd));
lblFile = fullfile(dataDir, sprintf('ldem_%d.lbl', ppd));

opts = weboptions('Timeout', 600);
if ~isfile(lblFile)
    fprintf('Downloading %s\n', [base sprintf('ldem_%d.lbl', ppd)]);
    websave(lblFile, [base sprintf('ldem_%d.lbl', ppd)], opts);
end
if ~isfile(imgFile)
    fprintf('Downloading %s (this can take a while)\n', [base sprintf('ldem_%d.img', ppd)]);
    websave(imgFile, [base sprintf('ldem_%d.img', ppd)], opts);
end

% --- parse the PDS label -------------------------------------------------
lbl     = fileread(lblFile);
lines   = labelNum(lbl, 'LINES');
samples = labelNum(lbl, 'LINE_SAMPLES');
scale   = labelNum(lbl, 'SCALING_FACTOR');
offset  = labelNum(lbl, 'OFFSET');

% --- read binary image ---------------------------------------------------
fid = fopen(imgFile, 'r', 'ieee-le');
assert(fid > 0, 'Cannot open %s', imgFile);
dn = fread(fid, [samples, lines], 'int16=>single');
fclose(fid);
assert(numel(dn) == lines*samples, 'Truncated file %s: delete it and re-run', imgFile);

dem.height = double(dn.') * scale;            % lines x samples
dem.lat    = 90 - ((1:lines).' - 0.5) / ppd;
dem.lon    = ((1:samples) - 0.5) / ppd;
dem.R_ref  = offset;
dem.ppd    = ppd;

% Interpolant on an ascending-latitude grid with one wrapped column each side
latAsc = flipud(dem.lat);
H      = flipud(dem.height);
lonPad = [dem.lon(end) - 360, dem.lon, dem.lon(1) + 360];
Hpad   = [H(:, end), H, H(:, 1)];
F = griddedInterpolant({latAsc, lonPad}, Hpad, 'linear', 'nearest');
dem.interp = @(lat, lon) F(lat, mod(lon, 360));
end

function v = labelNum(txt, key)
tok = regexp(txt, ['^\s*' key '\s*=\s*([-+0-9.Ee]+)'], 'tokens', 'once', 'lineanchors');
assert(~isempty(tok), 'Key %s not found in label', key);
v = str2double(tok{1});
end
