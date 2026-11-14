function [V, lat, lon] = quickmap_numeric(layer, bbox, width, height)
%QUICKMAP_NUMERIC Real-valued raster from LROC QuickMap (Lunaserv WMS at wms.im-ldi.com).
%   [V, lat, lon] = quickmap_numeric('gld100', [lon0 lat0 lon1 lat1], 1440, 720)
%   layer : 'gld100'  -> GLD100 stereo DTM, planetary radius [m]
%           'freeair' -> GRAIL free-air gravity anomaly [mGal]
%           'tio2'    -> TiO2 abundance [wt%]
%           or any Lunaserv numeric layer name
%   bbox  : [lon0 lat0 lon1 lat1], longitudes -180..180
%   V (height x width), lat north->south, lon west->east; nodata -> NaN.
%   The server returns corner-registered pixels (pixel i at lon0 + i*dlon).

if nargin < 2 || isempty(bbox), bbox = [-180 -90 180 90]; end
if nargin < 3, width = 1440; end
if nargin < 4, height = 720; end
names = struct('gld100', 'luna_wac_dtm_numeric_meters_absolute', ...
               'freeair', 'luna_grail_freeair_numeric', 'tio2', 'luna_wac_tio2_numeric');
if isfield(names, layer), layer = names.(layer); end

params = {'SERVICE','WMS', 'VERSION','1.1.1', 'REQUEST','GetMap', 'SRS','EPSG:4326', 'STYLES','', ...
          'LAYERS',layer, 'FORMAT','image/tiff; mode=32bit', 'WIDTH',sprintf('%d', width), ...
          'HEIGHT',sprintf('%d', height), 'BBOX',sprintf('%g,%g,%g,%g', bbox)};
file = nasa_get('quickmap', 'https://wms.im-ldi.com/', params, '.tif');

fid = fopen(file, 'r'); head = fread(fid, 5, '*char').'; fclose(fid);
if strcmp(head, '<?xml')
    error('quickmap:wms', 'QuickMap WMS error: %s', fileread(file));
end
t = Tiff(file, 'r');
V = double(read(t));
close(t);
if ndims(V) == 3, V = V(:,:,1); end
V(abs(V) > 1e30) = NaN;

lat = bbox(4) - (0:height-1).' * (bbox(4) - bbox(2)) / height;
lon = bbox(1) + (0:width-1) * (bbox(3) - bbox(1)) / width;
end
