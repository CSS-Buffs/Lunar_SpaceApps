function [img, lat, lon] = moontrek_mosaic(level, bbox)
%MOONTREK_MOSAIC LRO WAC global mosaic from NASA Moon Trek WMTS tiles.
%   [img, lat, lon] = moontrek_mosaic(3)                    % global, 4096 x 2048
%   [img, lat, lon] = moontrek_mosaic(6, [lon0 lat0 lon1 lat1])   % only tiles covering bbox
%   Level z has 2^(z+1) x 2^z tiles of 256 px over lon -180..180, lat 90..-90.
%   img : grey reflectance 0..1 (single), lat north->south, lon west->east (pixel centres).

if nargin < 1, level = 3; end
layer = 'LRO_WAC_Mosaic_Global_303ppd_v02';
nrow = 2^level; ncol = 2^(level+1); tdeg = 180/nrow;
if nargin < 2 || isempty(bbox)
    r0 = 0; r1 = nrow-1; c0 = 0; c1 = ncol-1;
else
    c0 = floor((bbox(1)+180)/tdeg); c1 = min(floor((bbox(3)+180)/tdeg), ncol-1);
    r0 = floor((90-bbox(4))/tdeg);  r1 = min(floor((90-bbox(2))/tdeg), nrow-1);
end
img = zeros((r1-r0+1)*256, (c1-c0+1)*256, 'single');
for r = r0:r1
    for c = c0:c1
        url  = sprintf('https://trek.nasa.gov/tiles/Moon/EQ/%s/1.0.0/default/default028mm/%d/%d/%d.jpg', ...
                       layer, level, r, c);
        file = nasa_get(sprintf('moontrek/%s/%d', layer, level), url, {}, '.jpg');
        t = imread(file);
        if size(t, 3) == 3, t = rgb2gray(t); end
        img((r-r0)*256 + (1:256), (c-c0)*256 + (1:256)) = single(t) / 255;
    end
end
px  = tdeg/256;
lat = 90 - r0*tdeg - ((1:size(img,1)).' - 0.5)*px;
lon = -180 + c0*tdeg + ((1:size(img,2)) - 0.5)*px;
end
