function fig = moon_viewer(varargin)
%MOON_VIEWER Interactive 3D Moon from NASA data (LOLA, Moon Trek, QuickMap, Horizons).
%
%   moon_viewer                                   % photo-textured Moon (Moon Trek WAC mosaic)
%   moon_viewer('date', '2026-11-24T12:00')       % real sunlight / phase from JPL Horizons
%   moon_viewer('color', 'tio2', 'labels', true)  % QuickMap TiO2 + Moon Trek feature names
%   moon_viewer('color', 'freeair')               % GRAIL free-air gravity (QuickMap)
%   moon_viewer('region', [-45 -41.5 -13.5 -9], 'source', 'gld100', 'exaggeration', 2)  % Tycho
%
%   Options (name/value):
%     ppd          4      LOLA grid (4, 16, 64 pixels/degree)
%     stride       2      mesh sub-sampling
%     exaggeration 3      vertical relief exaggeration
%     color        'wac'  'wac' | 'elevation' | 'slope' | 'freeair' | 'tio2'
%     date         ''     UTC time for sunlight (ISO string), '' = studio lighting
%     labels       false  IAU names from Moon Trek
%     region       []     [lat_min lat_max lon_min lon_max] (lon -180..360)
%     source       'lola' terrain for regions: 'lola' or 'gld100' (QuickMap, 128 ppd)
%
%   Mouse: drag = rotate, scroll = zoom. Slider = relief, menu = colour layer.

p = inputParser;
addParameter(p, 'ppd', 4);          addParameter(p, 'stride', 2);
addParameter(p, 'exaggeration', 3); addParameter(p, 'color', 'wac');
addParameter(p, 'date', '');        addParameter(p, 'labels', false);
addParameter(p, 'region', []);      addParameter(p, 'source', 'lola');
parse(p, varargin{:}); o = p.Results;
C = lunar_constants();

% ---------------------------------------------------------------- geometry
[lat, lon, H, geoLabel] = load_geometry(o);
[LON, LAT] = meshgrid(lon, lat);                     % degrees, lon may be 0..360 or -180..180
LONw = mod(LON + 180, 360) - 180;                    % -180..180 for web-service grids

% ---------------------------------------------------------------- figure
fig = figure('Name', 'Moon - NASA LOLA / Moon Trek / QuickMap / Horizons', 'Color', 'k', 'NumberTitle', 'off');
ax  = axes(fig, 'Color', 'k', 'Position', [0 0.1 0.86 0.83]);
hs  = surf(ax, zeros(2), zeros(2), zeros(2), 'EdgeColor', 'none', 'FaceLighting', 'gouraud', ...
           'AmbientStrength', 0.4, 'DiffuseStrength', 0.8, 'SpecularStrength', 0.02);
axis(ax, 'equal', 'off'); hold(ax, 'on');
cb = colorbar(ax, 'Color', 'w');
hl = light(ax, 'Style', 'infinite', 'Position', [-0.4 -1 0.3]);

state.exag = o.exaggeration;
state.color = o.color;
redraw_geometry();
redraw_color();

% ---------------------------------------------------------------- sunlight from Horizons
if ~isempty(o.date)
    jd = horizons_time(o.date);
    R  = lunar_orientation(jd);
    sun   = horizons_vectors('sun',   jd - 0.1, jd + 0.1, '1h');
    earth = horizons_vectors('earth', jd - 0.1, jd + 0.1, '1h');
    s_me = R * sun.at(jd).';   s_me = s_me / norm(s_me);
    e_me = R * earth.at(jd).'; e_me = e_me / norm(e_me);
    hl.Position = s_me.';
    set(hs, 'AmbientStrength', 0.04, 'DiffuseStrength', 1.0);
    if isempty(o.region)
        view(ax, atan2d(e_me(1), -e_me(2)), asind(e_me(3)));      % look from Earth
    end
    title(ax, sprintf('%s  |  Sun & Earth from JPL Horizons at %s UTC', geoLabel, o.date), ...
          'Color', 'w', 'Interpreter', 'none');
else
    view(ax, 90, 10);
    title(ax, geoLabel, 'Color', 'w', 'Interpreter', 'none');
end
if ~isempty(o.region), view(ax, mean(o.region(3:4)) + 90, 35); end

% ---------------------------------------------------------------- labels
if o.labels
    F = moontrek_features();
    if ~isempty(o.region)
        in = [F.lat] >= o.region(1) & [F.lat] <= o.region(2) & ...
             mod([F.lon] - o.region(3), 360) <= mod(o.region(4) - o.region(3), 360);
        F = F(in);
    end
    rr = (C.R_lola + max(H(:))*state.exag + 3e3 + 20e3*isempty(o.region)) / 1000;
    for k = 1:numel(F)
        [x, y, z] = sph2cart(deg2rad(F(k).lon), deg2rad(F(k).lat), rr);
        col = [1 1 1];
        if any(strcmp(F(k).type, {'Mare', 'Oceanus'})), col = [0.6 0.82 1]; end
        if strcmp(F(k).type, 'Crewed Landing'), col = [1 0.82 0.4]; end
        plot3(ax, x, y, z, '.', 'Color', col, 'MarkerSize', 8);
        text(ax, x, y, z, ['  ' F(k).name], 'Color', col, 'FontSize', 8, 'Clipping', 'on');
    end
end

% ---------------------------------------------------------------- interaction & controls
try
    ax.Interactions = [rotateInteraction zoomInteraction];
    enableDefaultInteractivity(ax);
catch
    rotate3d(fig, 'on');
end
uicontrol(fig, 'Style','text', 'Units','normalized', 'Position',[0.01 0.03 0.10 0.04], ...
    'String','Relief x', 'BackgroundColor','k', 'ForegroundColor','w');
uicontrol(fig, 'Style','slider', 'Units','normalized', 'Position',[0.11 0.03 0.25 0.04], ...
    'Min',0, 'Max',20, 'Value',state.exag, 'Callback',@(s,~) setExag(s.Value));
layers = {'wac','elevation','slope','freeair','tio2'};
uicontrol(fig, 'Style','popupmenu', 'Units','normalized', 'Position',[0.40 0.03 0.18 0.04], ...
    'String',layers, 'Value',find(strcmp(layers, state.color)), ...
    'Callback',@(s,~) setColor(s.String{s.Value}));

    function setExag(v), state.exag = v; redraw_geometry(); end
    function setColor(c), state.color = c; redraw_color(); end

    function redraw_geometry()
        Rk = (C.R_lola + state.exag*H) / 1000;
        [x, y, z] = sph2cart(deg2rad(LON), deg2rad(LAT), Rk);
        set(hs, 'XData', x, 'YData', y, 'ZData', z);
    end

    function redraw_color()
        bb = [];
        if ~isempty(o.region)
            l0 = mod(o.region(3) + 180, 360) - 180; l1 = mod(o.region(4) + 180, 360) - 180;
            bb = [l0 - 0.2, o.region(1) - 0.2, l1 + 0.2, o.region(2) + 0.2];
        end
        cb.Visible = 'on';
        switch state.color
            case 'wac'
                lvl = 3;
                if ~isempty(o.region)
                    lvl = min(7, max(3, ceil(log2(size(H,2)/max(1e-6, abs(lon(end)-lon(1))) * 180/256))));
                end
                [img, ilat, ilon] = moontrek_mosaic(lvl, bb);
                set(hs, 'CData', sampleGrid(img, ilat, ilon), 'FaceColor', 'interp');
                colormap(ax, gray(256)); clim_(ax, [0 1]); cb.Visible = 'off';
            case 'elevation'
                set(hs, 'CData', H/1000, 'FaceColor', 'interp');
                colormap(ax, gray(256)); clim_(ax, 'auto'); cb.Label.String = 'Elevation [km]';
            case 'slope'
                set(hs, 'CData', slopeDeg(H, lat, lon, C.R_lola), 'FaceColor', 'interp');
                colormap(ax, hot(256)); clim_(ax, [0 40]); cb.Label.String = 'Slope [deg]';
            case {'freeair', 'tio2'}
                if isempty(bb), bb = [-180 -90 180 90]; w = 1440; h = 720; else, w = 800; h = 800; end
                [V, qlat, qlon] = quickmap_numeric(state.color, bb, w, h);
                Vm = sampleGrid(V, qlat, qlon);
                set(hs, 'CData', Vm, 'FaceColor', 'interp');
                if strcmp(state.color, 'tio2')
                    if exist('turbo', 'file'), cm = turbo(256); else, cm = jet(256); end
                    colormap(ax, cm); clim_(ax, [0 12]); cb.Label.String = 'TiO_2 [wt%] (QuickMap)';
                else
                    colormap(ax, redblue(256)); clim_(ax, [-400 400]);
                    cb.Label.String = 'GRAIL free-air anomaly [mGal] (QuickMap)';
                end
        end
    end

    function out = sampleGrid(img, ilat, ilon)
        % bilinear sample of a lat/lon raster (lat descending, lon -180..180) at the mesh
        img = double(img);
        img(isnan(img)) = mean(img(:), 'omitnan');
        if ilat(1) > ilat(end), ilat = flipud(ilat(:)); img = flipud(img); end
        [IL, IA] = meshgrid(ilon, ilat);
        if isempty(o.region)                     % wrap across the date line
            IL = [IL(:, end) - 360, IL, IL(:, 1) + 360];
            IA = [IA(:, end), IA, IA(:, 1)];
            img = [img(:, end), img, img(:, 1)];
        end
        out = interp2(IL, IA, img, LONw, LAT, 'linear', NaN);
        out(isnan(out)) = mean(img(:));
    end
end

% ======================================================================
function [lat, lon, H, label] = load_geometry(o)
C = lunar_constants();
if ~isempty(o.region) && strcmpi(o.source, 'gld100')
    l0 = mod(o.region(3) + 180, 360) - 180; l1 = mod(o.region(4) + 180, 360) - 180;
    w  = min(2000, max(64, round((l1 - l0)*128)));
    h  = min(2000, max(64, round((o.region(2) - o.region(1))*128)));
    [R, lat, lon] = quickmap_numeric('gld100', [l0 o.region(1) l1 o.region(2)], w, h);
    R = R(1:o.stride:end, 1:o.stride:end);
    lat = lat(1:o.stride:end); lon = lon(1:o.stride:end);
    H = R - C.R_lola;
    H(isnan(H)) = mean(H(:), 'omitnan');
    label = sprintf('LROC GLD100 via QuickMap (%dx%d)', numel(lon), numel(lat));
    return
end
dem = lola_read(o.ppd);
lat = dem.lat; lon = dem.lon; H = dem.height;
if ~isempty(o.region)
    lo0 = mod(o.region(3), 360); lo1 = mod(o.region(4), 360);
    rows = find(lat >= o.region(1) & lat <= o.region(2));
    if lo0 <= lo1, cols = find(lon >= lo0 & lon <= lo1);
    else,          cols = [find(lon >= lo0), find(lon <= lo1)]; end
    rows = rows(1:o.stride:end); cols = cols(1:o.stride:end);
    H = H(rows, cols); lat = lat(rows); lon = lon(cols);
else
    rows = 1:o.stride:numel(lat); cols = 1:o.stride:numel(lon);
    H = H(rows, cols); lat = lat(rows); lon = lon(cols);
    H = [H, H(:,1)]; lon = [lon, lon(1) + 360];                    % close seam
    H = [repmat(mean(H(1,:)), 1, size(H,2)); H; repmat(mean(H(end,:)), 1, size(H,2))];
    lat = [90; lat; -90];                                           % cap poles
end
label = sprintf('NASA LOLA LDEM_%d', o.ppd);
end

function s = slopeDeg(H, lat, lon, R)
[dHdlon, dHdlat] = gradient(H);
dlat = gradient(deg2rad(lat(:)));
dlon = gradient(deg2rad(lon(:).'));
s = atand(hypot(dHdlat ./ dlat / R, dHdlon ./ dlon ./ (R * max(cosd(lat(:)), 1e-3))));
end

function clim_(ax, v)
% caxis/clim compatible with older and newer MATLAB releases
if exist('clim', 'file'), clim(ax, v); else, caxis(ax, v); end %#ok<CAXIS>
end

function c = redblue(n)
% blue -> white -> red diverging colormap
x = linspace(0, 1, n).';
c = [min(1, 2*x), 1 - abs(2*x - 1), min(1, 2 - 2*x)];
end
