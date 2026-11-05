function fig = moon_viewer(ppd, stride, exaggeration, region)
%MOON_VIEWER Interactive 3D Moon from NASA LRO/LOLA laser-altimeter data.
%   moon_viewer                       % LDEM_4, 0.5 deg mesh, 3x relief
%   moon_viewer(4, 1, 5)              % full LDEM_4 resolution, 5x relief
%   moon_viewer(16, 1, 2, [-50 -35 340 355])   % Tycho close-up from LDEM_16
%
%   Mouse: drag = rotate, scroll = zoom, double-click = reset view.
%   The controls at the bottom change relief exaggeration, colouring and sun angle.

if nargin < 1 || isempty(ppd),          ppd = 4;          end
if nargin < 2 || isempty(stride),       stride = 2;       end
if nargin < 3 || isempty(exaggeration), exaggeration = 3; end
if nargin < 4,                          region = [];      end

dem = lola_read(ppd);
C   = lunar_constants();

% ---- choose grid ---------------------------------------------------------
lat = dem.lat; lon = dem.lon; H = dem.height;
if isempty(region)
    rows = 1:stride:numel(lat);
    cols = 1:stride:numel(lon);
    H   = H(rows, cols);
    lat = lat(rows);
    lon = lon(cols);
    % close seam and poles -> watertight sphere
    H   = [H, H(:,1)];
    lon = [lon, lon(1) + 360];
    H   = [repmat(mean(H(1,:)), 1, size(H,2)); H; repmat(mean(H(end,:)), 1, size(H,2))];
    lat = [90; lat; -90];
else
    lo0 = mod(region(3), 360); lo1 = mod(region(4), 360);
    rows = find(lat >= region(1) & lat <= region(2));
    if lo0 <= lo1
        cols = find(lon >= lo0 & lon <= lo1);
    else
        cols = [find(lon >= lo0), find(lon <= lo1)];
    end
    rows = rows(1:stride:end); cols = cols(1:stride:end);
    H = H(rows, cols); lat = lat(rows); lon = lon(cols);
end
[LON, LAT] = meshgrid(deg2rad(lon), deg2rad(lat));

% ---- precompute colour fields -------------------------------------------
fields.Elevation = H / 1000;                              % km
[r0x, r0y, r0z] = sph2cart(LON, LAT, C.R_lola + H);
[g, ~] = lunar_gravity([r0x(:) r0y(:) r0z(:)].', false);
g(1,:) = g(1,:) + C.omega^2 * r0x(:).';
g(2,:) = g(2,:) + C.omega^2 * r0y(:).';
fields.Gravity = reshape(vecnorm(g), size(H));            % m/s^2
fields.Slope   = slopeDeg(H, lat, lon, C.R_lola);         % deg
labels = struct('Elevation','Elevation [km]', 'Gravity','|g| surface [m/s^2]', 'Slope','Slope [deg]');

% ---- figure --------------------------------------------------------------
fig = figure('Name', 'Moon - NASA LRO LOLA', 'Color', 'k', 'NumberTitle', 'off');
ax  = axes(fig, 'Color', 'k', 'Position', [0 0.1 0.88 0.85]);
hs  = surf(ax, zeros(2), zeros(2), zeros(2), 'EdgeColor', 'none', 'FaceColor', 'interp', ...
           'AmbientStrength', 0.35, 'DiffuseStrength', 0.9, ...
           'SpecularStrength', 0.05, 'FaceLighting', 'gouraud');
axis(ax, 'equal', 'off'); hold(ax, 'on');
colormap(ax, gray(256));
cb = colorbar(ax, 'Color', 'w'); cb.Label.String = labels.Elevation;
hl = light(ax, 'Style', 'infinite');
title(ax, sprintf('The Moon - LOLA LDEM\\_%d  (%dx%d mesh)', ppd, size(H,2), size(H,1)), 'Color', 'w');
view(ax, 3);

state.field = 'Elevation';
state.exag  = exaggeration;
state.sun   = -60;
redraw();

% ---- interaction: rotate with drag, zoom with scroll ---------------------
try
    ax.Interactions = [rotateInteraction zoomInteraction];
    enableDefaultInteractivity(ax);
catch
    rotate3d(fig, 'on');            % pre-R2019a fallback
end

% ---- controls -------------------------------------------------------------
uicontrol(fig, 'Style','text', 'Units','normalized', 'Position',[0.01 0.03 0.12 0.04], ...
    'String','Relief x', 'BackgroundColor','k', 'ForegroundColor','w');
uicontrol(fig, 'Style','slider', 'Units','normalized', 'Position',[0.13 0.03 0.22 0.04], ...
    'Min',0, 'Max',20, 'Value',exaggeration, 'Callback',@(s,~) setState('exag', s.Value));
uicontrol(fig, 'Style','popupmenu', 'Units','normalized', 'Position',[0.38 0.03 0.15 0.04], ...
    'String',{'Elevation','Gravity','Slope'}, 'Callback',@(s,~) setState('field', s.String{s.Value}));
uicontrol(fig, 'Style','text', 'Units','normalized', 'Position',[0.56 0.03 0.10 0.04], ...
    'String','Sun lon', 'BackgroundColor','k', 'ForegroundColor','w');
uicontrol(fig, 'Style','slider', 'Units','normalized', 'Position',[0.66 0.03 0.22 0.04], ...
    'Min',-180, 'Max',180, 'Value',state.sun, 'Callback',@(s,~) setState('sun', s.Value));

    function setState(name, val)
        state.(name) = val;
        redraw();
    end

    function redraw()
        R = (C.R_lola + state.exag * H) / 1000;           % km
        [x, y, z] = sph2cart(LON, LAT, R);
        set(hs, 'XData', x, 'YData', y, 'ZData', z, 'CData', fields.(state.field));
        switch state.field
            case 'Elevation', colormap(ax, gray(256));
            case 'Gravity',   colormap(ax, parula(256));
            case 'Slope',     colormap(ax, hot(256));
        end
        cb.Label.String = labels.(state.field);
        [lx, ly, lz] = sph2cart(deg2rad(state.sun), deg2rad(10), 1);
        hl.Position = [lx ly lz];
    end
end

function s = slopeDeg(H, lat, lon, R)
dlat = gradient(deg2rad(lat(:)));
dlon = gradient(deg2rad(lon(:).'));
[dHdlon, dHdlat] = gradient(H);
dHdlat = dHdlat ./ dlat / R;
coslat = max(cosd(lat(:)), 1e-3);
dHdlon = dHdlon ./ dlon ./ (R * coslat);
s = atand(hypot(dHdlat, dHdlon));
end
