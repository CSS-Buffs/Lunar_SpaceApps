function out = moon_orbit_sim(varargin)
%MOON_ORBIT_SIM Lunar orbit simulation driven by the NASA/JPL Horizons API.
%
%   out = moon_orbit_sim('lro', true, 'epoch', '2024-08-20T00:00:00', 'days', 1)
%         % start from LRO's real state (Horizons -85) and compare with its real trajectory
%   out = moon_orbit_sim('alt', 50, 'inc', 90, 'days', 1, 'epoch', '2026-11-20T00:00:00')
%   out = moon_orbit_sim('alt', 8, 'ecc', 0.002, 'inc', 85, 'days', 3)   % ends in terrain impact
%
%   Physics
%     * GRAIL GRGM1200A spherical-harmonic gravity (grail_load / grail_accel), PA frame
%     * Moon orientation: IAU 2015 + DE440 ME->PA (lunar_orientation)
%     * Earth and Sun third-body forces from JPL Horizons ephemerides
%     * LOLA terrain: integration stops on impact (ode113 event)
%
%   Options: epoch, days, degree (60), lro (false), alt [km], ecc, inc, raan, argp [deg],
%            thirdbody (true), ppd (4), plot (true)

p = inputParser;
addParameter(p, 'epoch', '2025-06-01T00:00:00'); addParameter(p, 'days', 1);
addParameter(p, 'degree', 60);   addParameter(p, 'lro', false);
addParameter(p, 'alt', 50);      addParameter(p, 'ecc', 0);  addParameter(p, 'inc', 90);
addParameter(p, 'raan', 0);      addParameter(p, 'argp', 0);
addParameter(p, 'thirdbody', true); addParameter(p, 'ppd', 4); addParameter(p, 'plot', true);
parse(p, varargin{:}); o = p.Results;

C   = lunar_constants();
G   = grail_load(o.degree);
dem = lola_read(o.ppd);
jd0 = horizons_time(o.epoch);
tEnd = o.days * 86400;

% ---------------------------------------------------------------- initial state
truth = [];
if o.lro
    truth = horizons_vectors('lro', jd0, jd0 + o.days, '2m');
    jd0 = truth.jd(1);
    r0 = truth.pos(1,:).'; v0 = truth.vel(1,:).';
    fprintf('LRO state from JPL Horizons: |r| = %.2f km, |v| = %.2f m/s\n', norm(r0)/1e3, norm(v0));
    tspan = (truth.jd - jd0).' * 86400;
else
    a = (C.R_lola + o.alt*1e3) / (1 - o.ecc);
    [rme, vme] = kepler2state(a, o.ecc, o.inc, o.raan, o.argp, 0, G.GM);
    R = lunar_orientation(jd0);
    r0 = R.' * rme; v0 = R.' * vme;                   % lunar-equator elements -> ICRF
    tspan = linspace(0, tEnd, round(min(20000, max(1500, tEnd/60))));
end

earth = []; sun = [];
if o.thirdbody
    earth = horizons_vectors('earth', jd0 - 0.5, jd0 + o.days + 0.5, '1h');
    sun   = horizons_vectors('sun',   jd0 - 0.5, jd0 + o.days + 0.5, '1h');
end

% ---------------------------------------------------------------- integrate
maxStep = 60; if norm(r0) - C.R_lola < 40e3, maxStep = 10; end
opts = odeset('RelTol', 1e-11, 'AbsTol', 1e-5, 'MaxStep', maxStep, ...
              'Events', @(t, s) groundEvent(t, s, jd0, dem, C));
tic;
[t, S, te, Se] = ode113(@(t, s) rhs(t, s, jd0, G, earth, sun, C), tspan, [r0; v0], opts);
fprintf('Integrated %.2f days with GRAIL degree %d in %.1f s\n', o.days, o.degree, toc);
if ~isempty(te), t = [t; te(1)]; S = [S; Se(1,:)]; end
r = S(:, 1:3); v = S(:, 4:6);

% Moon-fixed coordinates and altitude above LOLA
rme = zeros(size(r));
for k = 1:numel(t)
    rme(k,:) = (lunar_orientation(jd0 + t(k)/86400) * r(k,:).').';
end
rn  = vecnorm(rme, 2, 2);
lat = asind(rme(:,3) ./ rn);
lon = mod(atan2d(rme(:,2), rme(:,1)), 360);
alt = rn - (C.R_lola + dem.interp(lat, lon));
fprintf('Altitude above LOLA terrain: %.2f .. %.2f km\n', min(alt)/1e3, max(alt)/1e3);
if ~isempty(te)
    fprintf('IMPACT after %.2f h at lat %.3f, lon %.3f E\n', te(1)/3600, lat(end), lon(end));
end

err = [];
if o.lro
    n = min(numel(t), size(truth.pos, 1));
    err = vecnorm(r(1:n,:) - truth.pos(1:n,:), 2, 2);
    fprintf('Error vs real LRO after %.1f h: max %.1f m, final %.1f m\n', t(n)/3600, max(err), err(end));
end

% ---------------------------------------------------------------- plots
if o.plot
    figure('Name', 'Orbit diagnostics');
    subplot(1 + o.lro, 1, 1);
    plot(t/3600, alt/1e3); grid on; ylabel('Altitude above LOLA [km]');
    title(sprintf('GRAIL degree %d + Earth/Sun (JPL Horizons) + LOLA terrain', o.degree));
    if o.lro
        subplot(2, 1, 2);
        semilogy(t(1:numel(err))/3600, err + 1e-2); grid on;
        ylabel('Error vs real LRO [m]');
    end
    xlabel(sprintf('Hours after %s UTC', o.epoch));

    fig = moon_viewer('exaggeration', 1, 'date', o.epoch);
    ax  = findobj(fig, 'Type', 'axes');
    plot3(ax(1), rme(:,1)/1e3, rme(:,2)/1e3, rme(:,3)/1e3, 'c', 'LineWidth', 1.2);
    if o.lro
        tme = zeros(size(truth.pos));
        for k = 1:size(truth.pos, 1)
            tme(k,:) = (lunar_orientation(truth.jd(k)) * truth.pos(k,:).').';
        end
        plot3(ax(1), tme(:,1)/1e3, tme(:,2)/1e3, tme(:,3)/1e3, 'm--', 'LineWidth', 0.8);
    end
    if ~isempty(te)
        plot3(ax(1), rme(end,1)/1e3, rme(end,2)/1e3, rme(end,3)/1e3, 'ro', 'MarkerFaceColor', 'r');
    end
end

out = struct('t', t, 'jd_tdb', jd0 + t/86400, 'r_icrf', r, 'v_icrf', v, 'r_me', rme, ...
             'lat', lat, 'lon', lon, 'alt_terrain', alt, 'impact_time', te, 'error_vs_lro', err);
end

% ======================================================================
function ds = rhs(t, s, jd0, G, earth, sun, C)
jd = jd0 + t/86400;
[Rme, Rpa] = lunar_orientation(jd);
Q  = Rpa * Rme;                                   % ICRF -> PA
r  = s(1:3);
a  = Q.' * grail_accel(Q * r, G);
if ~isempty(earth)
    a = a + thirdBody(r, earth.at(jd).', C.GM_earth) + thirdBody(r, sun.at(jd).', C.GM_sun);
end
ds = [s(4:6); a];
end

function a = thirdBody(r, rb, gm)
d = rb - r;
a = gm * (d / norm(d)^3 - rb / norm(rb)^3);
end

function [val, isterminal, direction] = groundEvent(t, s, jd0, dem, C)
rb  = lunar_orientation(jd0 + t/86400) * s(1:3);
rn  = norm(rb);
val = rn - (C.R_lola + dem.interp(asind(rb(3)/rn), mod(atan2d(rb(2), rb(1)), 360)));
isterminal = 1; direction = -1;
end

function [r, v] = kepler2state(a, e, i, O, w, nu, mu)
i = deg2rad(i); O = deg2rad(O); w = deg2rad(w); nu = deg2rad(nu);
p  = a*(1 - e^2);
rp = p/(1 + e*cos(nu)) * [cos(nu); sin(nu); 0];
vp = sqrt(mu/p) * [-sin(nu); e + cos(nu); 0];
Q = [cos(O)*cos(w)-sin(O)*sin(w)*cos(i), -cos(O)*sin(w)-sin(O)*cos(w)*cos(i),  sin(O)*sin(i);
     sin(O)*cos(w)+cos(O)*sin(w)*cos(i), -sin(O)*sin(w)+cos(O)*cos(w)*cos(i), -cos(O)*sin(i);
     sin(w)*sin(i),                       cos(w)*sin(i),                        cos(i)];
r = Q*rp; v = Q*vp;
end
