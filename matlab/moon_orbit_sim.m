function out = moon_orbit_sim(varargin)
%MOON_ORBIT_SIM Propagate a lunar orbit over real LOLA terrain.
%   out = moon_orbit_sim('alt',50,'inc',90,'days',1)
%   out = moon_orbit_sim('alt',3,'inc',30,'days',1)     % low orbit -> terrain impact
%
%   Name/value options (defaults):
%     alt   50     periapsis altitude above 1737.4 km sphere [km]
%     ecc   0      eccentricity
%     inc   90     inclination [deg]
%     raan  0, argp 0 [deg]
%     days  1      duration
%     earth true   include Earth tidal perturbation
%     ppd   4      LOLA grid resolution
%
%   Physics: GRAIL GRGM1200A J2 + C22 gravity, Earth tide, synchronous rotation,
%   integrated with ode113 in a Moon-centred inertial frame. The Jacobi constant
%   (energy in the rotating frame) is conserved by this model, so its drift is
%   reported as a check on numerical accuracy.

p = inputParser;
addParameter(p, 'alt', 50);  addParameter(p, 'ecc', 0);  addParameter(p, 'inc', 90);
addParameter(p, 'raan', 0);  addParameter(p, 'argp', 0); addParameter(p, 'days', 1);
addParameter(p, 'earth', true); addParameter(p, 'ppd', 4);
parse(p, varargin{:}); o = p.Results;

C   = lunar_constants();
dem = lola_read(o.ppd);

rp = C.R_lola + o.alt*1e3;
a  = rp / (1 - o.ecc);
[r0, v0] = kepler2state(a, o.ecc, o.inc, o.raan, o.argp, 0, C.GM);
T = 2*pi*sqrt(a^3 / C.GM);
fprintf('Initial orbit: a = %.1f km, e = %g, i = %g deg, period = %.1f min\n', ...
        a/1e3, o.ecc, o.inc, T/60);

tEnd = o.days * 86400;
nOut = round(min(20000, max(2000, tEnd/30)));
opts = odeset('RelTol',1e-11, 'AbsTol',1e-6, 'MaxStep',10, ...
              'Events', @(t,s) groundEvent(t, s, dem, C));
[t, S, te, Se] = ode113(@(t,s) rhs(t, s, o.earth, C), linspace(0, tEnd, nOut), [r0; v0], opts);
if ~isempty(te)                      % append exact impact point
    t = [t; te(1)]; S = [S; Se(1,:)];
end
t = t.'; rin = S(:,1:3).'; vin = S(:,4:6).';

% body-fixed coordinates, lat/lon, altitude above LOLA terrain
th  = C.omega * t;
rbf = [ cos(th).*rin(1,:) + sin(th).*rin(2,:);
       -sin(th).*rin(1,:) + cos(th).*rin(2,:);
        rin(3,:)];
rn  = vecnorm(rbf);
lat = asind(rbf(3,:) ./ rn);
lon = mod(atan2d(rbf(2,:), rbf(1,:)), 360);
alt = rn - (C.R_lola + dem.interp(lat, lon));

% Jacobi constant
vrel = [vin(1,:) + C.omega*rin(2,:); vin(2,:) - C.omega*rin(1,:); vin(3,:)];
[~, U] = lunar_gravity(rbf, o.earth);
J = 0.5*sum(vrel.^2) - U - 0.5*C.omega^2*(rbf(1,:).^2 + rbf(2,:).^2);
drift = max(abs(J - J(1))) / abs(J(1));

fprintf('Max relative Jacobi drift (integration error): %.2e\n', drift);
fprintf('Altitude above terrain: min %.2f km, max %.2f km\n', min(alt)/1e3, max(alt)/1e3);
if ~isempty(te)
    fprintf('IMPACT after %.2f h at lat %.2f, lon %.2f E\n', te(1)/3600, lat(end), lon(end));
else
    fprintf('No impact during simulation.\n');
end

% ---- plots ----------------------------------------------------------------
figure('Name','Orbit diagnostics');
subplot(2,1,1); plot(t/3600, alt/1e3); grid on; ylabel('Altitude above LOLA terrain [km]');
subplot(2,1,2); plot(t/3600, (J - J(1))/abs(J(1))); grid on;
ylabel('Relative Jacobi drift'); xlabel('Time [h]');

fig = moon_viewer(o.ppd, 2, 1);   % true relief so low orbits are drawn correctly
ax  = findobj(fig, 'Type', 'axes');
plot3(ax(1), rbf(1,:)/1e3, rbf(2,:)/1e3, rbf(3,:)/1e3, 'c', 'LineWidth', 1.5);
if ~isempty(te)
    plot3(ax(1), rbf(1,end)/1e3, rbf(2,end)/1e3, rbf(3,end)/1e3, 'ro', 'MarkerFaceColor','r');
end

out = struct('t',t, 'r_inertial',rin, 'v_inertial',vin, 'r_body',rbf, 'lat',lat, ...
             'lon',lon, 'alt_terrain',alt, 'jacobi',J, 'impact_time',te);
end

% ===========================================================================
function ds = rhs(t, s, useEarth, C)
th = C.omega * t; c = cos(th); sn = sin(th);
r  = s(1:3);
rb = [c*r(1) + sn*r(2); -sn*r(1) + c*r(2); r(3)];      % inertial -> body
ab = lunar_gravity(rb, useEarth);
ai = [c*ab(1) - sn*ab(2); sn*ab(1) + c*ab(2); ab(3)];  % body -> inertial
ds = [s(4:6); ai];
end

function [val, isterminal, direction] = groundEvent(t, s, dem, C)
th = C.omega * t; c = cos(th); sn = sin(th);
rb = [c*s(1) + sn*s(2); -sn*s(1) + c*s(2); s(3)];
rn = norm(rb);
lat = asind(rb(3)/rn);
lon = mod(atan2d(rb(2), rb(1)), 360);
val = rn - (C.R_lola + dem.interp(lat, lon));
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
