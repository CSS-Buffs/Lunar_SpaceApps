function C = lunar_constants()
%LUNAR_CONSTANTS Physical constants for the Moon (SI units).
%   Gravity coefficients: GRAIL GRGM1200A (unnormalised degree-2 terms).

C.GM        = 4.9028001218467e12;          % m^3/s^2
C.R_grav    = 1738.0e3;                    % m, gravity reference radius
C.R_lola    = 1737.4e3;                    % m, LOLA reference sphere
C.J2        = 2.0321568e-4;                % = -C20*sqrt(5)
C.C22       = 2.2382e-5;
C.omega     = 2*pi / (27.321661*86400);    % rad/s sidereal rotation
C.GM_earth  = 3.986004418e14;              % m^3/s^2
C.d_earth   = 384400e3;                    % m, mean Earth-Moon distance
end
