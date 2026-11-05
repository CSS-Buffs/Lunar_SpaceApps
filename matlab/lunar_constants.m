function C = lunar_constants()
%LUNAR_CONSTANTS Physical constants (SI units). Lunar gravity itself comes from GRAIL (grail_load).

C.R_lola    = 1737.4e3;                    % m, LOLA reference sphere
C.GM_earth  = 3.986004418e14;              % m^3/s^2
C.GM_sun    = 1.32712440041279419e20;      % m^3/s^2 (DE440)
C.AU        = 1.495978707e11;              % m
C.P_sun     = 4.56e-6;                     % N/m^2 solar radiation pressure at 1 AU
end
