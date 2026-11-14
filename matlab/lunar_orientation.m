function [R_icrf2me, R_me2pa] = lunar_orientation(jd_tdb)
%LUNAR_ORIENTATION Rotation from ICRF to the Moon Mean-Earth (ME) frame, and ME -> PA.
%   [R_icrf2me, R_me2pa] = lunar_orientation(jd_tdb)     r_ME = R_icrf2me * r_ICRF
%
%   ICRF -> ME : IAU WGCCRE 2015 lunar rotation model with libration terms
%                (Archinal et al. 2018). Agrees with JPL Horizons sub-Earth points to ~0.006 deg.
%   ME -> PA   : DE440 constant rotation from NASA NAIF frame kernel moon_de440_250416.tf
%                (67.8526", 78.6944", 0.2785" about axes 3,2,1). GRAIL gravity is in PA.
%                Sign verified against real LRO trajectories (see python/validate.py).

d = jd_tdb - 2451545.0;
T = d / 36525;
E = [125.045 - 0.0529921*d, 250.089 - 0.1059842*d, 260.008 + 13.0120009*d, ...
     176.625 + 13.3407154*d, 357.529 + 0.9856003*d, 311.589 + 26.4057084*d, ...
     134.963 + 13.0649930*d, 276.617 + 0.3287146*d, 34.226 + 1.7484877*d, ...
     15.134 - 0.1589763*d, 119.743 + 0.0036096*d, 239.961 + 0.1643573*d, 25.053 + 12.9590088*d];
s = sind(E); c = cosd(E);
a0 = 269.9949 + 0.0031*T - 3.8787*s(1) - 0.1204*s(2) + 0.0700*s(3) - 0.0172*s(4) ...
     + 0.0072*s(6) - 0.0052*s(10) + 0.0043*s(13);
d0 = 66.5392 + 0.0130*T + 1.5419*c(1) + 0.0239*c(2) - 0.0278*c(3) + 0.0068*c(4) ...
     - 0.0029*c(6) + 0.0009*c(7) + 0.0008*c(10) - 0.0009*c(13);
W  = 38.3213 + 13.17635815*d - 1.4e-12*d^2 + 3.5610*s(1) + 0.1208*s(2) - 0.0642*s(3) ...
     + 0.0158*s(4) + 0.0252*s(5) - 0.0066*s(6) - 0.0047*s(7) - 0.0046*s(8) + 0.0028*s(9) ...
     + 0.0052*s(10) + 0.0040*s(11) + 0.0019*s(12) - 0.0044*s(13);

R_icrf2me = R3(deg2rad(W)) * R1(deg2rad(90 - d0)) * R3(deg2rad(90 + a0));

as = pi/(180*3600);
R_me2pa = R3(67.8526*as) * R2(78.6944*as) * R1(0.2785*as);
end

% passive (frame) rotation matrices
function R = R1(a), R = [1 0 0; 0 cos(a) sin(a); 0 -sin(a) cos(a)]; end
function R = R2(a), R = [cos(a) 0 -sin(a); 0 1 0; sin(a) 0 cos(a)]; end
function R = R3(a), R = [cos(a) sin(a) 0; -sin(a) cos(a) 0; 0 0 1]; end
