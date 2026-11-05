function [a, U] = lunar_gravity(r, useEarth)
%LUNAR_GRAVITY Acceleration and potential in the Moon body-fixed frame.
%   [a, U] = lunar_gravity(r, useEarth)
%   r : 3xN body-fixed positions [m] (Mean Earth / Polar Axis frame)
%   a : 3xN acceleration [m/s^2],  U : 1xN potential [m^2/s^2] (a = +grad U)
%
%   Model: point mass + J2 + C22 (GRAIL GRGM1200A) + optional Earth tide
%   (Earth sits on the body-fixed +X axis because of synchronous rotation).

if nargin < 2, useEarth = false; end
C = lunar_constants();
x = r(1,:); y = r(2,:); z = r(3,:);
r2 = x.^2 + y.^2 + z.^2;
rn = sqrt(r2);
r3 = r2.*rn; r5 = r3.*r2; r7 = r5.*r2;
GM = C.GM; R2 = C.R_grav^2;

% central
a = -GM .* r ./ r3;

% J2
k  = -1.5 * C.J2 * GM * R2 ./ r5;
zz = 5 * z.^2 ./ r2;
a(1,:) = a(1,:) + k .* x .* (1 - zz);
a(2,:) = a(2,:) + k .* y .* (1 - zz);
a(3,:) = a(3,:) + k .* z .* (3 - zz);

% C22 :  U22 = 3 GM R^2 C22 (x^2 - y^2) / r^5
c = 3 * GM * R2 * C.C22;
d = x.^2 - y.^2;
a(1,:) = a(1,:) + c .* ( 2*x ./ r5 - 5*x.*d ./ r7);
a(2,:) = a(2,:) + c .* (-2*y ./ r5 - 5*y.*d ./ r7);
a(3,:) = a(3,:) + c .* (-5*z.*d ./ r7);

U = GM ./ rn ...
    - GM * C.J2 * R2 ./ (2*r3) .* (3*z.^2 ./ r2 - 1) ...
    + c .* d ./ r5;

if useEarth
    De = C.d_earth;
    dx = De - x;
    dist = sqrt(dx.^2 + y.^2 + z.^2);
    a(1,:) = a(1,:) + C.GM_earth * ( dx ./ dist.^3 - 1/De^2);
    a(2,:) = a(2,:) + C.GM_earth * (-y  ./ dist.^3);
    a(3,:) = a(3,:) + C.GM_earth * (-z  ./ dist.^3);
    U = U + C.GM_earth * (1 ./ dist - x / De^2);
end
end
