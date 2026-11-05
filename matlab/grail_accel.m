function a = grail_accel(r, G)
%GRAIL_ACCEL Gravitational acceleration [m/s^2] of the GRAIL spherical-harmonic field.
%   a = grail_accel(r_pa, G)   r_pa : 3x1 position in the lunar Principal-Axis frame [m]
%   Fully-normalised Cunningham / Montenbruck-Gill V,W recursion (stable to high degree);
%   identical algorithm to python/grail_gravity.py (validated there against finite differences).

N = G.N; M = N + 2;
x = r(1); y = r(2); z = r(3);
r2 = x*x + y*y + z*z;
rho = G.R / r2;
x0 = x*rho; y0 = y*rho; z0 = z*rho; rr = G.R*rho;
V = zeros(M); W = zeros(M);          % V(n+1, m+1)
V(1,1) = G.R / sqrt(r2);
for m = 0:M-1
    if m > 0
        V(m+1, m+1) = G.c(m+1) * (x0*V(m, m) - y0*W(m, m));
        W(m+1, m+1) = G.c(m+1) * (x0*W(m, m) + y0*V(m, m));
    end
    for n = m+1:M-1
        V(n+1, m+1) = G.a(n+1, m+1) * z0 * V(n, m+1);
        W(n+1, m+1) = G.a(n+1, m+1) * z0 * W(n, m+1);
        if n >= m + 2
            V(n+1, m+1) = V(n+1, m+1) - G.b(n+1, m+1) * rr * V(n-1, m+1);
            W(n+1, m+1) = W(n+1, m+1) - G.b(n+1, m+1) * rr * W(n-1, m+1);
        end
    end
end
ax = 0; ay = 0; az = 0;
for n = 0:N
    for m = 0:n
        Cn = G.C(n+1, m+1); Sn = G.S(n+1, m+1);
        if m == 0
            ax = ax - Cn * G.kup(n+1, 1) * V(n+2, 2);
            ay = ay - Cn * G.kup(n+1, 1) * W(n+2, 2);
        else
            f = (n-m+2)*(n-m+1);
            ax = ax + 0.5*(G.kup(n+1,m+1)*(-Cn*V(n+2,m+2) - Sn*W(n+2,m+2)) ...
                      + f*G.kdn(n+1,m+1)*( Cn*V(n+2,m)   + Sn*W(n+2,m)));
            ay = ay + 0.5*(G.kup(n+1,m+1)*(-Cn*W(n+2,m+2) + Sn*V(n+2,m+2)) ...
                      + f*G.kdn(n+1,m+1)*(-Cn*W(n+2,m)   + Sn*V(n+2,m)));
        end
        az = az + (n-m+1) * G.kz(n+1,m+1) * (-Cn*V(n+2,m+1) - Sn*W(n+2,m+1));
    end
end
a = (G.GM / G.R^2) * [ax; ay; az];
end
