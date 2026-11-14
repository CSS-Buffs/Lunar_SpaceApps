function eph = horizons_vectors(body, jd_start, jd_stop, step)
%HORIZONS_VECTORS State vectors from the NASA/JPL Horizons API, relative to the Moon.
%   eph = horizons_vectors('earth', jd0, jd1, '1h')
%   body : 'earth' | 'sun' | 'lro' | any Horizons COMMAND id (e.g. '-85')
%   Times are TDB Julian dates. Frame: ICRF, centre: Moon (500@301). SI units.
%
%   eph.jd   (N x 1)   TDB Julian dates
%   eph.pos  (N x 3)   m
%   eph.vel  (N x 3)   m/s
%   eph.at(jd)         cubic-Hermite interpolated position(s) [m], jd scalar or vector -> (n x 3)

ids = struct('earth', '399', 'sun', '10', 'lro', '-85', 'moon', '301');
if isfield(ids, lower(body)), cmd = ids.(lower(body)); else, cmd = body; end
q = @(s) ['''' s ''''];
params = {'format', 'json', 'COMMAND', q(cmd), 'CENTER', q('500@301'), 'MAKE_EPHEM', q('YES'), ...
          'OBJ_DATA', q('NO'), 'EPHEM_TYPE', q('VECTORS'), ...
          'START_TIME', q(sprintf('JD %.9f', jd_start)), 'STOP_TIME', q(sprintf('JD %.9f', jd_stop)), ...
          'STEP_SIZE', q(step), 'VEC_TABLE', q('2'), 'REF_PLANE', q('FRAME'), 'REF_SYSTEM', q('ICRF'), ...
          'OUT_UNITS', q('KM-S'), 'CSV_FORMAT', q('YES'), 'VEC_LABELS', q('NO'), 'TIME_TYPE', q('TDB')};
file = nasa_get('horizons', 'https://ssd.jpl.nasa.gov/api/horizons.api', params, '.json');
res  = jsondecode(fileread(file));
rows = horizons_rows(res.result);

M = zeros(numel(rows), 7);
for k = 1:numel(rows)
    c = strsplit(rows{k}, ',');
    M(k, :) = str2double(c([1 3:8]));
end
eph.name = body;
eph.jd   = M(:, 1);
eph.pos  = M(:, 2:4) * 1e3;
eph.vel  = M(:, 5:7) * 1e3;
eph.at   = @(jd) hermite(eph.jd, eph.pos, eph.vel, jd);
end

function rows = horizons_rows(txt)
i0 = strfind(txt, '$$SOE'); i1 = strfind(txt, '$$EOE');
if isempty(i0)
    error('horizons:noData', 'Horizons returned no ephemeris:\n%s', txt(max(1, end-1500):end));
end
body = strtrim(txt(i0(1)+5 : i1(1)-1));
rows = strtrim(splitlines(body));
rows = rows(~cellfun(@isempty, rows));
end

function P = hermite(jd, pos, vel, jq)
% cubic Hermite interpolation on a uniform grid using tabulated positions and velocities
h  = (jd(2) - jd(1)) * 86400;
u  = (jq(:) - jd(1)) * 86400 / h;
k  = min(max(floor(u), 0), numel(jd) - 2);
s  = u - k;  k = k + 1;
h00 = 2*s.^3 - 3*s.^2 + 1;  h10 = s.^3 - 2*s.^2 + s;
h01 = -2*s.^3 + 3*s.^2;     h11 = s.^3 - s.^2;
P = h00.*pos(k,:) + h10.*h.*vel(k,:) + h01.*pos(k+1,:) + h11.*h.*vel(k+1,:);
end
