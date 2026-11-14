function G = grail_load(degree)
%GRAIL_LOAD GRAIL GRGM1200A lunar gravity field (NASA PDS), truncated to `degree`.
%   G = grail_load(100)
%   Only the bytes needed are downloaded (HTTP Range request): the file is ordered by
%   degree with fixed-length 122-byte records after a 244-byte header.
%   Coefficients are fully normalised, Principal-Axis frame, R = 1738 km.
%
%   G.GM, G.R, G.N, G.C, G.S ((N+1)x(N+1), index (n+1, m+1)) plus precomputed
%   recursion/normalisation tables used by grail_accel.

if nargin < 1, degree = 100; end
dataDir = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'grail');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
cache = fullfile(dataDir, sprintf('gggrx_1200a_d%d.mat', degree));
if isfile(cache)
    S = load(cache); G = S.G; return
end

url    = 'https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200a_sha.tab';
nrec   = degree*(degree+3)/2;
nbytes = 244 + 122*nrec;
fprintf('Downloading GRAIL GRGM1200A to degree %d (%.1f MB) from NASA PDS...\n', degree, nbytes/1e6);
opts = weboptions('Timeout', 600, 'ContentType', 'text', ...
                  'HeaderFields', {'Range', sprintf('bytes=0-%d', nbytes-1)});
try
    txt = webread(url, opts);                         % partial download (HTTP 206)
catch
    fprintf('Range request not supported here; downloading the full 88 MB file once...\n');
    txt = fileread(nasa_get('grail', url, {}, '.tab'));
end
lines = splitlines(txt);
head  = str2double(strsplit(lines{1}, ','));
G.R   = head(1)*1e3;
G.GM  = head(2)*1e9;
G.N   = degree;
G.C   = zeros(degree+1); G.S = zeros(degree+1);
G.C(1,1) = 1;
for k = 2:numel(lines)
    v = sscanf(strrep(lines{k}, ',', ' '), '%f');
    if numel(v) < 4, continue, end
    n = v(1); m = v(2);
    if n > degree, break, end
    G.C(n+1, m+1) = v(3); G.S(n+1, m+1) = v(4);
end
G = add_tables(G);
save(cache, 'G');
end

function G = add_tables(G)
N = G.N; M = N + 2;
logN = @(n, m) 0.5*(log(2 - (m == 0)) + log(2*n + 1) + gammaln(n - m + 1) - gammaln(n + m + 1));
G.kup = zeros(N+1); G.kdn = zeros(N+1); G.kz = zeros(N+1);
for n = 0:N
    for m = 0:n
        G.kup(n+1, m+1) = exp(logN(n, m) - logN(n+1, m+1));
        G.kz(n+1, m+1)  = exp(logN(n, m) - logN(n+1, m));
        if m > 0, G.kdn(n+1, m+1) = exp(logN(n, m) - logN(n+1, m-1)); end
    end
end
G.a = zeros(M); G.b = zeros(M); G.c = zeros(M, 1);
for n = 1:M-1
    for m = 0:n-1
        G.a(n+1, m+1) = sqrt((2*n+1)*(2*n-1) / ((n-m)*(n+m)));
        if n >= 2
            G.b(n+1, m+1) = sqrt((2*n+1)*(n+m-1)*(n-m-1) / ((2*n-3)*(n+m)*(n-m)));
        end
    end
end
for m = 1:M-1
    if m == 1, G.c(m+1) = sqrt(3); else, G.c(m+1) = sqrt((2*m+1)/(2*m)); end
end
end
