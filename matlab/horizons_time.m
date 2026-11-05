function jd_tdb = horizons_time(utc)
%HORIZONS_TIME UTC ISO string (e.g. '2025-06-01T00:00:00') or datetime -> Julian date (TDB).
%   TDB - UTC = 69.184 s (37 leap seconds + 32.184 s), valid since 2017-01-01.
if ~isdatetime(utc)
    utc = datetime(strrep(char(utc), 'Z', ''), 'InputFormat', inputFormat(char(utc)), 'TimeZone', 'UTC');
end
utc.TimeZone = 'UTC';
jd_tdb = 2440587.5 + posixtime(utc)/86400 + 69.184/86400;
end

function f = inputFormat(s)
s = strrep(s, 'Z', '');
switch numel(s)
    case 10, f = 'yyyy-MM-dd';
    case 16, f = 'yyyy-MM-dd''T''HH:mm';
    otherwise
        if contains(s, '.'), f = 'yyyy-MM-dd''T''HH:mm:ss.SSS'; else, f = 'yyyy-MM-dd''T''HH:mm:ss'; end
end
end
