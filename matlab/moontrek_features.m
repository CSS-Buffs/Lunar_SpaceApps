function F = moontrek_features()
%MOONTREK_FEATURES IAU names with coordinates from the NASA Moon Trek search index.
%   F = struct array with fields name, type, lat, lon (east, -180..180)
%   Includes all maria/oceani, the Apollo landing sites and a list of major craters.

base = 'https://trek.nasa.gov/moon/TrekServices/ws/index/eq/searchItems';
F = struct('name', {}, 'type', {}, 'lat', {}, 'lon', {});
types = {'Mare, maria', 'Oceanus, oceani', 'Crewed Landing'};
for k = 1:numel(types)
    docs = search({'start','0', 'rows','100', 'facetKeys','itemType|productType', ...
                   'facetValues', ['nomenclature|' types{k}]});
    for i = 1:numel(docs)
        d = docs{i};
        [la, lo, ok] = point(d);
        if ok
            nm = strrep(d.title, ' LM descent stage', '');
            F(end+1) = struct('name', nm, 'type', strtok(types{k}, ','), 'lat', la, 'lon', lo); %#ok<AGROW>
        end
    end
end
craters = {'Tycho','Copernicus','Kepler','Aristarchus','Plato','Clavius','Grimaldi','Ptolemaeus', ...
           'Theophilus','Langrenus','Petavius','Tsiolkovskiy','Korolev','Hertzsprung','Shackleton', ...
           'Daedalus','Mendeleev'};
for k = 1:numel(craters)
    docs = search({'start','0', 'rows','5', 'facetKeys','itemType', 'facetValues','nomenclature', ...
                   'key', craters{k}});
    for i = 1:numel(docs)
        d = docs{i};
        [la, lo, ok] = point(d);
        if ok && strcmpi(d.title, craters{k})
            F(end+1) = struct('name', d.title, 'type', 'Crater', 'lat', la, 'lon', lo); %#ok<AGROW>
            break
        end
    end
end

    function docs = search(params)
        file = nasa_get('moontrek/search', base, params, '.json');
        j = jsondecode(fileread(file));
        docs = j.response.docs;               % struct array or cell (mixed fields)
        if ~iscell(docs), docs = num2cell(docs); end
    end
end

function [lat, lon, ok] = point(d)
ok = isfield(d, 'shape') && startsWith(d.shape, 'POINT');
lat = NaN; lon = NaN;
if ok
    v = sscanf(erase(d.shape, {'POINT (', ')'}), '%f');
    lon = v(1); lat = v(2);
end
end
