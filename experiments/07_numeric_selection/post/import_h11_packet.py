"""Import a saved H11 packet and bind it to the raw server-file hashes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]/'results'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path, help='Directory copied from stage6/h11/packet over SSH')
    args = parser.parse_args()
    summary = json.loads((args.packet/'h11.json').read_text())
    provenance = json.loads((ROOT/'provenance.json').read_text())
    raw_names = {}
    for name, info in summary['sources'].items():
        public_name = f'h11_raw_{name}.gz'
        raw = gzip.decompress((args.packet/public_name).read_bytes())
        assert len(raw) == info['bytes'] and hashlib.sha256(raw).hexdigest() == info['sha256']
        source = 'stage6/h11/'+name
        provenance['sources'][source] = info
        raw_names[public_name] = source
    names = ['h11.json'] + list(raw_names)
    if summary['gate']['passed']:
        names += ['h11_per_seed.csv'] + [f'h11_sets_{s}.json' for s in range(20261201,20261209)]
    for name in names:
        source_path = args.packet/name
        assert source_path.stat().st_size < 2_000_000
        if source_path.resolve() != (ROOT/name).resolve():
            shutil.copyfile(source_path, ROOT/name)
        if name in raw_names:
            dependencies = [raw_names[name]]
        elif name.startswith('h11_sets_'):
            seed = name.removeprefix('h11_sets_').removesuffix('.json')
            dependencies = [f'stage6/h11/patch_{seed}.json']
        else:
            dependencies = list(raw_names.values())
        provenance['summary_sources'][name] = sorted(dependencies)
    if summary['gate']['passed']:
        provenance.setdefault('table_parents', {})['h11_per_seed.csv'] = ['h11.json']
    provenance['note'] = ('Hashes measured from saved server files at export. Earlier raw files are not bundled; '
                          'H11 raw JSON is bundled as deterministic gzip. Public summary hashes include path redactions.')
    (ROOT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    print(f'Imported {len(names)} H11 files; all compressed raw hashes agree. Run make_tables.py to refresh checksums.')


if __name__ == '__main__':
    main()
