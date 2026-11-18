# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Verify the distributed files against the SHA256 manifest."""
from pathlib import Path
import argparse,hashlib,json
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.root.resolve();manifest=root/'checksums.sha256'
    checked=0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        digest,name=line.split('  ',1);path=(root/name).resolve()
        assert path.is_relative_to(root),name
        assert path.is_file(),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        checked+=1
    print(json.dumps(dict(passed=True,files_checked=checked)))
if __name__=='__main__':main()
