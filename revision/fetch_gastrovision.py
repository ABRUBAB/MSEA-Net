"""Download GastroVision (OSF project 84e7f, file 'Gastrovision.zip') and verify its SHA-256."""
import hashlib
import os
import sys
import time
import urllib.request
import zipfile

URL = "https://osf.io/download/gvx3q/"
SHA256 = "583bb7c35871c7e1414f2357aaaee74a724c41367860177ef63eeec74e011800"
SIZE = 1793518442

dst_dir = sys.argv[1]
os.makedirs(dst_dir, exist_ok=True)
zp = os.path.join(dst_dir, "Gastrovision.zip")
if not (os.path.exists(zp) and os.path.getsize(zp) == SIZE):
    t0, n = time.time(), 0
    req = urllib.request.Request(URL, headers={"User-Agent": "msea-net-revision/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(zp + ".part", "wb") as f:
        while True:
            b = r.read(8 << 20)
            if not b:
                break
            f.write(b)
            n += len(b)
    os.replace(zp + ".part", zp)
    print(f"downloaded {n / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)
h = hashlib.sha256()
with open(zp, "rb") as f:
    for b in iter(lambda: f.read(8 << 20), b""):
        h.update(b)
print("sha256 ok:", h.hexdigest() == SHA256, flush=True)
assert h.hexdigest() == SHA256, "checksum mismatch"
with zipfile.ZipFile(zp) as z:
    z.extractall(os.path.join(dst_dir, "gastrovision"))
print("unzipped", flush=True)
