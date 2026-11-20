"""Download generation-pinned official GCS objects and verify their CRC32C."""
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote

import google_crc32c
import requests

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/pi05_base"
DEST = ROOT.parent / "openpi/checkpoints/pi05_base"
PREFIX = "checkpoints/pi05_base/"


def segmented(item, temporary, url):
    """Keep any existing contiguous prefix and fetch the remainder in ranged pieces."""
    parts = temporary.with_name(temporary.name + ".parts")
    parts.mkdir(exist_ok=True)
    plan_path = parts / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan["generation"] == item["generation"]
    else:
        prefix = temporary.stat().st_size if temporary.exists() else 0
        plan = {"generation": item["generation"], "prefix": prefix, "size": int(item["size"])}
        plan_path.write_text(json.dumps(plan))
    assert (temporary.stat().st_size if temporary.exists() else 0) == plan["prefix"]
    ranges = [(start, min(start + (64 << 20), plan["size"]) - 1)
              for start in range(plan["prefix"], plan["size"], 64 << 20)]

    def fetch(bounds):
        start, end = bounds
        final = parts / f"part_{start:012d}"
        part = final.with_suffix(".partial")
        if final.exists():
            assert final.stat().st_size == end - start + 1
            return final
        session = requests.Session()
        session.trust_env = False
        for attempt in range(5):
            try:
                have = part.stat().st_size if part.exists() else 0
                assert have <= end - start + 1
                if have < end - start + 1:
                    with session.get(url, params={"generation": item["generation"]},
                                     headers={"Range": f"bytes={start + have}-{end}"},
                                     timeout=(20, 120), stream=True) as response:
                        response.raise_for_status()
                        assert response.status_code == 206
                        assert response.headers["Content-Range"].startswith(f"bytes {start + have}-{end}/")
                        with part.open("ab" if have else "wb") as stream:
                            for chunk in response.iter_content(chunk_size=1 << 20):
                                stream.write(chunk)
                assert part.stat().st_size == end - start + 1
                part.replace(final)
                return final
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    with ThreadPoolExecutor(max_workers=16) as pool:
        files = list(pool.map(fetch, ranges))
    assembled = temporary.with_name(temporary.name + ".assembling")
    with assembled.open("wb") as output:
        if plan["prefix"]:
            with temporary.open("rb") as prefix:
                shutil.copyfileobj(prefix, output, length=8 << 20)
        for path in files:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 << 20)
    result = verify(assembled, item)
    target = temporary.with_name(temporary.name.removesuffix(".partial"))
    assembled.replace(target)
    temporary.unlink(missing_ok=True)
    shutil.rmtree(parts)
    return result


def verify(path, item):
    assert path.stat().st_size == int(item["size"]), path
    crc, md5, sha = google_crc32c.Checksum(), hashlib.md5(), hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            crc.update(block)
            md5.update(block)
            sha.update(block)
    actual = base64.b64encode(crc.digest()).decode()
    assert actual == item["crc32c"], f"CRC32C mismatch: {path}"
    if item.get("md5Hash"):
        assert base64.b64encode(md5.digest()).decode() == item["md5Hash"], path
    return {"name": item["name"], "bytes": int(item["size"]), "crc32c": actual,
            "sha256": sha.hexdigest(), "generation": item["generation"]}


def download(item):
    assert item["name"].startswith(PREFIX)
    relative = Path(item["name"][len(PREFIX):])
    assert not relative.is_absolute() and ".." not in relative.parts
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return verify(target, item)
    temporary = target.with_name(target.name + ".partial")
    session = requests.Session()
    session.trust_env = False  # Direct GCS route was verified; inherited proxy fails TLS.
    url = "https://storage.googleapis.com/openpi-assets/" + quote(item["name"], safe="/")
    size = int(item["size"])
    if size > 128 << 20:
        return segmented(item, temporary, url)
    for attempt in range(4):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            assert offset <= size, temporary
            if size == 0:
                temporary.touch()
            elif offset < size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with session.get(url, params={"generation": item["generation"]}, headers=headers,
                                 timeout=(20, 120), stream=True) as response:
                    response.raise_for_status()
                    if offset:
                        assert response.status_code == 206, "Server did not honor resume Range"
                        assert response.headers["Content-Range"].startswith(f"bytes {offset}-")
                    with temporary.open("ab" if offset else "wb") as output:
                        for chunk in response.iter_content(chunk_size=4 << 20):
                            output.write(chunk)
            result = verify(temporary, item)
            temporary.replace(target)
            return result
        except Exception:
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 8))


def main():
    items = json.loads((RUN / "gcs_manifest.json").read_text())
    assert items and all(x["name"].startswith(PREFIX) for x in items)
    DEST.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(download, item) for item in items}
        while pending:
            completed, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in completed:
                results.append(future.result())
            downloaded = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
            status = {"stage": "downloading" if pending else "complete", "verified_files": len(results),
                      "total_files": len(items), "downloaded_bytes": downloaded,
                      "total_bytes": sum(int(item["size"]) for item in items),
                      "elapsed_seconds": time.time() - started, "destination": str(DEST)}
            temp = RUN / "download_status.tmp"
            temp.write_text(json.dumps(status, indent=2) + "\n")
            temp.replace(RUN / "download_status.json")
            print(json.dumps(status), flush=True)
    (RUN / "verified_files.json").write_text(json.dumps(sorted(results, key=lambda x: x["name"]), indent=2) + "\n")


if __name__ == "__main__":
    main()
