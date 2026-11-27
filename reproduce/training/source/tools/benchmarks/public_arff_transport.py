"""Bounded public ARFF transport; validates the same pinned OpenML checksum.

Uses the existing tested range/identity journal when supported. A server that
does not serve ranges is retried as a whole stream; incomplete bytes never
become the OpenML cache file. No preprocessing or model statement is changed.
"""
from hashlib import md5,sha256
from pathlib import Path
import json,time
import requests
from download_transport_base import range_download


def download_arff(url,target,expected_md5,getter=None,attempts=5,sleep=time.sleep):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    getter=requests.get if getter is None else getter
    def valid(path):
        return md5(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected_md5
    if target.exists():
        assert valid(target),'Existing ARFF checksum differs; preserve evidence'
        return dict(verified=True,already_cached=True,sha256=sha256(target.read_bytes()).hexdigest())
    last=None
    for index in range(attempts):
        partial=target.with_suffix('.arff.download-%d'%index)
        try:
            response=getter(url,stream=True,headers={'Accept-Encoding':'identity'},timeout=(20,45))
            try:
                response.raise_for_status()
                length=response.headers.get('Content-Length')
                validator=response.headers.get('ETag') or response.headers.get('Last-Modified')
                ranged=length and validator and response.headers.get('Accept-Ranges','').lower()=='bytes'
                if ranged:
                    actual=response.url;response.close()
                    identity=dict(public_source_sha256=sha256(url.encode()).hexdigest(),size=int(length),
                        validator=validator,expected_md5=expected_md5)
                    partial=target.with_suffix('.arff.ranges');marker=partial.with_suffix('.ranges.json')
                    range_download(actual,identity,partial,marker,getter=getter)
                else:
                    assert not partial.exists(),'Preserve an earlier incomplete download attempt'
                    size=0
                    with partial.open('xb') as handle:
                        for block in response.iter_content(256*1024):
                            size+=len(block);assert size<=200*1024**2,'ARFF exceeds transport cap'
                            handle.write(block)
                    if length:assert size==int(length),'Incomplete public ARFF body'
            finally:response.close()
            assert valid(partial),'Downloaded ARFF checksum differs'
            partial.replace(target)
            return dict(verified=True,already_cached=False,attempts=index+1,
                sha256=sha256(target.read_bytes()).hexdigest(),md5=expected_md5)
        except (requests.RequestException,OSError) as error:
            last=type(error).__name__
            if index+1<attempts:sleep(min(2**index,8))
    raise RuntimeError('Public ARFF transport failed after bounded retries: '+str(last))
