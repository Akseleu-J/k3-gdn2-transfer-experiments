"""Publish the tail of a failed workflow run as a check-run annotation (readable via the public API)."""
from pathlib import Path

NL, CR = chr(10), chr(13)
log = Path('ci.log')
lines = log.read_text(errors='replace').splitlines() if log.is_file() else ['ci.log not found']
tail = NL.join(line.rstrip()[:220] for line in [x for x in lines if 'Warning' not in x][-40:])
print('::error title=CI failure tail::' + tail.replace('%', '%25').replace(CR, '%0D').replace(NL, '%0A'))
