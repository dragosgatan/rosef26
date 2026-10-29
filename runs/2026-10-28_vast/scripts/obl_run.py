"""Run the OBLITERATUS CLI with host memory measured by psutil's MemAvailable only.

Its cgroup reading (limit - memory.current) counts reclaimable page cache as used, so on
this container it sees ~1 GB free while ~26 GB is available and refuses to save. Nothing
else is changed.
"""
import sys

import obliteratus.models.host_resources as host_resources

host_resources._cgroup_available_bytes = lambda *a, **k: None

from obliteratus.cli import main  # noqa: E402

sys.argv[0] = "obliteratus"
sys.exit(main())
