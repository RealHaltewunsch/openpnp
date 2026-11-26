"""Local vision service for OpenPnP.

Loads a shared encoder once per process, loads task heads on demand,
warms them before first time-critical use, and exposes a loopback HTTP
API (see docs/VISION-SERVICE.md).
"""

__version__ = "0.1.0"
