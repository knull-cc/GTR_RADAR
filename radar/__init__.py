"""RADAR plug-in for GTR-family forecasters (see radar/radar.py for the wrapper)."""
from radar import compat  # noqa: F401  (numpy>=2 shim for the original utils/tools.py)


def __getattr__(name):  # lazy, so `import radar.compat` stays lightweight
    if name == 'RADARWrapper':
        from radar.radar import RADARWrapper
        return RADARWrapper
    raise AttributeError(name)
