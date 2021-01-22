"""
Logger __init__.
"""
try:
    from spikey.logging.log import log
    from spikey.logging.multilogger import MultiLogger
    from spikey.logging.reader import Reader, MetaReader, dejsonize
    from spikey.logging.sanitize import sanitize, sanitize_dictionary
    from spikey.logging.serialize import compressnd, uncompressnd

except ImportError as e:
    raise ImportError(f"logging/__init__.py failed: {e}")
