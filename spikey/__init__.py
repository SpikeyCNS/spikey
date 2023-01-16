"""
Spikey __init__.
"""
try:
    # import spikey.<analysis, viz>
    import spikey.games  # import spikey.games.<game_class>
    from spikey.snn import *  # import spikey.<part_type>.<part_class>

    from spikey._metapath_dir_skip import install_skipfolder

    install_skipfolder()

    from spikey.module import Module, Key, save, load

except ImportError as e:
    raise ImportError(f"snn/__init__.py failed: {e}")
