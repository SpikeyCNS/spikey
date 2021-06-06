"""
Custom MetaPathFinder.

Help find imports eg spikey.synapse.RLSTDP
where the path is spikey/snn/synapse/rlstdp.py::RLSTDP.
"""
## TODO Replace with pathhook?
import sys
import os.path

from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_file_location


class SkipFolderPathFinder(MetaPathFinder):
    """
    Custom module loader for our project.
    eg.
    importing RLSTDP from spikey/snn/synapse/rlstdp
    as spikey.synapse.RLSTDP.
    importing CartPole from spikey/games/CartPole
    as spikey.RL.Cartpole.

    type(spikey.snn.synapse.full_rule.RLSTDP) == type(spikey.synapse.RLSTDP).
    """

    def find_spec(self, fullname: str, path: str, target: object = None):
        """
        Finding spec for specific module.

        Parameters
        ----------
        spec: str
            Import name.
        path: str
            If this is top-level import, path=None
            Otherwise, this is a search for a subpackage or
            module and path will be the value of __path__
            from the parent package.
        target: module
            Module object that the finder may use to make a
            more educated guess about what spec to return.

        Returns
        -------
        None if spec not found, ???
        """
        if path is None or path == "":
            path = [os.getcwd()]

        *parents, name = fullname.split(".")

        if not len(parents) or parents[0] != "spikey" or len(parents) > 1:
            return None

        for entry in path:
            for subdir in os.listdir(entry):
                subdir_path = os.path.join(entry, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                full_path = os.path.join(subdir_path, name)

                if os.path.isdir(full_path):
                    filename = os.path.join(full_path, "__init__.py")
                    submodule_locations = [full_path]
                    break
            else:
                continue

            return spec_from_file_location(
                fullname,
                filename,
                loader=CustomLoader(filename),
                submodule_search_locations=submodule_locations,
            )

        return None

    def find_module(self, fullname, path):
        """
        A legacy method for finding a loader for the specified
        module. If this is a top-level import, path will be
        None. Otherwise, this is a search for a subpackage or
        module and path will be the value of __path__ from the
        parent package. If a loader cannot be found, None is
        returned.

        If find_spec() is defined, backwards-compatible
        functionality is provided.
        """
        return self.find_spec(fullname, path)

    def invalidate_caches(self):
        """
        An optional method which, when called, should invalidate
        any internal cache used by the finder. Used by
        importlib.invalidate_caches() when invalidating the
        caches of all finders on sys.meta_path.
        """
        return None


class CustomLoader(Loader):
    def __init__(self, filename):
        self.filename = filename

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(self.filename) as f:
            data = f.read()

        exec(data, vars(module))


def install_skipfolder():
    """Inserts the finder into the import machinery"""
    sys.meta_path.insert(0, SkipFolderPathFinder())
