import os
import shutil
from itertools import product
from os import path
from setuptools import setup


# Define global path variables
setup_dir = path.dirname(path.abspath(__file__))
package_name = "gridrec"
package_dir = path.join(setup_dir, package_name)


# Define utility functions to build the extensions
def get_common_extension_args():
    import numpy

    fft_libs = [
        "fftw3" + suf + thrd_ext
        for suf, thrd_ext in product(["", "f", "l"], ["", "_threads"])
    ]
    common_extension_args = dict(
        libraries=fft_libs + ["m"],
        library_dirs=[],
        include_dirs=[numpy.get_include()],
        extra_compile_args="-O3 -fomit-frame-pointer -fstrict-aliasing -ffast-math".split(),
    )
    return common_extension_args


def get_cython_extensions():
    from distutils.extension import Extension
    from Cython.Build import cythonize

    ext_modules = []
    common_extension_args = get_common_extension_args()
    ext_modules.append(
        Extension(
            name=package_name + "._gridrec",
            sources=[
                path.join(package_dir, "_gridrec.pyx"),
                path.join(package_dir, "_gridrec_backproj.c"),
                path.join(package_dir, "_gridrec_fwdproj.c"),
            ],
            **common_extension_args
        )
    )
    return cythonize(ext_modules)


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if path.exists("MANIFEST"):
    os.remove("MANIFEST")


# Define custom clean command
from distutils.core import Command


class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = []
        # Clean Cython generated files and cache
        gen_file_exts = [".pyc",
                    ".so",
                    ".o",
                    ".pyo",
                    ".pyd",
                    ".orig",]
        for root, dirs, files in os.walk(package_dir):
            for f in files:
                if f in self._clean_exclude:
                    continue
                base, ext = path.splitext(f)
                if ext in gen_file_exts:
                    self._clean_me.append(path.join(root, f))
                if ext == ".c" and path.exists(path.join(root, base + ".pyx")):
                    self._clean_me.append(path.join(root, f))
            for d in dirs:
                if d == "__pycache__":
                    self._clean_trees.append(path.join(root, d))
        # clean build and sdist directories in root
        for d in ("build", "dist"):
            if path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                shutil.rmtree(clean_tree)
            except Exception:
                pass


cmdclass = {"clean": CleanCommand}

setup(cmdclass=cmdclass, ext_modules=get_cython_extensions(), packages=["gridrec"])
