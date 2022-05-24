# Copyright (c)  2021-2022  Xiaomi Corporation (author: Fangjun Kuang)

import glob
import os
import platform
import shutil
import sys
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext


def is_for_pypi():
    ans = os.environ.get("SHERPA_IS_FOR_PYPI", None)
    return ans is not None


def is_macos():
    return platform.system() == "Darwin"


def is_windows():
    return platform.system() == "Windows"


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            # In this case, the generated wheel has a name in the form
            # sherpa-xxx-pyxx-none-any.whl
            if is_for_pypi() and not is_macos():
                self.root_is_pure = True
            else:
                # The generated wheel has a name ending with
                # -linux_x86_64.whl
                self.root_is_pure = False


except ImportError:
    bdist_wheel = None


def cmake_extension(name, *args, **kwargs) -> setuptools.Extension:
    kwargs["language"] = "c++"
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext):
    def build_extension(self, ext: setuptools.extension.Extension):
        # build/temp.linux-x86_64-3.8
        os.makedirs(self.build_temp, exist_ok=True)

        # build/lib.linux-x86_64-3.8
        os.makedirs(self.build_lib, exist_ok=True)

        sherpa_dir = Path(__file__).parent.parent.resolve()

        cmake_args = os.environ.get("SHERPA_CMAKE_ARGS", "")
        make_args = os.environ.get("SHERPA_MAKE_ARGS", "")
        system_make_args = os.environ.get("MAKEFLAGS", "")

        if cmake_args == "":
            cmake_args = "-DCMAKE_BUILD_TYPE=Release"

        if "PYTHON_EXECUTABLE" not in cmake_args:
            print(f"Setting PYTHON_EXECUTABLE to {sys.executable}")
            cmake_args += f" -DPYTHON_EXECUTABLE={sys.executable}"

        if is_windows():
            build_cmd = f"""
                cmake {cmake_args} -B {self.build_temp} -S {sherpa_dir}
                cmake --build {self.build_temp} --target _sherpa --config Release -- -m
            """
            print(f"build command is:\n{build_cmd}")
            ret = os.system(f"cmake {cmake_args} -B {self.build_temp} -S {sherpa_dir}")
            if ret != 0:
                raise Exception("Failed to build sherpa")

            ret = os.system(
                f"cmake --build {self.build_temp} --target _sherpa --config release -- -m"
            )
            if ret != 0:
                raise exception("failed to build sherpa")
        else:
            if make_args == "" and system_make_args == "":
                print("for fast compilation, run:")
                print('export SHERPA_MAKE_ARGS="-j"; python setup.py install')

            build_cmd = f"""
                cd {self.build_temp}

                cmake {cmake_args} {sherpa_dir}


                make {make_args} _sherpa
            """
            print(f"build command is:\n{build_cmd}")

            ret = os.system(build_cmd)
            if ret != 0:
                raise Exception(
                    "\nBuild sherpa failed. Please check the error message.\n"
                    "You can ask for help by creating an issue on GitHub.\n"
                    "\nClick:\n\thttps://github.com/k2-fsa/sherpa/issues/new\n"  # noqa
                )

        lib_so = glob.glob(f"{self.build_temp}/lib/*sherpa*.so")
        lib_so += glob.glob(f"{self.build_temp}/lib/*sherpa*.dylib")  # macOS

        # bin/Release/_sherpa.cp38-win_amd64.pyd
        lib_so += glob.glob(
            f"{self.build_temp}/**/*sherpa*.pyd", recursive=True
        )  # windows

        # lib/Release/*.lib
        lib_so += glob.glob(
            f"{self.build_temp}/**/*sherpa*.lib", recursive=True
        )  # windows
        for so in lib_so:
            print(f"Copying {so} to {self.build_lib}/")
            shutil.copy(f"{so}", f"{self.build_lib}/")
