# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import logging
import os
import os.path as osp
import shutil
import sys
import tarfile
from distutils.util import get_platform
from glob import glob
from subprocess import CalledProcessError, run
from typing import Dict

import yaml

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGING_DIR = osp.join(CUR_DIR, 'packaging')
PLATFORM_TAG = get_platform().replace('-', '_').replace('.', '_')


def get_version(version_file):
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def _merge_cfg(cfg0, cfg1):
    cfg = copy.deepcopy(cfg0)
    for k, v in cfg1.items():
        if k in cfg:
            cfg[k] = _merge_cfg(cfg0[k], cfg1[k])
        else:
            cfg[k] = v
    return cfg


def _remove_if_exist(path):
    if osp.exists(path):
        logging.info(f'Remove path: {path}.')
        if osp.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _copy(src_path, dst_path):
    assert osp.exists(src_path), f'src path: {src_path} not exist.'

    logging.info(f'copy path: {src_path} to {dst_path}.')
    if osp.isdir(src_path):
        if osp.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
    else:
        shutil.copy(src_path, dst_path)


def _call_command(cmd, cwd, stdout=None, stderr=None):
    if cmd == '':
        return
    logging.info(f'Process cmd: {cmd}')
    logging.info(f'work_path: {cwd}')
    try:
        ret = run(cmd, stdout=stdout, stderr=stderr, cwd=cwd, shell=True)
        if ret.returncode != 0:
            logging.error(f'Process cmd: "{cmd}"'
                          f' failed with returncode: {ret.returncode}')
            exit()
    except Exception:
        logging.error(f'Process cmd: {cmd} failed.')
        exit()


def _create_tar(path, tar_name):
    logging.info(f'create tar file: {tar_name}')
    with tarfile.open(tar_name, 'w:gz') as tar:
        tar.add(path, arcname=os.path.basename(path))


def _create_bdist_cmd(cfg, dist_dir=None):

    bdist_tags = cfg.get('bdist_tags', {})

    # base
    bdist_cmd = 'python setup.py bdist_wheel '

    # platform
    bdist_cmd += f' --plat-name {PLATFORM_TAG} '

    # python tag
    python_tag = f'py{sys.version_info.major}{sys.version_info.minor}'
    if 'python_tag' in bdist_tags:
        python_tag = bdist_tags['python_tag']
    bdist_cmd += f' --python-tag {python_tag} '

    # dist dir
    if dist_dir is not None:
        dist_dir = osp.abspath(dist_dir)
        bdist_cmd += f' --dist-dir {dist_dir} '
    return bdist_cmd


def clear_mmdeploy(mmdeploy_dir: str):
    logging.info(f'cleaning mmdeploy: {mmdeploy_dir}')

    def _remove_in_mmdeploy(path):
        remove_dir = osp.join(mmdeploy_dir, path)
        _remove_if_exist(remove_dir)

    # remove build file
    _remove_in_mmdeploy('build')

    # remove dist
    _remove_in_mmdeploy('dist')

    # remove installed library
    _remove_in_mmdeploy('mmdeploy/lib')

    # remove onnx2ncnn and ncnn ext
    _remove_in_mmdeploy('mmdeploy/backend/ncnn/onnx2ncnn')
    _remove_in_mmdeploy('mmdeploy/backend/ncnn/onnx2ncnn.exe')
    ncnn_ext_paths = glob(
        osp.join(mmdeploy_dir, 'mmdeploy/backend/ncnn/ncnn_ext.*'))
    for ncnn_ext_path in ncnn_ext_paths:
        os.remove(ncnn_ext_path)


def build_mmdeploy(cfg, mmdeploy_dir, dist_dir=None):
    cmake_flags = cfg.get('cmake_flags', [])
    cmake_envs = cfg.get('cmake_envs', dict())

    args = [f'-D{k}={v}' for k, v in cmake_envs.items()]

    # clear mmdeploy
    clear_mmdeploy(mmdeploy_dir)

    build_dir = osp.join(mmdeploy_dir, 'build')
    if not osp.exists(build_dir):
        os.mkdir(build_dir)

        # cmake cmd
        cmake_cmd = ' '.join(['cmake ..'] + cmake_flags + args)
        _call_command(cmake_cmd, build_dir)

    if sys.platform == 'win32':
        # build cmd
        build_cmd = 'cmake --build . --config Release -- /m'
        _call_command(build_cmd, build_dir)
        install_cmd = 'cmake --install . --config Release'
        _call_command(install_cmd, build_dir)
    else:
        # build cmd
        build_cmd = 'cmake --build . -- -j$(nproc) && cmake --install .'
        _call_command(build_cmd, build_dir)

    # build wheel
    bdist_cmd = _create_bdist_cmd(cfg, dist_dir)
    _call_command(bdist_cmd, mmdeploy_dir)


def get_dir_name(cfg, tag, default_name):
    if tag not in cfg:
        logging.warning(f'{tag} not found, use `{default_name}` as default.')
    else:
        default_name = cfg[tag]
    return cfg, default_name


def create_package(cfg: Dict, mmdeploy_dir: str):
    build_dir = 'build'
    sdk_tar_name = 'sdk'

    # load flags
    cfg, build_dir = get_dir_name(cfg, 'BUILD_NAME', build_dir)
    cmake_envs = cfg.get('cmake_envs', dict())
    build_sdk_flag = cmake_envs.get('MMDEPLOY_BUILD_SDK', False)
    if 'TAR_NAME' in cfg:
        cfg, sdk_tar_name = get_dir_name(cfg, 'TAR_NAME', sdk_tar_name)

    # add version tag
    version_file = osp.join(mmdeploy_dir, 'mmdeploy', 'version.py')
    version_id = get_version(version_file)
    build_dir = build_dir + '-' + version_id

    # create package directory.
    if osp.exists(build_dir):
        logging.info(f'{build_dir} existed, deleting...')
        shutil.rmtree(build_dir)
    os.mkdir(build_dir)

    logging.info(f'build mmdeploy in {build_dir}:')
    logging.debug(f'with config: {cfg}')

    try:
        # build dist
        dist_dir = osp.join(build_dir, 'dist')
        build_mmdeploy(cfg, mmdeploy_dir, dist_dir=dist_dir)

        if build_sdk_flag:

            sdk_tar_dir = osp.join(build_dir, sdk_tar_name)

            # copy lib and install into sdk dir
            install_dir = osp.join(mmdeploy_dir, 'build/install/')
            _copy(install_dir, sdk_tar_dir)
            _remove_if_exist(osp.join(sdk_tar_dir, 'example', 'build'))

            # create sdk python api wheel
            # for linux
            python_api_lib_path = glob(
                osp.join(mmdeploy_dir, 'build/lib/mmdeploy_python.*.so'))
            # for windows
            python_api_lib_path += glob(
                osp.join(mmdeploy_dir, 'build/bin/*/mmdeploy_python.*.pyd'))
            num_libs = len(python_api_lib_path)
            if num_libs != 1:
                logging.info('find multiple mmdeploy_python libraries.')
            python_api_lib_path = python_api_lib_path[0]

            sdk_python_package_dir = osp.join(build_dir, '.mmdeploy_python')
            _copy(PACKAGING_DIR, sdk_python_package_dir)
            _copy(
                osp.join(mmdeploy_dir, 'mmdeploy', 'version.py'),
                osp.join(sdk_python_package_dir, 'mmdeploy_python',
                         'version.py'))
            _copy(python_api_lib_path,
                  osp.join(sdk_python_package_dir, 'mmdeploy_python'))
            sdk_wheel_dir = osp.abspath(osp.join(sdk_tar_dir, 'python'))
            bdist_cmd = _create_bdist_cmd(cfg, sdk_wheel_dir)
            _call_command(bdist_cmd, sdk_python_package_dir)

            # remove temp package dir
            _remove_if_exist(sdk_python_package_dir)

        logging.info('build finish.')

    except CalledProcessError:
        logging.error('build failed')
        exit()


def parse_args():
    parser = argparse.ArgumentParser(description='Build mmdeploy from yaml.')
    parser.add_argument('build_cfgs', help='The build config yaml file.')
    parser.add_argument('mmdeploy_dir', help='The source code of MMDeploy.')
    args = parser.parse_args()

    return args


def parse_configs(cfg_path: str):
    with open(cfg_path, mode='r') as f:
        cfgs = yaml.load(f, yaml.Loader)

    global_cfg = cfgs.get('global_config', dict())
    local_cfgs = cfgs.get('local_configs', [])

    merged_cfgs = [
        _merge_cfg(global_cfg, local_cfg) for local_cfg in local_cfgs
    ]

    return merged_cfgs


def main():
    args = parse_args()
    cfgs = parse_configs(args.build_cfgs)
    mmdeploy_dir = osp.abspath(args.mmdeploy_dir)
    logging.info(f'Using mmdeploy_dir: {mmdeploy_dir}')

    logging.info(f'Using PACKAGING_DIR: {PACKAGING_DIR}')

    for cfg in cfgs:
        create_package(cfg, mmdeploy_dir)


if __name__ == '__main__':
    main()
