import glob
import os.path as osp


def get_ops_path() -> str:
    """Get path of the torchscript extension library.

    Returns:
        str: A path of the torchscript extension library.
    """
    wildcard = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            '../../../build/lib/libmmdeploy_torchscript_ops.so'))

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path
