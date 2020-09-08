import os
import os.path as op
import re
import numpy as np
import subprocess
from contextlib import contextmanager
import logging


def _vc_home():
    """Find philly's VC home in scratch space
    :rtype: str
    """
    home = os.environ.get('PHILLY_VC_NFS_DIRECTORY',
                          os.environ.get('PHILLY_VC_DIRECTORY'))
    if not home:
        home = op.expanduser('~')
        home = '/'.join(home.split('/')[:5])
    return home


_VC_HOME = _vc_home()


def vc_name():
    """Find philly's VC name
    :rtype: str
    """
    name = os.environ.get('PHILLY_VC')
    if name:
        return name
    name = op.basename(_VC_HOME)
    if name:
        return name
    return op.basename(op.dirname(_VC_HOME))


_VC_NAME = vc_name()


def _vc_hdfs_base():
    base = os.environ.get("PHILLY_DATA_DIRECTORY") or os.environ.get(
        "PHILLY_HDFS_PREFIX")
    if base:
        return base
    for base in ["/hdfs", "/home"]:
        if op.isdir(base):
            return base
    return _VC_HOME


def vc_hdfs_root():
    """Find the HDFS root of the VC
    :rtype: str
    """
    path = os.environ.get('PHILLY_VC_HDFS_DIRECTORY')
    if path:
        return path
    path = op.join(
        os.environ.get('PHILLY_HDFS_PREFIX', _vc_hdfs_base()), _VC_NAME)
    return path


_VC_HDFS_ROOT = vc_hdfs_root()


def expand_vc_user(path):
    """Expand ~ to VC's home
    :param path: the path to expand VC user
    :type path: str
    :return:/var/storage/shared/$VC_NAME
    :rtype: str
    """
    if path.startswith('~'):
        path = op.abspath(op.join(_VC_HOME, '.' + path[1:]))

    return path


def abspath(path, roots=None):
    """Expand ~ to VC's home and resolve relative paths to absolute paths
    :param path: the path to resolve
    :type path: str
    :param roots: CWD roots to resolve relative paths to them
    :type roots: list
    """
    path = expand_vc_user(path)
    if op.isabs(path):
        return path
    if not roots:
        roots = ["~"]
    roots = [expand_vc_user(root) for root in roots]
    for root in roots:
        resolved = op.abspath(op.join(root, path))
        if op.isfile(resolved) or op.isdir(resolved):
            return resolved
    # return assuming the first root (even though it does not exist)
    return op.abspath(op.join(roots[0], path))


def job_id(path=None):
    """Get the philly job ID (from a path)
    :param path:Path to seach for app id
    :rtype: str
    """
    if path is None:
        return os.environ.get('PHILLY_JOB_ID') or job_id(op.expanduser('~'))
    m = re.search('/(?P<app_id>application_[\d_]+)[/\w]*$', path)
    if m:
        return m.group('app_id')
    return ''


def get_model_path(path=None):
    """Find the default location to output/models
    """
    return abspath(
        op.join('sys', 'jobs', job_id(path), 'models'), roots=[vc_hdfs_root()])


def get_master_machine():
    mpi_host_file = op.expanduser('~/mpi-hosts')
    with open(mpi_host_file, 'r') as f:
        master_name = f.readline().strip()
    return master_name


def get_master_ip(master_name=None):
    if master_name is None:
        master_name = get_master_machine()
    etc_host_file = '/etc/hosts'
    with open(etc_host_file, 'r') as f:
        name_ip_pairs = f.readlines()

    print('name_ip_pairs: {}'.format(name_ip_pairs))
    print('master_name: {}'.format(master_name))

    name2ip = {}
    for name_ip_pair in name_ip_pairs:
        pair_list = name_ip_pair.split(' ')
        key = pair_list[2].strip()
        value = pair_list[0]
        name2ip[key] = value
        print('{}, key: {}, value: {}'.format(name_ip_pair, key, value))
    return name2ip[master_name]

def ompi_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)


def ompi_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)


def ompi_local_rank():
    """Find OMPI local rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') or 0)


def ompi_local_size():
    """Find OMPI local size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE') or 1)


def ompi_universe_size():
    """Find OMPI universe size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_UNIVERSE_SIZE') or 1)


@contextmanager
def run_and_terminate_process(*args, **kwargs):
    """Run a process and terminate it at the end
    """
    p = None
    try:
        p = subprocess.Popen(*args, **kwargs)
        yield p
    finally:
        if not p:
            return
        try:
            p.terminate()  # send sigterm
        except OSError:
            pass
        try:
            p.kill()  # send sigkill
        except OSError:
            pass


def get_gpus_nocache():
    """List of NVIDIA GPUs
    """
    cmds = 'nvidia-smi --query-gpu=name --format=csv,noheader'.split(' ')
    with run_and_terminate_process(
            cmds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1) as process:
        return [
            str(line).strip() for line in iter(process.stdout.readline, b'')
        ]


_GPUS = get_gpus_nocache()


def get_gpus():
    """List of NVIDIA GPUs
    """
    return _GPUS


def gpu_indices(divisible=True):
    """Get the GPU device indices for this process/rank
    :param divisible: if GPU count of all ranks must be the same
    :rtype: list[int]
    """
    local_size = ompi_local_size()
    local_rank = ompi_local_rank()
    assert 0 <= local_rank < local_size, \
        "Invalid local_rank: {} local_size: {}".format(local_rank, local_size)
    gpu_count = len(get_gpus())
    assert gpu_count >= local_size > 0, \
        "GPU count: {} must be >= LOCAL_SIZE: {} > 0".format(gpu_count, local_size)
    if divisible:
        ngpu = int(gpu_count / local_size)
        gpus = np.arange(local_rank * ngpu, (local_rank + 1) * ngpu)
        if gpu_count % local_size != 0:
            logging.warning(
                "gpu_count: {} not divisible by local_size: {}; " + "some GPUs may be unused"
                .format(gpu_count, local_size))
    else:
        gpus = np.array_split(range(gpu_count), local_size)[local_rank]
    return gpus.astype(int)
