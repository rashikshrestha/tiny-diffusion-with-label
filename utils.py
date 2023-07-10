import numpy as np
from read_write_model import read_images_text, qvec2rotmat, rotmat2qvec

def normalize_qt(qt):
    """
    Parameters
    ----------
    qt: np.ndarray
        (N, 7)

    Returns
    -------
    qt: np.ndarray
        qt with normalized q (N,7)
    """
    q = qt[:, :4]
    q_norm = np.linalg.norm(q, axis=1)
    q = q/q_norm.reshape(-1,1)
    qt[:, :4] = q
    return qt 


def get_identity_cam(scale=1):
    """
    Gets an identity camera plot

    Parameters
    ----------
    scale: int
        Scale of the camera plot

    Returns
    -------
    cam_plot: np.ndarray
        (11, 3) Sequence of points to make camera plot
    """
    f = 10
    unit_cam = np.array([
        [0,0,0],
        [3,-2,f],
        [3,2,f],
        [-3,2,f],
        [-3,-2,f],
        [0,-4,f]
    ])

    seq = np.array([3,4,1,2,0,1,5,4,0,3,2])
    draw_cam = unit_cam[seq]

    return draw_cam*scale


def get_cam_plot(qvec, tvec, unit_cam):
    """
    Gets camera plot, pose defined by given quaternion and translation

    Parameters
    ----------
    qvec: np.ndarray
        (4,) Quaternion of Camera Pose
    tvec: np.ndarray
        (3,) Translation of Camera Pose
    unit_cam: np.ndarray
        (11,3) Ideal camera sequence

    Returns
    -------
    cam: np.ndarray
        (11,3) Sequence of corrdinates to draw given camera pose
    """
    R = qvec2rotmat(qvec)
    return (R@unit_cam.T + tvec.reshape(3,1)).T