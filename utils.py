import numpy as np
from read_write_model import read_images_text, qvec2rotmat, rotmat2qvec
from scipy.spatial.transform import Rotation as R


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


def diff_qctc(qctc, qctc_gt):
    """
    Find difference between two qctc

    Parameters
    ---------
    qctc: np.ndarray
        (N,7) q might be un-normalized
    qctc2_gt: np.ndarray
        (7,) q is normalized, since it is GT

    Returns
    -------
    mse: float
        Mean Square error between two qctc
    yaw,pitch,roll: float
        Angle errors
    trans: float
        Translational error
    """
    # print(qctc.shape, qctc_gt.shape)
    #! Normalize qc (just to be sure)
    qctc = normalize_qt(qctc)

    #! MSE
    mse = np.sum(np.square(qctc-qctc_gt), axis=1)/qctc_gt.shape[0] # Error in each sample of N
    mean_mse = np.mean(mse)

    #! YPR error
    R_gt = qvec2rotmat(qctc_gt[:4])

    eul_diff = []
    for qt in qctc:
        r = qvec2rotmat(qt[:4])
        r_diff = R_gt.T@r
        r_diff_scipy = R.from_matrix(r_diff)
        eul = r_diff_scipy.as_euler('zxy', degrees=True)
        eul_diff.append(eul)
    eul_diff = np.array(eul_diff)
    eul_diff_mean = np.mean(np.abs(eul_diff), axis=0)

    #! Trans error
    trans_error = np.sqrt(np.sum(np.square(qctc[:,4:]-qctc_gt[4:]), axis=1))
    trans_error_mean = np.mean(trans_error)

    return mean_mse, eul_diff_mean, trans_error_mean