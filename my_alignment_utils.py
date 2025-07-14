import numpy as np

def umeyama_sim3(src_pts: np.ndarray,
                 dst_pts: np.ndarray,
                 with_scale: bool = True):
    """
    Estimate the similarity transform (Sim-3) that aligns `src_pts` to `dst_pts`.

    Args
    ----
    src_pts : (N, 3) ndarray
        3-D points in the *source* frame.
    dst_pts : (N, 3) ndarray
        Corresponding 3-D points in the *destination* frame.
    with_scale : bool, default True
        If False, returns an SE(3) transform (scale = 1).

    Returns
    -------
    s : float
        Scale factor. 1.0 when with_scale == False.
    R : (3, 3) ndarray
        Rotation matrix.
    t : (3,) ndarray
        Translation vector such that  x_dst = s · R · x_src + t
    """
    assert src_pts.shape == dst_pts.shape, "Source and destination must have same shape"
    assert src_pts.ndim == 2 and src_pts.shape[1] == 3, "Points must be (N, 3)"

    N = src_pts.shape[0]
    if N < 3:
        raise ValueError("At least 3 points are required")

    # 1. Compute centroids
    mu_src = src_pts.mean(axis=0)
    mu_dst = dst_pts.mean(axis=0)

    src_demean = src_pts - mu_src
    dst_demean = dst_pts - mu_dst

    # 2. Compute covariance matrix
    Sigma = dst_demean.T @ src_demean / N

    # 3. SVD on covariance
    U, D, Vt = np.linalg.svd(Sigma)

    # 4. Construct rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:            # reflection case
        Vt[-1, :] *= -1
        R = U @ Vt

    # 5. Scale
    if with_scale:
        var_src = (src_demean ** 2).sum() / N
        s = (D @ np.ones_like(D)) / var_src
    else:
        s = 1.0

    # 6. Translation
    t = mu_dst - s * R @ mu_src

    return s, R, t
