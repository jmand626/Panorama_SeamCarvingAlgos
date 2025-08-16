import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above. If you use convolve(), remember to
        specify zero-padding to match our equations, for example:

            out_image = convolve(in_image, kernel, mode='constant', cval=0)

        You can also use for nested loops compute M and the subsequent Harris
        corner response for each output pixel, intead of using convolve().
        Your implementation of conv_fast or conv_nested in HW1 may be a
        useful reference!

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # 1. Compute x and y derivatives (I_x, I_y) of an image
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Steps 2 and 3
    convolve_x = convolve(dx ** 2, window, mode='constant', cval=0)
    convolve_y = convolve(dy ** 2, window, mode='constant', cval=0)
    convolve_xy = convolve(dx * dy, window, mode='constant', cval=0)

    # Steps 4 and 5. Cannot use the numpy funcs for determinants or trace
    # because we dont have the matrix M
    response = (convolve_x * convolve_y - (convolve_xy ** 2)) \
    - k * (convolve_x + convolve_y) ** 2
    # Because python is stupid, we have to do a backslash for this split
    # to not throw an error

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        In this case of normalization, if a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    mean = np.mean(patch)
    std = np.std(patch)

    if std == 0.0:
      feature = (patch - mean)
    else:
      feature = (patch - mean) / std

    # Flatten out to 1-d
    feature = feature.flatten()
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be STRICTLY SMALLER
    than the threshold (NOT equal to). Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

        The Scipy function cdist calculates Euclidean distance between all
        pairs of inputs
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    M = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i, row in enumerate(dists):
      # For each index and element

      # Parition takes the second param as to the put what should be there
      # if the array was sorted is there. We only need to get the two smallest
      # distances, so since it leaves those less than it to the left in some order
      # but there would only be one value to the left of it, we slice these
      # sorted pieces knowing we have already sorted what we need
      two = np.partition(row, 1)[:2]
      # And find the best descriptor from image 1 for these
      one_best = np.argmin(row)

      # Simple threshold check
      if (two[0] / two[1]) < threshold:
        matches.append([i, one_best])

    matches = np.array(matches)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return matches


def fit_affine_matrix(p1, p2):
    """
    Fit affine matrix such that p2 * H = p1. First, pad the descriptor vectors
    with a 1 using pad() to convert to homogeneous coordinates, then return
    the least squares fit affine matrix in homogeneous coordinates.

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None
        to suppress deprecation warnings, and match the autograder.

    Args:
        p1: an array of shape (M, P) holding descriptors of size P about M keypoints
        p2: an array of shape (M, P) holding descriptors of size P about M keypoints

    Return:
        H: a matrix of shape (P+1, P+1) that transforms p2 to p1 in homogeneous
        coordinates
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # I thought we would also have to pad this at least, but I guess not..
    H = np.linalg.lstsq(p2, p1, rcond=None)[0]
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation:

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers via Euclidean distance
        4. Keep the largest set of inliers (use >, i.e. break ties by whichever set is seen first)
        5. Re-compute least-squares estimate on all of the inliers

    Update max_inliers as a boolean array where True represents the keypoint
    at this index is an inlier, while False represents that it is not an inlier.

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None
        to suppress deprecation warnings, and match the autograder.

        You can compute elementwise boolean operations between two numpy arrays,
        and use boolean arrays to select array elements by index:
        https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N, dtype=bool)
    n_inliers = 0

    # RANSAC iteration start

    # Note: while there're many ways to do random sampling, we use
    # `np.random.shuffle()` followed by slicing out the first `n_samples`
    # matches here in order to align with the auto-grader.
    # Sample with this code: 
    for i in range(n_iters):
        # 1. Select random set of matches
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        sample1 = pad(keypoints1[samples[:,0]])
        sample2 = pad(keypoints2[samples[:,1]])
    
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        temp_n = 0
        currinliner_match = np.zeros(N, dtype=np.int32)
        # max_inliners is for the running max, so obviously want temp
        # variables to calculate this seperately

        H = np.linalg.lstsq(sample2, sample1, rcond=None)[0]
        # Computed affine matrix with new parameter

        H[:, 2] = np.array([0, 0, 1])
        # Force 3rd column to be 1 as affine/homography matrices are

        currinliner_match = np.linalg.norm(matched2.dot(H) -matched1, axis=1) < threshold
        temp_n = np.sum(currinliner_match)
        # Specifically find inliers with Eucilidean distance through the all matched2
        # due to the norm. As mentioned before and in oh, the forcing of [0,0,1] 
        # previously makes the dot product work here

        if temp_n > n_inliers:
            # Obviously just take a new maximze
            max_inliers = currinliner_match.copy()
            n_inliers = temp_n

    # Recompute estimate on all inliers
    H = np.linalg.lstsq(matched2[max_inliers],matched1[max_inliers], rcond=None)[0]
    H[:, 2] = np.array([0, 0, 1])

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten 3D matrix of histograms into a 1D feature vector.
    4. Normalize flattened histogram feature vector by L2 norm
       Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    hists = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(rows):
      for j in range(cols):
        G_curr = G_cells[i, j]
        theta_curr = theta_cells[i, j]
        hist_curr = np.zeros(n_bins)

        G_flat = G_curr.flatten()
        theta_flat = theta_curr.flatten()

        for l in range(len(G_flat)):
          # For each pixel in the current cell, because it was mentioned above
          # that each G cell of shape (M, N) has M*N pixels per cell
          hist_curr[int(theta_flat[l] //degrees_per_bin)% n_bins] += G_flat[l]
          # Floor division to assign into histograph by orinetation angle, which
          # was mentioned in the instructions
        hists[i, j] = hist_curr


    # Flatten 3d matrix of histograms
    block = hists.flatten()
    # use l2 norm to normalize histogram feature vector
    norm = np.linalg.norm(block)
    if norm > 0:
      block /= norm 
         
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### YOUR CODE HERE

    return block


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # As mentioned above, the margin definitons were already done
    w = np.ones(out_W, dtype=float)
    w[:left_margin] = 1.0 
    # Create weight vector, left to left amrgin is 1.

    width = right_margin - left_margin
    # "From the left margin to the right margin, 
    # the weight linearly decrements from 1 to 0"
    # linspace places values at regularly scheduled intervals
    w[right_margin:] = 0.0
    if width>0:
        w[left_margin:right_margin] = np.linspace(1.0, 0.0, width)

    # Now use tile as mentioned above to work with weight masks
    w1 = np.tile(w,(out_H, 1)) * img1_mask
    w2 = np.tile(1.0 - w, (out_H, 1)) * img2_mask
    # Remember that tile repeats out_H the height of the image times
    # without changing the width, translating our vector into a matrix
    # The binary maskes show the pixels for where the image data is valid
    # By multiplying with the image, we apply the weighted masks only to
    # the valid pixels per image, and specficially with the inverse weights
    # for image 2 because if we dont, then we havent accomplished anything
    # different. The whole point here was to blend instead of simply add
    # or double with the same weights.

    merged = (img1_warped*w1) + (img2_warped*w2)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 1) "every neighboring pair of images" (Ii+1 - Ii) via RANSAC
    transforms = [np.identity(3)]
    # We start out with the identity matrix because we want to start out
    # with the base, unedited images but with them already put into the
    # stitching process
    for idx, m in enumerate(matches):
        H_est, _ = ransac(keypoints[idx], keypoints[idx+1], m)
        transforms.append(H_est)
        # Ransac does its thing over the pair of images that we send in,
        # and we just send in the images at idx and idx+1 because thats easier
        # The H matrix it returns is our transform for this pair

    for j in range(1,len(transforms)):
        transforms[j] = transforms[j-1]@ transforms[j]
        # To be clear, the previous step calculated the matrices, but didnt
        # apply them together. That is what this does

    image_pairs = []
    # Combined images to transform matrices

    # Use utility function used throughout the notebook
    final_shape, offset = get_output_space(imgs[0], imgs, transforms)

    for img, H in zip(imgs, transforms):
        W = warp_image(img, H, final_shape, offset)
        # Mentioned in OH (finally...), warp_image (also from utils)
        # adds in the values of -1 for values of new pixels that dont
        # correspond with this

        real = (W != -1)
        # bitmask to assign true to real pixel values//not -1
        W[~real] = 0
        image_pairs.append(W)

    # Orignally i used linear_blend, but the instructions explictally
    # said to avoid it, so we just dont do that
    panorama = image_pairs[0]
    for W in image_pairs[1:]:
      panorama[(W > 0)] = W[(W > 0)]
      # copy valid pixels into panorma

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return panorama


