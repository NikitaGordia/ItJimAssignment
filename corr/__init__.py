import numpy as np
import matplotlib.image as img

# Implementation of 2D correlation for two images
def corr2D(a_raw, b_raw):
    a, b = np.array(a_raw), np.array(b_raw)

    # Check for dimensions
    if len(a.shape) != 2:
        raise Exception("Wrong shape for the first matrix")
    if len(b.shape) != 2:
        raise Exception("Wrong shape for the second matrix")

    # Calc output shape
    out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
    out = np.empty(out_shape)

    # Generate extended version of a
    a_ext = np.zeros(
        (a.shape[0] + 2 * (b.shape[0] - 1),
         a.shape[1] + 2 * (b.shape[1] - 1))
    )
    a_ext_shift = b.shape[0] - 1, b.shape[1] - 1
    a_ext[a_ext_shift[0]:(a_ext_shift[0] + a.shape[0]), a_ext_shift[1]:(a_ext_shift[1] + a.shape[1])] = a

    # Make calculations
    for i, j in np.ndindex(out_shape):
        out[i, j] = (a_ext[i:(i + b.shape[0]), j:(j + b.shape[1])] * b).sum()

    return out

# Makes correlation of a and b, and stores result in result_path
def corr2D_image(a, b, result_path):
    res = corr2D(a, b)
    img.imsave(result_path, res)