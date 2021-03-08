import numpy as np
import pydensecrf.densecrf as dcrf


def dense_crf(img, output_probs):
    n_class = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    # output_probs = np.expand_dims(output_probs, 0)
    # output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, n_class)
    U = -np.log(output_probs)
    U = U.reshape((n_class, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3,
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
