"""Correct the legacy B2D image-crop projection without rewriting caches."""


def corrected_front_projection(projection):
    """Consume the legacy 1600x900, crop-28, resize-512x256 cache contract.

    A projection returns [u*depth, v*depth, depth, 1]. Cropping must subtract
    28*scale*depth from its second coordinate, not the constant 28*scale.
    The shared cache and legacy online builder both contain the old matrix;
    apply this correction once at the model boundary in both paths.
    """
    out = projection.clone()
    crop = 28.0 * 256.0 / (900.0 - 56.0)
    out[..., 1, 3] += crop
    out[..., 1, :] -= crop * projection[..., 2, :]
    return out
