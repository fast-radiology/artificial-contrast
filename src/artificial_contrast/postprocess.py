# TODO: Develop one for contrast
def postprocess(image):
    image = image.clone().byte()

    num_img, _, _ = image.shape
    for i in range(1, num_img - 1):
        image[i] |= image[i - 1] & image[i + 1]
        image[i] &= image[i - 1] | image[i + 1]
    return image.long()
