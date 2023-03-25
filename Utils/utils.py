from PIL import Image

def keep_image_size_open(path, size=(256,256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('rgb', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask