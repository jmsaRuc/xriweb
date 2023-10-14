import cv2
import matplotlib.pyplot as plt
from PIL import Image

from xriweb.settings import settings
from xriweb.web.xri import plot

plt.switch_backend("agg")
cv2.destroyAllWindows()
plt.close("all")

temp2dir: str = settings.modeltemp2_dir


def reshapeAndPlot(reside_image2):
    img_size = 224
    reside_image2.save(settings.modeltemp2_dir)

    read_image2 = cv2.imread(temp2dir)
    reside_image2.close()
    read_image25 = cv2.resize(read_image2, (img_size, img_size))

    loadedImage2 = read_image25.reshape(img_size, img_size, 3)
    temp2 = runplot(loadedImage2)
    plt.close("all")
    plt.close()
    temp = Image.open(settings.modeltemp_dir)
    reside_image2.close()
    return temp


def runplot(temp):
    plot(temp)
    plt.close("all")
    plt.close()
