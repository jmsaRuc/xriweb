import cv2
import matplotlib.container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import Model, model_from_json

from xriweb.settings import settings

plt.switch_backend("agg")

plt.close("all")

"""def resource_path(relative_path):
     #Get absolute path to resource, works for dev and for PyInstaller
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)"""

fI = open(settings.modeltext_dir, "r")
modelLode = int(fI.read())
fI.close()

if modelLode == 1:
    json_file = open()
    loaded_model2_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model2_json)

    model.load_weights()

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    class_info = {0: "Normal", 1: "Pneumonia"}
    z = 1
    f = 2

elif modelLode == 2 or modelLode == 0:

    json_file = open(settings.modeljon_dir, "r")
    loaded_model2_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model2_json)

    model.load_weights(settings.modelm_dir)

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    z = 1.15
    f = 1
    modelLode = 2
    class_info = {0: "Pneumonia", 1: "Normal"}


def zoom_at(img, zoom=1, angle=0, coord=None):

    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(
        img,
        rot_mat,
        img.shape[1::-1],
        flags=cv2.INTER_LINEAR,
    )

    return result


def decode_prediction(pred):

    pred = tf.where(pred < 0.5, 0, 1)
    return pred.numpy()


def GradCam(model, img_array, layer_name, eps=1e-07):
    """
    Creates a grad-cam heatmap given a model and a layer name contained with that model


    Args:
    model: tf model
    img_array: (img_width x img_width) numpy array
    layer_name: str


    Returns
    uint8 numpy array with shape (img_height, img_width)

    """

    gradModel = Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(img_array, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, 0]
        # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)

    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    # heatmap = (heatmap * 255).astype("uint8")
    # return the resulting heatmap to the calling function

    return heatmap


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))


def superimpose(img_bgr, cam, thresh, emphasize=False):
    """
    Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.


    Args:
    image: (img_width x img_height x 3) numpy array
    grad-cam heatmap: (img_width x img_width) numpy array
    threshold: float
    emphasize: boolean

    Returns
    uint8 numpy array with shape (img_height, img_width, 3)

    """
    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    hif = 0.8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255).astype(
        np.uint8,
    )  # scale 0 to 255
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img_rgb


def makehistoOfZon2(imgH2):
    def maskofgrad_cam(grad_cam, imgM, thresh):
        thresholded_heatmap = (grad_cam > thresh).astype(np.uint8)
        mask = imgM * thresholded_heatmap[:, :, np.newaxis]

        return mask

    def leftRig2(masked_imageI):
        height, width = masked_imageI.shape[:2]
        x_center = width // 2

        masked_imageData = masked_imageI[:, :x_center, :]

        left_halfN = np.zeros_like(masked_imageI)
        left_halfN[:, :x_center, :] = masked_imageData
        masked_imageData2 = masked_imageI[:, x_center:, :]
        right_halfN = np.zeros_like(masked_imageI)
        right_halfN[:, x_center:, :] = masked_imageData2

        return left_halfN, right_halfN

    def polygons_mask(maskH):
        gray = cv2.cvtColor(maskH, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            gray,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons_mask = np.zeros_like(maskH)
        hull = cv2.convexHull(np.concatenate(contours))
        cv2.drawContours(
            polygons_mask,
            [hull],
            -1,
            (255, 255, 255),
            thickness=cv2.FILLED,
        )
        polygons_mask = polygons_mask.astype(np.uint8)

        return polygons_mask

    def cords3Zons(coordsZ):
        coordsZsp = [coordsZ[0], coordsZ[1]]
        coordsZsp = np.array(coordsZ)
        cords1H, cords2H, cords3H = np.split(coordsZsp, 3, axis=1)
        Coords123Zo = [cords1H, cords2H, cords3H]

        return Coords123Zo

    def image_in_mask(imgFM, coords_from_maskH):
        image_data = imgFM[coords_from_maskH[0], coords_from_maskH[1], :]
        extracted_imagei = np.zeros_like(imgFM)
        extracted_imagei[
            coords_from_maskH[0],
            coords_from_maskH[1],
            :,
        ] = image_data

        return extracted_imagei

    def redDetecter(extracted_image1H):
        colors = {}
        data = extracted_image1H
        n = 0
        color_range = (240, 255, 0, 200, 0, 200)
        for row in data:
            for pix in row:
                colorH = "{}_{}_{}".format(*pix)
                colors[colorH] = colors.get(colorH, 0)
                colors[colorH] += 1
        for colorH, count in colors.items():
            r, g, b = colorH.split("_")
            r = int(r)
            g = int(g)
            b = int(b)
            if (
                color_range[0] <= r <= color_range[1]
                and color_range[2] <= g <= color_range[3]
                and color_range[4] <= b <= color_range[5]
            ):
                n = n + 1
        return n

    Model_ActivitH = [0, 0, 0, 0, 0, 0]

    grad_camH1 = GradCam(model, np.expand_dims(imgH2, axis=0), "conv2d")
    grad_cam_superimposedH1 = superimpose(
        imgH2,
        grad_camH1,
        0.7,
        emphasize=True,
    )

    masked_imageH = maskofgrad_cam(grad_camH1, grad_cam_superimposedH1, 0.7)

    polygons_maskH = polygons_mask(masked_imageH)

    coordsH = np.where(polygons_maskH != 0)
    coords123Zon = cords3Zons(coordsH)

    grad_camH2 = GradCam(model, np.expand_dims(imgH2, axis=0), "conv2d_4")
    grad_cam_superimposedH2 = superimpose(
        imgH2,
        grad_camH2,
        0.5,
        emphasize=True,
    )

    extracted_imageHZ = (
        image_in_mask(grad_cam_superimposedH2, coords123Zon[0]),
        image_in_mask(
            grad_cam_superimposedH2,
            coords123Zon[1],
        ),
        image_in_mask(grad_cam_superimposedH2, coords123Zon[2]),
    )

    extracted_imageZ1rl = (
        leftRig2(extracted_imageHZ[0]),
        leftRig2(
            extracted_imageHZ[1],
        ),
        leftRig2(extracted_imageHZ[2]),
    )

    polygons_maskHZ = extracted_imageHZ = (
        image_in_mask(polygons_maskH, coords123Zon[0]),
        image_in_mask(
            polygons_maskH,
            coords123Zon[1],
        ),
        image_in_mask(polygons_maskH, coords123Zon[2]),
    )

    polygons_maskHZrl = (
        leftRig2(polygons_maskHZ[0]),
        leftRig2(
            polygons_maskHZ[1],
        ),
        leftRig2(polygons_maskHZ[2]),
    )

    im = imgH2

    for i, zones in enumerate(extracted_imageZ1rl):
        for c, zonesRL in enumerate(zones):

            d = 3
            d = d * c
            Model_ActivitH[i + d] = redDetecter(zonesRL)

            n = 0
            n = redDetecter(zonesRL)
            aply = 0
            sat = 0
            aply = 80 + n // 2
            n = 0
            n = redDetecter(zonesRL)
            sat = n / 250

            # f=np.ones((224,224,3),dtype=np.uint8) * 100
            masked = polygons_maskHZrl[i][c]
            # masked=np.where(masked != 0,cv2.multiply(masked,f),masked)
            masked = np.where(
                masked != 0,
                cv2.applyColorMap(
                    masked - aply,
                    cv2.COLORMAP_AUTUMN,
                ),
                masked,
            )
            maskedc = cv2.Canny(masked, 100, 200)
            contours, hierarchy = cv2.findContours(
                maskedc,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            cv2.drawContours(masked, contours, -1, (0, 0, 0), 2)
            im = masked * sat + im

    im = np.minimum(im, 255).astype(np.uint8)
    zoneimg = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    Chest_Radiograph_ZonesH = [
        "Left lung: Apical and Upper",
        "Lef lung: Mid",
        "Lef lung: Lower",
        "Right lung: Apical and Upper",
        "Right lung: Mid",
        "Right lung: Lower",
    ]
    dfH = pd.DataFrame(
        {
            "Zones": Chest_Radiograph_ZonesH,
            "Red Pixel Amount/Model Activity": Model_ActivitH,
        },
    )
    return dfH, zoneimg


def plot(img):
    plt.close("all")
    # b=-1
    # imgH=x_test[b]
    imgH = img
    imgzH = zoom_at(imgH, z)

    imgH = imgzH

    pred_raw = model.predict(np.expand_dims(imgH, axis=0))[0][0]
    pred = decode_prediction(pred_raw)
    pred_label = class_info[pred]

    if pred_raw < 0.5:
        predP = 1 - pred_raw
    else:
        predP = pred_raw

    grad_cam = GradCam(model, np.expand_dims(imgH, axis=0), "conv2d_4")
    grad_cam_superimposed = superimpose(imgH, grad_cam, 0.5, emphasize=True)

    sns.set_theme(
        context="talk",
        style="ticks",
        rc=plt.style.library["dark_background"],
    )
    sns.set_style(
        {
            "grid.color": "darkorange",
            "text.color": "antiquewhite",
            "axes.edgecolor": "antiquewhite",
            "axes.labelcolor": "antiquewhite",
            "text.color": "antiquewhite",
            "xtick.color": "antiquewhite",
            "ytick.color": "antiquewhite",
            "patch.edgecolor": "antiquewhite",
            "patch.force_edgecolor": True,
        },
    )

    matplotlib.rcParams["axes.grid"] = False
    matplotlib.rcParams["savefig.transparent"] = False
    fig = plt.figure(figsize=(30, 30), facecolor="black")
    axs1 = plt.subplot2grid((4, 3), (0, 0), zorder=1)
    plt.imshow(imgH)
    ax1 = axs1.axis()
    recL = plt.Rectangle(
        (ax1[0] - 0.2, ax1[2] - 223.55),
        (ax1[1] - ax1[0]) + 0.8,
        (ax1[3] - ax1[2]) + 208,
        fill=False,
        lw=1.8,
        linestyle="-",
        color="darkorange",
    )
    rec1B = plt.Rectangle(
        (ax1[0] - 0.2, ax1[2] - 0),
        (ax1[1] - ax1[0]) + 0.8,
        (ax1[3] - ax1[2]) + 0.4,
        fill=False,
        lw=1.8,
        linestyle="-",
        color="antiquewhite",
    )
    rec1B = axs1.add_patch(rec1B)
    # recL = axs1.add_patch(recL)
    recL.set_clip_on(False)
    rec1B.set_clip_on(False)
    plt.axis("off")
    plt.title(pred_label, color="antiquewhite", pad=5, size=25)
    if pred_label == "Pneumonia":
        Tx = 370
    else:
        Tx = 349
    plt.text(
        x=(ax1[2] - Tx),
        y=(ax1[0] + 77),
        s="Model preidicts: " + pred_label + " -",
    )
    plt.text(
        x=(ax1[2] - 381.5),
        y=(ax1[0] + 151),
        s="Model confidence: " + str("{0:.5%}".format(predP)) + " -",
    )
    axs2 = plt.subplot2grid((4, 3), (0, 1))

    plt.imshow(grad_cam_superimposed, zorder=2)

    plt.axis("off")
    plt.title("Grad-CAM Heat-map", color="antiquewhite", pad=5, size=25)

    ax2 = axs2.axis()
    rec2L = plt.Rectangle(
        (ax2[0] - 0.2, ax2[2] - 223.55),
        (ax2[1] - ax2[0]) + 0.8,
        (ax2[3] - ax2[2]) + 208,
        fill=False,
        lw=1.8,
        linestyle="-",
        color="darkorange",
        zorder=2,
    )
    rec3B = plt.Rectangle(
        (ax2[0] - 0.2, ax2[2] + 0),
        (ax2[1] - ax2[0]) + 0.6,
        (ax2[3] - ax2[2]) + 0.4,
        fill=False,
        lw=1.8,
        linestyle="-",
        color="antiquewhite",
        zorder=2,
    )
    rec3B = axs2.add_patch(rec3B)
    # rec2L= axs2.add_patch(rec2L)
    rec2L.set_clip_on(False)
    rec3B.set_clip_on(False)
    l = 0
    recL = plt.Rectangle(
        (ax2[0] - 556, ax2[2] - 112),
        (ax2[1] - ax2[0]) + 957,
        (ax2[3] - ax2[2]) + 224,
        fill=False,
        lw=1.8,
        linestyle="-",
        color="antiquewhite",
    )
    recL = axs2.add_patch(recL)
    recL1 = plt.Rectangle(
        (ax2[0] - 556, ax2[2] - 186.66),
        (ax2[1] - ax2[0]) + 957,
        (ax2[3] - ax2[2]) + 224,
        fill=False,
        lw=1.8,
        linestyle="-",
        color="antiquewhite",
    )
    recL1 = axs2.add_patch(recL1)
    recL2 = plt.Rectangle(
        (ax2[0] - 556, ax2[2] - 37.34),
        (ax2[1] - ax2[0]) + 957,
        (ax2[3] - ax2[2]) + 224,
        fill=False,
        lw=1.8,
        linestyle="-",
        color="antiquewhite",
    )
    recL2 = axs2.add_patch(recL2)
    recL.set_clip_on(False)
    recL1.set_clip_on(False)
    recL2.set_clip_on(False)
    df, zoneImag = makehistoOfZon2(imgH)
    axs3 = plt.subplot2grid((4, 3), (0, 2))
    plt.imshow(zoneImag, zorder=3)
    ax3 = axs3.axis()
    rec4B = plt.Rectangle(
        (ax3[0] - 0.2, ax3[2] + 0),
        (ax3[1] - ax3[0]) + 0.6,
        (ax3[3] - ax3[2]) + 0.4,
        fill=False,
        lw=1.8,
        linestyle="-",
        color="antiquewhite",
        zorder=4,
    )
    rec4B = axs3.add_patch(rec4B)
    rec4B.set_clip_on(False)
    plt.axis("off")
    plt.title("Zones", color="antiquewhite", pad=5, size=25)
    soRH = df.sort_values(
        "Red Pixel Amount/Model Activity",
        ascending=False,
    ).Zones
    soRH2 = df.sort_values(
        "Red Pixel Amount/Model Activity",
        ascending=False,
    ).Zones
    palette = sns.color_palette("YlOrBr_r", n_colors=6, as_cmap=False)
    axs3 = plt.subplot2grid((4, 4), (1, 0), colspan=4)
    g = sns.barplot(
        data=df,
        x="Red Pixel Amount/Model Activity",
        y="Zones",
        hue="Zones",
        order=soRH,
        hue_order=soRH2,
        palette=palette,
        dodge=False,
        ax=axs3,
    )
    for bars in g.containers:
        g.bar_label(bars, padding=5)
    plt.legend(loc="lower right", title="Team")
    plt.savefig(settings.modeltemp_dir, dpi=70, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")
