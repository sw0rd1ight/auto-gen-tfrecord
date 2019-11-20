from PIL import Image, ImageEnhance
from pylab import *

import random

import os
import tensorflow as tf

from dataset_utils import int64_feature, float_feature, bytes_feature
from label import VOC_LABELS


def gRotImg(bimg_name, timg_name, dimg_name, d_shapes=500, tleast_shape=50, tmost_shape=400, t_num=1):
    bboxes = []
    labels = []
    labels_text = []

    label = os.path.basename(timg_name)
    label, _ = os.path.splitext(label)
    label = getLabel(label)
    bimg = Image.open(bimg_name)
    dimg = bimg.resize((d_shapes, d_shapes))
    timg = Image.open(timg_name)

    while t_num > 0:
        timga = timg.convert('RGBA')  # 要将转置的图片转换成RGBA格式，旋转后之前没有像素的点才是透明的
        angle = random.randint(0, 360)
        timga = timga.rotate(angle, expand=1)

        timga = distortImg(timga)

        # 这个地方长宽 1 是高 0 是宽
        if timga.size[1] > timga.size[0]:
            h_w = timga.size[1] / timga.size[0]
            height = random.randint(tleast_shape, tmost_shape)
            width = int(height / h_w)
        else:
            w_h = timga.size[0] / timga.size[1]
            width = random.randint(tleast_shape, tmost_shape)
            height = int(width / w_h)

        x_min = random.randint(0, d_shapes - width)
        y_min = random.randint(0, d_shapes - height)
        x_max = x_min + width
        y_max = y_min + height
        # timga缩放到与切剪下来的 crop 一样大
        timga = timga.resize((width, height), Image.ANTIALIAS)
        # 从背景图片窃取一个box 来供与 timg 融合
        crop = dimg.crop((x_min, y_min, x_max, y_max))
        # 原timg 与 crop 融合

        timga = Image.composite(timga, crop, mask=timga)
        # in_img = Image.blend(in_img, crop, 0.1)
        # timg.save('timgge.png')
        box = (x_min, y_min, x_max, y_max)
        # print(box)
        bbox = (float(x_min / dimg.size[0]),
                float(y_min / dimg.size[1]),
                float(x_max / dimg.size[0]),
                float(y_max / dimg.size[1]))

        # 把融合后的那一块重新放回背景图片中
        dimg.paste(timga, box)
        # draw=ImageDraw.Draw(dimg)
        # draw.rectangle(box)

        bboxes.append(bbox)
        labels.append(int(VOC_LABELS[label][0]))  # xml上面的label必须与VOC _labels 定义在common.py的一致
        labels_text.append(label.encode('ascii'))
        t_num -= 1
    dimg.save(dimg_name)

    channals = len(dimg.getbands())  # 获取通道数
    shape = [dimg.size[1],
             dimg.size[0],
             channals]
    dimg_data = tf.gfile.FastGFile(dimg_name, 'rb').read()
    return dimg_data, labels, labels_text, bboxes, shape


# 暂未启用
def gRotImgAlpha(bimg_name, timg_name, dimg_name, d_shapes=500, tleast_shape=50, tmost_shape=400, t_num=1):
    bboxes = []
    labels = []
    labels_text = []

    label = os.path.basename(timg_name);  # 获取目标图片的名字，不包括后缀名
    label, _ = os.path.splitext(label)
    label = getLabel(label)
    bimg = Image.open(bimg_name)
    dimg = bimg.resize((d_shapes, d_shapes))
    timg = Image.open(timg_name)

    while t_num > 0:
        timga = timg.convert('RGBA')  # 要将转置的图片转换成RGBA格式，旋转后之前没有像素的点才是透明的
        angle = random.randint(0, 360)
        timga = timga.rotate(angle, expand=1)

        timga = distortImg(timga)

        # 这个地方长宽 1 是高 0 是宽
        if timga.size[1] > timga.size[0]:
            h_w = timga.size[1] / timga.size[0]
            height = random.randint(tleast_shape, tmost_shape)
            width = int(height / h_w)
        else:
            w_h = timga.size[0] / timga.size[1]
            width = random.randint(tleast_shape, tmost_shape)
            height = int(width / w_h)

        x_min = random.randint(0, d_shapes - width)
        y_min = random.randint(0, d_shapes - height)
        x_max = x_min + width
        y_max = y_min + height
        # timga缩放到于切剪下来的 crop 一样大
        timga = timga.resize((width, height), Image.ANTIALIAS)
        # 从背景图片窃取一个box 来供与 timg 融合
        crop = dimg.crop((x_min, y_min, x_max, y_max))
        # crop = crop.convert('RGBA')
        # 原timg 与 crop 融合

        timga = Image.composite(timga, crop, mask=timga)
        timga = Image.blend(timga, crop, 0.5)
        # timga=timga.convert('RGBA')
        # in_img = Image.blend(in_img, crop, 0.1)
        # timg.save('timgge.png')
        box = (x_min, y_min, x_max, y_max)
        # print(box)
        bbox = (float(x_min / dimg.size[0]),
                float(y_min / dimg.size[1]),
                float(x_max / dimg.size[0]),
                float(y_max / dimg.size[1]))

        # 把融合后的那一块重新放回背景图片中
        dimg.paste(timga, box)
        # draw=ImageDraw.Draw(dimg)
        # draw.rectangle(box)

        bboxes.append(bbox)
        labels.append(int(VOC_LABELS[label][0]))  # xml上面的label必须与VOC _labels 定义在common.py的一致
        labels_text.append(label.encode('ascii'))
        t_num -= 1
    dimg.save(dimg_name)

    channals = len(dimg.getbands())  # 获取通道数
    shape = [dimg.size[1],
             dimg.size[0],
             channals]
    dimg_data = tf.gfile.FastGFile(dimg_name, 'rb').read()
    return dimg_data, labels, labels_text, bboxes, shape


def gMoveImgPNG(bimg_name, timg_name, dimg_name, d_shapes=500, tleast_shape=50, tmost_shape=400, t_num=1):
    bboxes = []
    labels = []
    labels_text = []

    label = os.path.basename(timg_name)
    label, _ = os.path.splitext(label)  # 获取目标图片的名字，不包括后缀名
    label = getLabel(label)
    bimg = Image.open(bimg_name)
    dimg = bimg.resize((d_shapes, d_shapes))
    timg = Image.open(timg_name)

    # 这个地方长宽 1 是高 0 是宽
    if timg.size[1] > timg.size[0]:
        h_w = timg.size[1] / timg.size[0]
        height = random.randint(tleast_shape, tmost_shape)
        width = int(height / h_w)
        # width = random.randint(tleast_shape, tmost_shape)
        # height = min(int(width * h_w), tmost_shape)
    else:
        w_h = timg.size[0] / timg.size[1]
        width = random.randint(tleast_shape, tmost_shape)
        height = int(width / w_h)

    while t_num > 0:
        # 要将转置的图片转换成RGBA格式，后之前没有像素的点才是透明的,如果本身就是RGBA就不用转
        timga = timg.convert('RGBA')
        # angle = random.randint(0, 360)
        # timga= timga.rotate(angle, expand=1)
        # timga=timg
        timga = distortImg(timga)

        x_min = random.randint(0, d_shapes - width)
        y_min = random.randint(0, d_shapes - height)
        x_max = x_min + width
        y_max = y_min + height
        # timga缩放到于切剪下来的 crop 一样大
        timga = timga.resize((width, height), Image.ANTIALIAS)
        # 从背景图片窃取一个box 来供与 timg 融合
        crop = dimg.crop((x_min, y_min, x_max, y_max))
        # 原timg 与 crop 融合

        timga = Image.composite(timga, crop, mask=timga)

        box = (x_min, y_min, x_max, y_max)

        bbox = (float(x_min / dimg.size[0]),
                float(y_min / dimg.size[1]),
                float(x_max / dimg.size[0]),
                float(y_max / dimg.size[1]))
        # 把融合后的那一块重新放回背景图片中
        dimg.paste(timga, box)
        # draw = ImageDraw.Draw(dimg)
        # draw.rectangle(box)
        bboxes.append(bbox)
        labels.append(int(VOC_LABELS[label][0]))  # label必须与VOC_labels 定义的一致
        labels_text.append(label.encode('ascii'))
        t_num -= 1
    dimg.save(dimg_name)

    channals = len(dimg.getbands())  # 获取通道数
    # 1 高 0 宽
    shape = [dimg.size[1],
             dimg.size[0],
             channals]
    dimg_data = tf.gfile.FastGFile(dimg_name, 'rb').read()

    return dimg_data, labels, labels_text, bboxes, shape


# 暂未启用
def gMoveImgPNGAlpha(bimg_name, timg_name, dimg_name, d_shapes=500, tleast_shape=50, tmost_shape=400, t_num=1):
    bboxes = []
    labels = []
    labels_text = []

    label = os.path.basename(timg_name)
    label, _ = os.path.splitext(label)  # 获取目标图片的名字，不包括后缀名
    label = getLabel(label)
    bimg = Image.open(bimg_name)
    dimg = bimg.resize((d_shapes, d_shapes))
    timg = Image.open(timg_name)

    # 这个地方长宽 1 是高 0 是宽
    if timg.size[1] > timg.size[0]:
        h_w = timg.size[1] / timg.size[0]
        height = random.randint(tleast_shape, tmost_shape)
        width = int(height / h_w)
        # width = random.randint(tleast_shape, tmost_shape)
        # height = min(int(width * h_w), tmost_shape)
    else:
        w_h = timg.size[0] / timg.size[1]
        width = random.randint(tleast_shape, tmost_shape)
        height = int(width / w_h)

    while t_num > 0:
        # 要将转置的图片转换成RGBA格式，旋转后之前没有像素的点才是透明的,如果本身就是RGBA就不用转
        timga = timg.convert('RGBA')
        # angle = random.randint(0, 360)
        # timga= timga.rotate(angle, expand=1)
        # timga=timg
        timga = distortImg(timga)

        x_min = random.randint(0, d_shapes - width)
        y_min = random.randint(0, d_shapes - height)
        x_max = x_min + width
        y_max = y_min + height
        # timga缩放到于切剪下来的 crop 一样大
        timga = timga.resize((width, height), Image.ANTIALIAS)
        # 从背景图片窃取一个box 来供与 timg 融合
        crop = dimg.crop((x_min, y_min, x_max, y_max))
        crop = crop.convert('RGBA')
        # 原timg 与 crop 融合

        # timga = Image.composite(timga, crop,mask=timga)
        timga = Image.blend(timga, crop, 0.5)

        box = (x_min, y_min, x_max, y_max)

        # 标准的数据集里面的Box的坐标范围都在0-1之间
        bbox = (float(x_min / dimg.size[0]),
                float(y_min / dimg.size[1]),
                float(x_max / dimg.size[0]),
                float(y_max / dimg.size[1]))
        # 把融合后的那一块重新放回背景图片中
        dimg.paste(timga, box)
        # draw = ImageDraw.Draw(dimg)
        # draw.rectangle(box)
        bboxes.append(bbox)
        labels.append(int(VOC_LABELS[label][0]))  # xml上面的label必须与VOC _labels 定义在common.py的一致
        labels_text.append(label.encode('ascii'))
        t_num -= 1
    dimg.save(dimg_name)

    channals = len(dimg.getbands())  # 获取通道数
    # 1 高 0 宽
    shape = [dimg.size[1],
             dimg.size[0],
             channals]
    dimg_data = tf.gfile.FastGFile(dimg_name, 'rb').read()

    return dimg_data, labels, labels_text, bboxes, shape


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned注意这里坐标的顺序
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


def gen_img_TFcord(bimgPath, timgPath, dimgPath, TFrecordPath, start=1, end=50, count=200):
    """
    :param bimgPath:背景图的文件夹路径
    :param timgPath:目标图的文件夹路径
    :param dimgPath：最终生成图片的放置文件夹路径
    :param TFrecordPath:最终生成的tfrecord文件的放置路径
    :param start:启用的背景图起始序号
    :param end:启用的背景图终止序号
    :param count:生成的图片（或tfrecord）的总数



    """
    timg_names = sorted(os.listdir(timgPath))
    for i in range(count):
        dimg_data = ''
        bboxes = []
        labels = []
        labels_text = []
        shape = []
        difficult = []
        truncated = []
        # 选取背景图并确定背景图路径
        j = i % (end - start) + start
        bimg_name = os.path.join(bimgPath, str(j) + '.jpg')
        # 随机选取目标图
        n = random.randint(0, len(timg_names) - 1)
        timg_name = os.path.join(timgPath, timg_names[n])
        # 确定最终生成图片的命名
        dimg_name = os.path.join(dimgPath, '{:06}.jpg'.format(i))
        # 确定最终生成的tfrecord的命名
        tfR_name = os.path.join(TFrecordPath, 'voc_2007_train_' + '{:03}.tfrecord'.format(i))

        cho = random.randint(0, 500)  # 主要用来确定目标的大小
        mr = bool(cho % 3 != 0)  # 确定是移动还是转动
        cn = 1  # 一张生成图片中包含的目标商品数量
        if cho <= 200:
            cn = 3 if cho < 50 else 2
            if mr:
                dimg_data, labels, labels_text, boxes, shape = gMoveImgPNG(bimg_name, timg_name, dimg_name,
                                                                           d_shapes=500,
                                                                           tleast_shape=60, tmost_shape=100, t_num=cn)
            else:
                dimg_data, labels, labels_text, boxes, shape = gRotImg(bimg_name, timg_name, dimg_name,
                                                                       d_shapes=500, tleast_shape=60, tmost_shape=100,
                                                                       t_num=cn)
        elif cho > 200 and cho <= 400:
            cn = 2 if cho > 360 else 1
            if mr:
                dimg_data, labels, labels_text, boxes, shape = gMoveImgPNG(bimg_name, timg_name, dimg_name,
                                                                           d_shapes=500,
                                                                           tleast_shape=100, tmost_shape=180, t_num=cn)
            else:
                dimg_data, labels, labels_text, boxes, shape = gRotImg(bimg_name, timg_name, dimg_name, d_shapes=500,
                                                                       tleast_shape=100, tmost_shape=180, t_num=cn)
        elif cho > 400 and cho <= 500:
            if mr:
                dimg_data, labels, labels_text, boxes, shape = gMoveImgPNG(bimg_name, timg_name, dimg_name,
                                                                           d_shapes=500,
                                                                           tleast_shape=180, tmost_shape=350,
                                                                           t_num=1)
            else:
                dimg_data, labels, labels_text, boxes, shape = gRotImg(bimg_name, timg_name, dimg_name, d_shapes=500,
                                                                       tleast_shape=180, tmost_shape=350,
                                                                       t_num=1)

        difficult.append(0)
        truncated.append(0)
        example = _convert_to_example(dimg_data, labels, labels_text, boxes, shape, difficult, truncated)

        with tf.python_io.TFRecordWriter(tfR_name) as tfrecord_writer:
            tfrecord_writer.write(example.SerializeToString())
        print("done!", i + 1)


def getLabel(basename):
    i = basename.index('-')
    return basename[:i]


def imgEnhColorR(img, st=0.6, ed=1.4):
    c = random.uniform(st, ed)
    return ImageEnhance.Color(img).enhance(c)


def imgEnhBrightnessR(img, st=0.6, ed=1.4):
    c = random.uniform(st, ed)
    return ImageEnhance.Brightness(img).enhance(c)


def imgEnhContrastR(img, st=0.7, ed=1.2):
    c = random.uniform(st, ed)
    return ImageEnhance.Contrast(img).enhance(c)


def imgEnhSharpnessR(img, st=0.7, ed=1.2):
    c = random.uniform(st, ed)
    return ImageEnhance.Sharpness(img).enhance(c)


def distortImg(img):
    i = random.randint(0, 4)
    if i == 0:
        img = imgEnhBrightnessR(img)
        img = imgEnhColorR(img)
    elif i == 1:
        img = imgEnhColorR(img)
        img = imgEnhBrightnessR(img)
    elif i == 2:
        img = imgEnhColorR(img)
        img = imgEnhContrastR(img)
    elif i == 3:
        img = imgEnhColorR(img)
        img = imgEnhSharpnessR(img)
    else:
        pass

    return img


# 用来去取生成的tfrecord,看看数据是否正常写入（暂只支持只有一个目标商品的图片）
def TFread():
    filepath = "./tfrecords_/voc_2007_test_1552812032.2088487.tfrecord"
    reader = tf.TFRecordReader()

    file_queue = tf.train.string_input_producer([filepath])
    _, se_example = reader.read(file_queue)
    features = tf.parse_single_example(se_example,
                                       features={
                                           'image/height': tf.FixedLenFeature((), tf.int64),
                                           'image/width': tf.FixedLenFeature((), tf.int64),
                                           'image/channels': tf.FixedLenFeature((), tf.int64),
                                           # 'image/shape':tf.FixedLenFeature((),tf.int64),
                                           'image/object/bbox/xmin': tf.FixedLenFeature((), tf.float32),
                                           'image/object/bbox/xmax': tf.FixedLenFeature((), tf.float32),
                                           'image/object/bbox/ymin': tf.FixedLenFeature((), tf.float32),
                                           'image/object/bbox/ymax': tf.FixedLenFeature((), tf.float32),
                                           'image/object/bbox/label': tf.FixedLenFeature((), tf.int64),
                                           'image/object/bbox/label_text': tf.FixedLenFeature((), tf.string,
                                                                                              default_value=''),
                                           # 'image/object/bbox/difficult': tf.FixedLenFeature((),tf.int64),
                                           # 'image/object/bbox/truncated': tf.FixedLenFeature((),tf.int64),
                                           'image/format': tf.FixedLenFeature((), tf.string),
                                           'image/encoded': tf.FixedLenFeature((), tf.string)
                                       })
    image = tf.decode_raw(features['image/encoded'], tf.uint8)

    box = (features['image/object/bbox/xmin'], features['image/object/bbox/xmax'],
           features['image/object/bbox/ymin'], features['image/object/bbox/ymax'])
    height = features['image/height']
    width = features['image/width']
    label = features['image/object/bbox/label']
    # label_txt=features['image/object/bbox/label_text']
    format = features['image/format']

    # image=tf.reshape
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('xmin xmax ymin ymax', sess.run(box))
        img = sess.run(image)
        print(img)
        with open('rem.txt', 'a') as f:
            f.write('\n')
            f.write(str(img))
        print('image', sess.run(image))
        print('width', sess.run(width))
        print('height', sess.run(height))
        print('label', sess.run(label))
        # print('label_txt', sess.run(label_txt))
        print('format', sess.run(format))


if __name__ == '__main__':
    gen_img_TFcord(r"./bimg",
                   r"./timg",
                   r"./autoGimg",
                   r"./autoGtfrecord", 1, 10, 20)
