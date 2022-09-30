import SimpleITK as Sit
import numpy as np

# A path to a T1-weighted brain .nii image:
import tf as tf
from keras.datasets.boston_housing import load_data

t1_fn = './brain_t1_0001.nii'

# Read the .nii image containing the volume with SimpleITK:
Sit_t1 = Sit.ReadImage(t1_fn)

# and access the numpy array:
t1 = Sit.GetArrayFromImage(Sit_t1)

# Load all data into memory
data = load_data(all_filenames, tf.estimator.ModeKeys.TRAIN, reader_params)

# Create placeholder variables and define their shapes (here,
# we input a volume image of size [128, 224, 244] and a single
# channel (i.e. greyscale):
x = tf.placeholder(reader_example_dtypes['features']['x'],
                   [None, 128, 224, 224, 1])
y = tf.placeholder(reader_example_dtypes['labels']['y'],
                   [None, 1])

# Create a tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.repeat(None)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

# Create an iterator
iterator = dataset.make_initializable_iterator()
nx = iterator.get_next()

with tf.train.MonitoredTrainingSession() as sess_dict:
    sess_dict.run(iterator.initializer,
                  feed_dict={x: data['features'], y: data['labels']})

    for i in range(iterations):
        # Get next features/labels pair
        dict_batch_feat, dict_batch_lbl = sess_dict.run(nx)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# path to save the TFRecords file
train_filename = 'train.tfrecords'

# open the file
writer = tf.python_io.TFRecordWriter(train_filename)

# iterate through all .nii files:
for meta_data in all_filenames:
    # Load the image and label
    img, label = load_img(meta_data, reader_params)

    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _float_feature(img.ravel())}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()


def decode(serialized_example):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.parse_single_example(
        serialized_example,
        features={'train/image': tf.FixedLenFeature([128, 224, 224, 1], tf.float32),
                  'train/label': tf.FixedLenFeature([], tf.int64)})

    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return features['train/image'], features['train/label']


dataset = tf.data.TFRecordDataset(train_filename).map(decode)
dataset = dataset.repeat(None)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

iterator = dataset.make_initializable_iterator()
features, labels = iterator.get_next()
nx = iterator.get_next()

with tf.train.MonitoredTrainingSession() as sess_rec:
    sess_rec.run(iterator.initializer)

    for i in range(iterations):
        try:
            # Get next features-labels pair
            rec_batch_feat, rec_batch_lbl = sess_rec.run([features, labels])

        except tf.errors.OutOfRangeError:
            pass


def read_fn(file_references, mode, params=None):
    # We define a `read_fn` and iterate through the `file_references`, which
    # can contain information about the data to be read (e.g. a file path):
    for meta_data in file_references:

        # Here, we parse the `subject_id` to construct a file path to read
        # an image from.
        subject_id = meta_data[0]
        data_path = '../../data/IXI_HH/1mm'
        t1_fn = os.path.join(data_path, '{}/T1_1mm.nii.gz'.format(subject_id))

        # Read the .nii image containing a brain volume with SimpleITK and get
        # the numpy array:
        sitk_t1 = Sit.ReadImage(t1_fn)
        t1 = Sit.GetArrayFromImage(sitk_t1)

        # Normalise the image to zero mean/unit std dev:
        t1 = whitening(t1)

        # Create a 4D Tensor with a dummy dimension for channels
        t1 = t1[..., np.newaxis]

        # If in PREDICT mode, yield the image (because there will be no label
        # present). Additionally, yield the sitk.Image pointer (including all
        # the header information) and some metadata (e.g. the subject id),
        # to facilitate post-processing (e.g. reslicing) and saving.
        # This can be useful when you want to use the same read function as
        # python generator for deployment.
        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': t1}}

        # Labels: Here, we parse the class *sex* from the file_references
        # \in [1,2] and shift them to \in [0,1] for training:
        sex = np.int32(meta_data[1]) - 1
        y = sex

        # If training should be done on image patches for improved mixing,
        # memory limitations or class balancing, call a patch extractor
        if params['extract_examples']:
            images = extract_random_example_array(
                t1,
                example_size=params['example_size'],
                n_examples=params['n_examples'])

            # Loop the extracted image patches and yield
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': y.astype(np.int32)}}

        # If desired (i.e. for evaluation, etc.), return the full images
        else:
            yield {'features': {'x': images},
                   'labels': {'y': y.astype(np.int32)}}

    return


# Generator function
def f():
    fn = read_fn(file_references=all_filenames,
                 mode=tf.estimator.ModeKeys.TRAIN,
                 params=reader_params)

    ex = next(fn)
    # Yield the next image
    yield ex


# Timed example with generator io
dataset = tf.data.Dataset.from_generator(
    f, reader_example_dtypes, reader_example_shapes)
dataset = dataset.repeat(None)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

iterator = dataset.make_initializable_iterator()
next_dict = iterator.get_next()

with tf.train.MonitoredTrainingSession() as sess_gen:
    # Initialize generator
    sess_gen.run(iterator.initializer)

    with Timer('Generator'):
        for i in range(iterations):
            # Fetch the next batch of images
            gen_batch_feat, gen_batch_lbl = sess_gen.run([next_dict['features'], next_dict['labels']])


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = Sit.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(Sit.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(Sit.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(Sit.sitkBSpline)

    return resample.Execute(itk_image)


# Assume to have some sitk image (itk_image) and label (itk_label)
resampled_sitk_img = resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False)
resampled_sitk_lbl = resample_img(itk_label, out_spacing=[2.0, 2.0, 2.0], is_label=True)
