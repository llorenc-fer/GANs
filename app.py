import streamlit as st
import streamlit.components.v1 as components
import base64
st.set_page_config(page_title='Generative Adversarial Networks', layout='centered', page_icon='🌇')

#-----empieza la app-----------------------------------------------------------------------------------
st.title("Generative Adversarial Networks")
st.text("by Llorenç Fernández Alsina")
st.image('https://data-science-blog.com/wp-content/uploads/2018/07/deep-learning-header-1030x352.png', caption='Image from data-science-blog.com')

st.subheader("Generative Adversarial Networks: brief introduction")
st.markdown('In 2014, Ian Goodfellow and his colleagues designed the first generative adversarial network(GAN), a class of Machine Learning frameworks consisting of two neural networks competing against each other, with the gain of one agent resulting in the loss of the other.')
st.image('GANformula.png', caption='GAN Scientific Formula, from 2014 original paper')
st.markdown('The process is as it follows: we train two models simultaneously, a Generative model (G) that learns the distribution of the data, and a Discriminative model (D) that predicts whether a sample came from the training data or from the Generator. The goal of training G is to maximize the likelihood of D making a mistake.')

st.image("https://miro.medium.com/max/1400/1*xOgw_4Wv2KHvGzm_x0zeIQ.webp", caption="Image from towardsdatascience.com")


st.markdown('The Generator and Discriminator are trained in turns: to train the Generator, we use a random distribution noise vector as input. To train the Discriminator, we use both labelled images from the Generator and actual images as input. ')



st.markdown("This process will result in an increase in Generator's ability to create data more and more similar to the original one, and an increase in Discriminator's ability to discern more realistic data (but synthetically generated) from original data.")
st.markdown("In the following lines, we'll explore a generative adversarial model trained with the MNIST dataset, threshing the most relevant parts of the code, as well as the architecture of our convolutional neural networks")

st.subheader("Loading and preprocessing dataset")
#-------------------LOADING AND PREPROCESSING----------------------------------------------------------------------------------------------
st.markdown('Loading dataset')
with st.expander("*Loading dataset*"):
        st.write("""
                The tf.keras.datasets module offers a selection of datasets
                """)
        st.write("""
                This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. More info can be found at the MNIST homepage.
                """)
code = """
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
"""
st.code(code, language='python')


st.markdown('Preprocessing dataset')
with st.expander("*Normalizing data*"):
        st.write("""
                Normalization aids in the training of neural networks by ensuring that the various features have a comparable scale, which stabilizes the gradient descent process and allows for larger learning rates
                """)
code = """
    train_images = (train_images - 127.5) / 127.5
"""
st.code(code, language='python')


st.markdown('Defining buffer and batch size')
with st.expander("*Defining buffer size*"):
        st.write("""
                Buffer size should be equal to the total number of training images
                """)
with st.expander("*Defining batch size*"):
        st.write("""
                Batch size is the number of samples processed per epoch: a higher number will increase our model's accuracy, but it will also require more computational resources
                """)
code = """

BUFFER_SIZE = 60000 
BATCH_SIZE = 256 
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
"""
st.code(code, language='python')








#----------GENERATOR CNN--------------------------------------------------------

st.subheader('First Convolutional Neural Network')
st.markdown('Architecture of the Generator model:')
code = """
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, 
                                        input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    """
st.code(code, language='python')
with st.expander("*Dense Layer*"):
        st.write("""
                The Dense Layer takes the seed (random noise) as an input.
                """)
with st.expander("*Activation function Leaky ReLU*"):
        st.write("""
                Leaky ReLU is based on ReLU (Rectified Linear Unit), but it has a small slope for negative values instead of a flat slope:
                """)
        st.image(r"https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-25_at_3.09.45_PM.png", caption='Formula by paperswithcode.com')
        st.image("relu.png", caption='Image by Massachussets Institute of Technology')




code = """
model.add(layers.Reshape((7, 7, 256)))
assert model.output_shape == (None, 7, 7, 256) 

model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), 
                                                padding='same', 
                                                use_bias=False))
assert model.output_shape == (None, 7, 7, 128)
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())
"""
st.code(code, language='python')

with st.expander("*Conv2DTranspose layers:*"):
        st.write("""
                The Generator model uses upsampling to produce an image from the seed.
                """)

code = """
model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), 
                                                padding='same', 
                                                use_bias=False))
assert model.output_shape == (None, 14, 14, 64)
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU())

model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), 
                                                padding='same', 
                                                use_bias=False, 
                                                activation='tanh'))
assert model.output_shape == (None, 28, 28, 1)

return model
"""
st.code(code, language='python')

with st.expander("*Output Shape*"):
        st.write("""
                The output will have a shape of 28x28 in grayscale (1).
                """)
        

st.subheader("Creating an image with the untrained Generator model")
code = """
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
"""
st.code(code, language='python')
with st.expander("*Output image*"):
        st.image('randomnoise.png', caption='Random noise generated image')

st.subheader('Second Convolutional Neural Network')
st.markdown('Architecture of the Discriminator model:')

code = """
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2),
                                        padding='same', 
                                        input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
"""
st.code(code, language='python')
with st.expander("*Strides*"):
        st.markdown('How many pixels the filter of the CNN moves.')
        st.image('https://images.deepai.org/django-summernote/2019-06-03/56e53bc1-bac3-48f4-a08c-dce77a57464b.png', caption='Image from deepai.org')
with st.expander("*Padding*"):
        st.markdown('Amount of pixels added to an image when it is being processed.')
        



code = """
model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), 
                                    padding='same'))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(0.3))
"""
st.code(code, language='python')
with st.expander("*Dropout*"):
        st.markdown('A way to prevent overfitting by randomly disconnecting neurons during training')
        st.image('https://cdn-images-1.medium.com/max/1000/1*uxpH46OpTIj63j1MKQ-T2Q.png', caption='Srivastava, Nitish, et al. ”Dropout: a simple way to prevent neural networks from overfitting”, JMLR 2014')

code = """
model.add(layers.Flatten())
model.add(layers.Dense(1))

return model
"""
st.code(code, language='python')

st.subheader("Classifying the image with the untrained Discriminator model")
st.markdown('The output will result in positive values for real images and negative values for synthetic images')
code = """
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)
"""
st.code(code, language='python')
with st.expander("*Output*"):
    st.write('tf.Tensor([[-0.00435742]], shape=(1, 1), dtype=float32)')


st.subheader('Defining Loss Function for both models')
st.markdown('Loss function for the models will be Binary Cross Entropy, as it fits the binary label output.')
code = """
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
"""
st.code(code, language='python')



st.subheader('Loss Functions')
st.markdown("This will help us quantify how well the Discriminator model can tell real images from synthetic ones. It will compare the Discriminator's predictions on real images to an array of 1s, and synthetic images to an array of 0s.")
code = """
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
"""
st.code(code, language='python')

st.markdown('Loss function for the Generator model will be Binary Cross Entropy, as it fits the binary label output.')
st.markdown("This will help us quantify how well the Generator model can trick the Discriminator.")
code = """
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
"""
st.code(code, language='python')

st.subheader('Setting the Optimizers')
st.markdown('For this case we will be using an Adam optimizer for both models through separate variables, since we are training two different models. An Adam optimizer is an extension to stochastic gradient descent.')
code = """
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
"""
st.code(code, language='python')

st.subheader('Creating checkpoints')
st.markdown('This is used to save and restore models, which can prove useful when training models for a long time.')
code = """
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)
"""
st.code(code, language='python')


st.subheader('Setting training loop parameters')
st.markdown('The loop begins with the Generator model receiving a random seed as input, which is used to generate an image. The Discriminator model will have to discern real images from synthetic ones.')
code = """
EPOCHS = 50 
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])
"""
st.code(code, language='python')
with st.expander("*Epochs*"):
    st.markdown('Number of times that it will iterate through the loop')
with st.expander("*Noise Dimensions*"):
    st.markdown('Noise data points')


st.subheader('Defining the training loop')
code="""
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
"""
st.code(code, language='python')
st.markdown("The loss for the models is calculated, and they both are updated with the gradients.")

code="""
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        display.clear_output(wait=True)
        generate_and_save_images(generator,epoch + 1,seed)
"""
st.code(code,language='python')
with st.expander("*Display.clear_output*"):
    st.markdown('Generates images of each epoch')
code="""
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
"""
st.code(code, language='python')
st.markdown('Create a checkpoint of the model every 15 epochs')

code="""
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,seed)
"""
st.code(code, language='python')
st.markdown('Generate after the final epoch')

st.subheader('Generate and save images')
code="""
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
"""
st.code(code,language='python')

st.subheader("Training the model")
st.markdown("It's important that the Generator model and the Discriminator model train at a similar rate.")
code="""
train(train_dataset, EPOCHS)
"""
st.markdown("The images generated at the beginning of the training will not differ a lot from random noise. As epochs go, the generated images will each time ressemble more the MNIST digits from the original dataset.")
st.subheader("Creating a GIF")
st.markdown('Once the training is finished, we can create a GIF with the data generated in all epochs')
code="""
anim_file = 'dcGAN.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
"""
st.code(code,language='python')
st.markdown('Using the library *imageio*, we will be able to create a GIF with all the images saved during the training epochs.')

file_ = open("dcgan.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)

