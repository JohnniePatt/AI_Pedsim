import tensorflow as tf


def reparameterize(mu, logvar):
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps


def conv_block(x, filters, stride=2):
    x = tf.keras.layers.Conv2D(filters, 4, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    return x


def deconv_block(x, filters, dropout=0.0):
    x = tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_condition_encoder(image_size, base_filters):
    inp = tf.keras.Input(shape=(image_size, image_size, 3), name="cond_input")
    e1 = conv_block(inp, base_filters, stride=2)
    e2 = conv_block(e1, base_filters * 2, stride=2)
    e3 = conv_block(e2, base_filters * 4, stride=2)
    e4 = conv_block(e3, base_filters * 8, stride=2)
    bottleneck = conv_block(e4, base_filters * 8, stride=2)
    return tf.keras.Model(inp, [bottleneck, e1, e2, e3, e4], name="condition_encoder")


def build_posterior_encoder(image_size, base_filters, latent_dim):
    inp = tf.keras.Input(shape=(image_size, image_size, 6), name="posterior_input")
    x = conv_block(inp, base_filters, stride=2)
    x = conv_block(x, base_filters * 2, stride=2)
    x = conv_block(x, base_filters * 4, stride=2)
    x = conv_block(x, base_filters * 8, stride=2)
    x = conv_block(x, base_filters * 8, stride=2)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    mu = tf.keras.layers.Dense(latent_dim, name="z_mu")(x)
    logvar = tf.keras.layers.Dense(latent_dim, name="z_logvar")(x)
    return tf.keras.Model(inp, [mu, logvar], name="posterior_encoder")


def build_decoder(image_size, base_filters, latent_dim):
    bottleneck_h = image_size // 32
    bottleneck_w = image_size // 32

    in_bot = tf.keras.Input(shape=(bottleneck_h, bottleneck_w, base_filters * 8), name="in_bottleneck")
    in_s1 = tf.keras.Input(shape=(image_size // 2, image_size // 2, base_filters), name="skip1")
    in_s2 = tf.keras.Input(shape=(image_size // 4, image_size // 4, base_filters * 2), name="skip2")
    in_s3 = tf.keras.Input(shape=(image_size // 8, image_size // 8, base_filters * 4), name="skip3")
    in_s4 = tf.keras.Input(shape=(image_size // 16, image_size // 16, base_filters * 8), name="skip4")
    in_z = tf.keras.Input(shape=(latent_dim,), name="z_input")

    z_proj = tf.keras.layers.Dense(bottleneck_h * bottleneck_w * (base_filters * 4), use_bias=False)(in_z)
    z_proj = tf.keras.layers.Reshape((bottleneck_h, bottleneck_w, base_filters * 4))(z_proj)
    x = tf.keras.layers.Concatenate()([in_bot, z_proj])

    x = deconv_block(x, base_filters * 8, dropout=0.1)
    x = tf.keras.layers.Concatenate()([x, in_s4])

    x = deconv_block(x, base_filters * 4)
    x = tf.keras.layers.Concatenate()([x, in_s3])

    x = deconv_block(x, base_filters * 2)
    x = tf.keras.layers.Concatenate()([x, in_s2])

    x = deconv_block(x, base_filters)
    x = tf.keras.layers.Concatenate()([x, in_s1])

    x = deconv_block(x, max(base_filters // 2, 16))
    out = tf.keras.layers.Conv2D(3, 3, padding="same", activation="tanh", name="prediction")(x)

    return tf.keras.Model([in_bot, in_s1, in_s2, in_s3, in_s4, in_z], out, name="decoder")


class CVAE:
    def __init__(self, image_size, base_filters, latent_dim):
        self.cond_encoder = build_condition_encoder(image_size, base_filters)
        self.posterior_encoder = build_posterior_encoder(image_size, base_filters, latent_dim)
        self.decoder = build_decoder(image_size, base_filters, latent_dim)
        self.latent_dim = latent_dim

    @property
    def trainable_variables(self):
        return (
            self.cond_encoder.trainable_variables
            + self.posterior_encoder.trainable_variables
            + self.decoder.trainable_variables
        )

    def forward_train(self, image_a, image_b, training=True):
        bottleneck, s1, s2, s3, s4 = self.cond_encoder(image_a, training=training)
        posterior_input = tf.concat([image_a, image_b], axis=-1)
        mu, logvar = self.posterior_encoder(posterior_input, training=training)
        z = reparameterize(mu, logvar)
        pred = self.decoder([bottleneck, s1, s2, s3, s4, z], training=training)
        return pred, mu, logvar

    def forward_infer(self, image_a, z=None, training=False):
        bottleneck, s1, s2, s3, s4 = self.cond_encoder(image_a, training=training)
        if z is None:
            z = tf.zeros((tf.shape(image_a)[0], self.latent_dim), dtype=tf.float32)
        pred = self.decoder([bottleneck, s1, s2, s3, s4, z], training=training)
        return pred


class CVAEInference:
    def __init__(self, image_size, base_filters, latent_dim):
        self.cond_encoder = build_condition_encoder(image_size, base_filters)
        self.decoder = build_decoder(image_size, base_filters, latent_dim)
        self.latent_dim = latent_dim

    def predict(self, image_a, z=None):
        bottleneck, s1, s2, s3, s4 = self.cond_encoder(image_a, training=False)
        if z is None:
            z = tf.zeros((tf.shape(image_a)[0], self.latent_dim), dtype=tf.float32)
        pred = self.decoder([bottleneck, s1, s2, s3, s4, z], training=False)
        return pred
