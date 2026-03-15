import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class CustomGRU(tf.keras.layers.Layer):
    def __init__(self, state_size=128, dropout_rate=0.3, initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.state_size = state_size
        self.dropout_rate = dropout_rate
        self.initializer = initializer

    def build(self, input_shapes):
        input_dim = input_shapes[-1]
        self.W_update = self.add_weight(shape=(input_dim, self.state_size), initializer=self.initializer)
        self.U_update = self.add_weight(shape=(self.state_size, self.state_size), initializer=self.initializer)
        self.B_update = self.add_weight(shape=(self.state_size,), initializer='zeros')
        self.W_reset = self.add_weight(shape=(input_dim, self.state_size), initializer=self.initializer)
        self.U_reset = self.add_weight(shape=(self.state_size, self.state_size), initializer=self.initializer)
        self.B_reset = self.add_weight(shape=(self.state_size,), initializer='zeros')
        self.W_h_ = self.add_weight(shape=(input_dim, self.state_size), initializer=self.initializer)
        self.U_h_ = self.add_weight(shape=(self.state_size, self.state_size), initializer=self.initializer)
        self.B_h_ = self.add_weight(shape=(self.state_size,), initializer='zeros')

    def call(self, inputs, states):
        h_t_1 = states[0]
        x_t = inputs
        z = tf.nn.sigmoid(tf.matmul(x_t, self.W_update) + tf.matmul(h_t_1, self.U_update) + self.B_update)
        r = tf.nn.sigmoid(tf.matmul(x_t, self.W_reset) + tf.matmul(h_t_1, self.U_reset) + self.B_reset)
        h_tilde = tf.matmul(x_t, self.W_h_) + tf.matmul(r * h_t_1, self.U_h_) + self.B_h_
        h_tilde = tf.nn.tanh(h_tilde)
        h_tilde = tf.nn.dropout(h_tilde, rate=self.dropout_rate)
        h_t = (1 - z) * h_t_1 + z * h_tilde
        return h_t, [h_t]

@register_keras_serializable()
class GRU1(CustomGRU): pass 

@register_keras_serializable()
class GRU2(GRU1):
    def call(self, inputs, states):
        h_t_1 = states[0]
        x_t = inputs
        z = tf.nn.sigmoid(tf.matmul(h_t_1, self.U_update))
        r = tf.nn.sigmoid(tf.matmul(x_t, self.W_reset))
        h_tilde = tf.matmul(x_t, self.W_h_) + tf.matmul(r * h_t_1, self.U_h_) + self.B_h_
        h_tilde = tf.nn.tanh(h_tilde)
        h_tilde = tf.nn.dropout(h_tilde, rate=self.dropout_rate)
        h_t = (1 - z) * h_t_1 + z * h_tilde
        return h_t, [h_t]

@register_keras_serializable()
class GRU3(tf.keras.layers.Layer):
    def __init__(self, state_size=128, dropout_rate=0.3, initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.state_size = state_size
        self.dropout_rate = dropout_rate
        self.initializer = initializer

    def build(self, input_shapes):
        input_dim = input_shapes[-1]
        s = self.state_size
        self.W_h_ = self.add_weight(shape=(input_dim, s), initializer=self.initializer)
        self.U_h_ = self.add_weight(shape=(s, s), initializer=self.initializer)
        self.B_h_ = self.add_weight(shape=(s,), initializer='zeros')
        self.B_update = self.add_weight(shape=(s,), initializer='zeros')
        self.B_reset = self.add_weight(shape=(s,), initializer='zeros')

    def call(self, inputs, states):
        h_t_1 = states[0]
        x_t = inputs
        z = tf.nn.sigmoid(self.B_update)
        r = tf.nn.sigmoid(self.B_reset)
        h_tilde = tf.matmul(x_t, self.W_h_) + tf.matmul(r * h_t_1, self.U_h_) + self.B_h_
        h_tilde = tf.nn.tanh(h_tilde)
        h_tilde = tf.nn.dropout(h_tilde, rate=self.dropout_rate)
        h_t = (1 - z) * h_t_1 + z * h_tilde
        return h_t, [h_t]

@register_keras_serializable()
class GRU4(CustomGRU):
    def __init__(self, units=128, dropout_rate=0.3, initializer='glorot_uniform', **kwargs):
        super().__init__(state_size=units, dropout_rate=dropout_rate, initializer=initializer, **kwargs)
        self.units = units

# BiGRU Wrappers
def create_bigru_class(GRU_Layer, name):
    @tf.keras.utils.register_keras_serializable(name=name)
    class BiGRU_Variant(tf.keras.layers.Layer):
        def __init__(self, state_size=128, dropout_rate=0.3, initializer='glorot_uniform', return_sequences=True, **kwargs):
            super().__init__(**kwargs)
            self.forward_layer = tf.keras.layers.RNN(GRU_Layer(state_size, dropout_rate, initializer), return_sequences=return_sequences)
            self.backward_layer = tf.keras.layers.RNN(GRU_Layer(state_size, dropout_rate, initializer), return_sequences=return_sequences, go_backwards=True)
            self.concat = tf.keras.layers.Concatenate(axis=-1)
            self.return_sequences = return_sequences

        def call(self, inputs):
            fwd = self.forward_layer(inputs)
            bwd = self.backward_layer(inputs)
            if self.return_sequences: bwd = tf.reverse(bwd, axis=[1])
            return self.concat([fwd, bwd])
    return BiGRU_Variant

BiGRU0 = create_bigru_class(CustomGRU, 'BiGRU0')
BiGRU1 = create_bigru_class(GRU1, 'BiGRU1')
BiGRU2 = create_bigru_class(GRU2, 'BiGRU2')
BiGRU3 = create_bigru_class(GRU3, 'BiGRU3')
BiGRU4 = create_bigru_class(GRU4, 'BiGRU4')

def get_gru_variants(units, dropout_rate, initializer='glorot_uniform'):
    return {
        'CustomGRU': CustomGRU(state_size=units, dropout_rate=dropout_rate, initializer=initializer),
        'GRU1': GRU1(state_size=units, dropout_rate=dropout_rate, initializer=initializer),
        'GRU2': GRU2(state_size=units, dropout_rate=dropout_rate, initializer=initializer),
        'GRU3': GRU3(state_size=units, dropout_rate=dropout_rate, initializer=initializer),
        'GRU4': GRU4(units=units, dropout_rate=dropout_rate, initializer=initializer),
    }

def get_bigru_variants(units, dropout_rate, initializer='glorot_uniform', return_sequences=False):
    return {
        'BiGRU0': BiGRU0(state_size=units, dropout_rate=dropout_rate, initializer=initializer, return_sequences=return_sequences),
        'BiGRU1': BiGRU1(state_size=units, dropout_rate=dropout_rate, initializer=initializer, return_sequences=return_sequences),
        'BiGRU2': BiGRU2(state_size=units, dropout_rate=dropout_rate, initializer=initializer, return_sequences=return_sequences),
        'BiGRU3': BiGRU3(state_size=units, dropout_rate=dropout_rate, initializer=initializer, return_sequences=return_sequences),
        'BiGRU4': BiGRU4(state_size=units, dropout_rate=dropout_rate, initializer=initializer, return_sequences=return_sequences),
    }
