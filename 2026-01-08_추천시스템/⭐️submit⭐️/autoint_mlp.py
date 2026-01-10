# 파일명: autoint_mlp.py (최종 수정: Sequential 제거 버전)
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, BatchNormalization, Activation, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal, GlorotUniform
import numpy as np

class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embed_dim, **kwargs):
        if 'name' not in kwargs: kwargs['name'] = 'fixed_embedding_layer'
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.total_dim = sum(field_dims)
        self.embed_dim = embed_dim
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        self.embedding = Embedding(input_dim=self.total_dim, output_dim=self.embed_dim, name='emb_matrix')

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.embedding.set_weights([GlorotUniform()(shape=self.embedding.weights[0].shape)])

    def call(self, x):
        x = tf.cast(x, dtype=tf.int32)
        x = x + tf.constant(self.offsets)
        return self.embedding(x)

class MultiHeadSelfAttention(Layer):
    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, scaling=False, seed=1024, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.scaling = scaling

    def build(self, input_shape):
        embedding_size = int(input_shape[-1])
        self.W_Query = self.add_weight(name='query_W', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key_W', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value_W', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed + 2))
        if self.use_res:
            self.W_Res = self.add_weight(name='res_W', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                         dtype=tf.float32, initializer=TruncatedNormal(seed=self.seed))
        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))
        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))
        inner_product = tf.matmul(querys, keys, transpose_b=True)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        normalized_att_scores = tf.nn.softmax(inner_product)
        result = tf.matmul(normalized_att_scores, values)
        result = tf.concat(tf.split(result, self.head_num, ), axis=-1)
        result = tf.squeeze(result, axis=0)
        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
        result = tf.nn.relu(result)
        return result

# [수정됨] Sequential 제거하고 리스트로 관리
class AutoIntMLP(Layer):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, dnn_hidden_units=(32, 32), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0.4, init_std=0.0001, **kwargs):
        if 'name' not in kwargs: kwargs['name'] = 'fixed_autoint_mlp_layer'
        super(AutoIntMLP, self).__init__(**kwargs)
        self.embedding = FeaturesEmbedding(field_dims, embedding_size, name='fixed_embedding')
        self.num_fields = len(field_dims)
        self.embedding_size = embedding_size

        self.final_layer = Dense(1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=init_std), name='fixed_final_output')
        
        # [핵심 변경] Sequential 대신 리스트(dnn_layers) 사용 -> 경로 꼬임 방지
        self.dnn_layers = []
        for i, units in enumerate(dnn_hidden_units):
            self.dnn_layers.append(Dense(units, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn), kernel_initializer=tf.random_normal_initializer(stddev=init_std), name=f'fixed_dnn_dense_{i}'))
            if dnn_use_bn:
                self.dnn_layers.append(BatchNormalization(name=f'fixed_dnn_bn_{i}'))
            self.dnn_layers.append(Activation(dnn_activation, name=f'fixed_dnn_act_{i}'))
            if dnn_dropout > 0:
                self.dnn_layers.append(Dropout(dnn_dropout, name=f'fixed_dnn_drop_{i}'))
        
        self.dnn_output_layer = Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=init_std), name='fixed_dnn_output_layer')
        
        self.int_layers = [MultiHeadSelfAttention(att_embedding_size=embedding_size, head_num=att_head_num, use_res=att_res, name=f'fixed_attention_{i}') for i in range(att_layer_num)]

    def call(self, inputs):
        embed_x = self.embedding(inputs)
        dnn_embed = tf.reshape(embed_x, shape=(-1, self.embedding_size * self.num_fields))
        
        att_input = embed_x
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = Flatten()(att_input)
        att_output = self.final_layer(att_output)
        
        # 리스트 순회 실행
        dnn_output = dnn_embed
        for layer in self.dnn_layers:
            dnn_output = layer(dnn_output)
        dnn_output = self.dnn_output_layer(dnn_output)
        
        y_pred = tf.keras.activations.sigmoid(att_output + dnn_output)
        return y_pred

class AutoIntMLPModel(Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2,
                 att_res=True, dnn_hidden_units=(32, 32), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False,
                 dnn_dropout=0.4, init_std=0.0001):
        super(AutoIntMLPModel, self).__init__(name='fixed_model_wrapper')
        self.autoInt_layer = AutoIntMLP(field_dims, embedding_size, att_layer_num, att_head_num, att_res, dnn_hidden_units, dnn_activation, l2_reg_dnn, l2_reg_embedding, dnn_use_bn, dnn_dropout, init_std)

    def call(self, inputs, training=False):
        return self.autoInt_layer(inputs, training=training)
