import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import spektral
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from spektral.data.graph import Graph
from spektral.data.dataset import Dataset
from spektral.data.loaders import SingleLoader
import scipy.sparse as sparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from tensorflow.python.keras import callbacks
import warnings
import tensorflow as tf
from keras import backend as K
from keras import constraints, initializers, regularizers
from keras.layers import Dropout
from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.layers.ops import modes
import networkx as nx
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------------------------------

data = pd.read_pickle('data_IRAN_100_updated_1.pkl')
data = data[data.index <= '2021-06-21']


column = data.columns
list_of_stocks = pd.DataFrame()
for i in column:
    if i[0] == 'Close':
        dic = {'stocks': i[1]}
        list_of_stocks = list_of_stocks.append(dic, ignore_index=True)

# --------------------------------------------------------------------------------------------

class DGAT(Conv):

    def __init__(
        self,
        channels,
        attn_heads=1,
        concat_heads=True,
        dropout_rate=0.5,
        return_attn_coef=False,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        if concat_heads:
            self.output_dim = self.channels * self.attn_heads
        else:
            self.output_dim = self.channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim, self.attn_heads, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.attn_kernel_self = self.add_weight(
            name="attn_kernel_self",
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.output_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
            )

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs):
        x, a = inputs

        mode = ops.autodetect_mode(x, a)
        if mode == modes.SINGLE and K.is_sparse(a):
            output, attn_coef = self._call_single(x, a)
        else:
            if K.is_sparse(a):
                a = tf.sparse.to_dense(a)
            output, attn_coef = self._call_dense(x, a)

        if self.concat_heads:
            shape = output.shape[:-2] + [self.attn_heads * self.channels]
            shape = [d if d is not None else -1 for d in shape]
            output = tf.reshape(output, shape)
        else:
            output = tf.reduce_mean(output, axis=-2)
        if self.use_bias:
            output += self.bias

        output = self.activation(output)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def _call_single(self, x, a):
        # Reshape kernels for efficient message-passing
        kernel = tf.reshape(self.kernel, (-1, self.attn_heads * self.channels))
        attn_kernel_self = ops.transpose(self.attn_kernel_self, (2, 1, 0))

        # Prepare message-passing
        indices = a.indices
        N = tf.shape(x, out_type=indices.dtype)[-2]
        indices = ops.add_self_loops_indices(indices, N)
        targets, sources = indices[:, 1], indices[:, 0]

        # Update node features
        x = K.dot(x, kernel)
        x = tf.reshape(x, (-1, self.attn_heads, self.channels))

        # Compute attention
        attn_for_self = tf.reduce_sum(x * attn_kernel_self, -1)
        attn_for_self = tf.gather(attn_for_self, targets)

        attn_coef = attn_for_self
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = ops.unsorted_segment_softmax(attn_coef, targets, N)
        attn_coef = self.dropout(attn_coef)
        attn_coef = attn_coef[..., None]

        # Update representation
        output = attn_coef * tf.gather(x, targets)
        output = tf.math.unsorted_segment_sum(output, targets, N)

        return output, attn_coef

    def _call_dense(self, x, a):
        shape = tf.shape(a)[:-1]
        a = tf.linalg.set_diag(a, tf.zeros(shape, a.dtype))
        a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
        x = tf.einsum("...NI , IHO -> ...NHO", x, self.kernel)
        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", x, self.attn_kernel_self)
        attn_for_neighs = tf.einsum(
            "...NHI , IHO -> ...NHO", x, self.attn_kernel_neighs
        )
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)

        mask = -10e9 * (1.0 - a)
        attn_coef += mask[..., None, :]
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)

        output = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, x)

        return output, attn_coef

    @property
    def config(self):
        return {
            "channels": self.channels,
            "attn_heads": self.attn_heads,
            "concat_heads": self.concat_heads,
            "dropout_rate": self.dropout_rate,
            "return_attn_coef": self.return_attn_coef,
            "attn_kernel_initializer": initializers.serialize(
                self.attn_kernel_initializer
            ),
            "attn_kernel_regularizer": regularizers.serialize(
                self.attn_kernel_regularizer
            ),
            "attn_kernel_constraint": constraints.serialize(
                self.attn_kernel_constraint
            ),
        }
class GNN(tf.keras.Model):

    def __init__(
            self,
            n_labels,
            channels=8,
            activation="swish",
            output_activation="softmax",
            use_bias=True,
            dropout_rate=0.3,
            l2_reg=1e-2,
            n_input_channels=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.channels = channels
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.n_input_channels = n_input_channels
        reg = tf.keras.regularizers.l2(l2_reg)

        self._d0 = tf.keras.layers.Dropout(dropout_rate)
        self._d1 = tf.keras.layers.Dropout(dropout_rate)

        self._dgta0 = DGAT(channels, attn_heads=3, concat_heads=False, dropout_rate=0.0, return_attn_coef=False,
                                activation=activation, use_bias=use_bias, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros', attn_kernel_initializer='glorot_uniform',
                                kernel_regularizer=reg, bias_regularizer=reg, attn_kernel_regularizer=None,
                                activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                                attn_kernel_constraint=None)

        self._dgta1 = DGAT(n_labels, attn_heads=1, concat_heads=False, dropout_rate=0.0, return_attn_coef=False,
                                activation=output_activation, use_bias=use_bias, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros', attn_kernel_initializer='glorot_uniform',
                                kernel_regularizer=reg, bias_regularizer=reg, attn_kernel_regularizer=None,
                                activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                                attn_kernel_constraint=None)

        if tf.version.VERSION < "2.2":
            if n_input_channels is None:
                raise ValueError("n_input_channels required for tf < 2.2")
            x = tf.keras.Input((n_input_channels,), dtype=tf.float32)
            a = tf.keras.Input((None,), dtype=tf.float32, sparse=True)
            self._set_inputs((x, a))

    def get_config(self):
        return dict(
            n_labels=self.n_labels,
            channels=self.channels,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            n_input_channels=self.n_input_channels,
        )

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError("Inputs should be (x, a), got {}".format(inputs))
        x, a = inputs
        if self.n_input_channels is None:
            self.n_input_channels = x.shape[-1]
        else:
            assert self.n_input_channels == x.shape[-1]
        x = self._d0(x)
        x = self._dgta0([x, a])
        x = self._d1(x)
        return self._dgta1([x, a])
class MyDataset(Dataset):
    def read(self):
        return list_of_Graphs
class MyEval(callbacks.Callback):
    acc_set = np.array([])

    def on_epoch_end(self, epoch, logs=None):
        predictian_GCN = self.model.predict(loader_test.load(), steps=loader_test.steps_per_epoch, verbose=0)
        for i in range(len(predictian_GCN)):
            if predictian_GCN[i, 0] >= predictian_GCN[i, 1]:
                predictian_GCN[i, 0] = 1
            else:
                predictian_GCN[i, 0] = -1
        if epoch >= 0:
            acc = accuracy_score(Y_TRUE, predictian_GCN[:, 0], sample_weight=test_samples_weight)
            self.acc_set = np.append(self.acc_set, acc)

# --------------------------------------------------------------------------------------------

def get_random_seed(i):
    # you can set it whatever you want
    # I emphasize that you should run this code 10 times and average the result
    seed = 1
    return seed
def score_function(y_pre, y_true, c):
    sorat, makhrag = 0, 0
    for i in range(len(y_pre)):
        makhrag += ((1 - c) ** (i))
        if y_pre[-i] == y_true.iloc[-i]:
            sorat += ((1 - c) ** (i))
    return sorat / makhrag
def LDA(X_train, Y_train, X_validation, Y_validation, k):
    result_k = pd.DataFrame()
    for element in k:
        F_S = SelectKBest(k=element).fit(X_train, Y_train)
        x_train = F_S.transform(X_train)
        y_train = Y_train
        x_validation = F_S.transform(X_validation)
        y_validation = Y_validation
        clf = LinearDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        s = score_function(clf.predict(x_validation), y_validation, c=0.1)
        dic = {'k': element, 'score': s}
        result_k = result_k.append(dic, ignore_index=True)
    score = result_k['score'].max()
    best_k = result_k['k'].iloc[result_k['score'].argmax()]
    model = LinearDiscriminantAnalysis()
    return score, best_k, model
def NB(X_train, Y_train, X_validation, Y_validation, k):
    result_k = pd.DataFrame()
    for element in k:
        F_S = SelectKBest(k=element).fit(X_train, Y_train)
        x_train = F_S.transform(X_train)
        y_train = Y_train
        x_validation = F_S.transform(X_validation)
        y_validation = Y_validation
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        s = score_function(clf.predict(x_validation), y_validation, c=0.1)
        dic = {'k': element, 'score': s}
        result_k = result_k.append(dic, ignore_index=True)
    score = result_k['score'].max()
    best_k = result_k['k'].iloc[result_k['score'].argmax()]
    model = GaussianNB()
    return score, best_k, model
def FDA(X_train, Y_train, X_validation, Y_validation, k):
    result_k = pd.DataFrame()
    for element in k:
        F_S = SelectKBest(k=element).fit(X_train, Y_train)
        x_train = F_S.transform(X_train)
        y_train = Y_train
        x_validation = F_S.transform(X_validation)
        y_validation = Y_validation
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        s = score_function(clf.predict(x_validation), y_validation, c=0.1)
        dic = {'k': element, 'score': s}
        result_k = result_k.append(dic, ignore_index=True)
    score = result_k['score'].max()
    best_k = result_k['k'].iloc[result_k['score'].argmax()]
    model = QuadraticDiscriminantAnalysis()
    return score, best_k, model
def DT(X_train, Y_train, X_validation, Y_validation, k):
    result_k = pd.DataFrame()
    for element in k:
        F_S = SelectKBest(k=element).fit(X_train, Y_train)
        x_train = F_S.transform(X_train)
        y_train = Y_train
        x_validation = F_S.transform(X_validation)
        y_validation = Y_validation
        clf = DecisionTreeClassifier()
        clf.fit(x_train, y_train)
        s = score_function(clf.predict(x_validation), y_validation, c=0.1)
        dic = {'k': element, 'score': s}
        result_k = result_k.append(dic, ignore_index=True)
    score = result_k['score'].max()
    best_k = result_k['k'].iloc[result_k['score'].argmax()]
    model = DecisionTreeClassifier()
    return score, best_k, model
def AdaBoost(X_train, Y_train, X_validation, Y_validation, k):
    result_k = pd.DataFrame()
    for element in k:
        F_S = SelectKBest(k=element).fit(X_train, Y_train)
        x_train = F_S.transform(X_train)
        y_train = Y_train
        x_validation = F_S.transform(X_validation)
        y_validation = Y_validation
        clf = AdaBoostClassifier()
        clf.fit(x_train, y_train)
        s = score_function(clf.predict(x_validation), y_validation, c=0.1)
        dic = {'k': element, 'score': s}
        result_k = result_k.append(dic, ignore_index=True)
    score = result_k['score'].max()
    best_k = result_k['k'].iloc[result_k['score'].argmax()]
    model = AdaBoostClassifier()
    return score, best_k, model
def get_prediction(X_train_plus, Y_train_plus, X_test, best_mooo, model):
    F_S = SelectKBest(k=best_mooo).fit(X_train_plus, Y_train_plus)
    x_train = F_S.transform(X_train_plus)
    y_train = Y_train_plus
    x_test = F_S.transform(X_test)
    model = model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred
def perf_measure(y_true, y_pred, sample_weight):
    Confusion_matrix = pd.DataFrame(0, index=np.arange(len(y_true)), columns=['TP', 'FP', 'TN', 'FN'])
    for i in range(len(y_pred)):
        if sample_weight[i] != 0:
            if y_pred[i] == 1 and y_true[i] == 1:
                Confusion_matrix['TP'].iloc[i] += 1
            if y_pred[i] == -1 and y_true[i] == -1:
                Confusion_matrix['TN'].iloc[i] += 1
            if y_pred[i] == 1 and y_true[i] == -1:
                Confusion_matrix['FP'].iloc[i] += 1
            if y_pred[i] == -1 and y_true[i] == 1:
                Confusion_matrix['FN'].iloc[i] += 1
    return Confusion_matrix

# --------------------------------------------------------------------------------------------

threshold = 0.85
epoch = 300

Confusion_matrix_total = pd.DataFrame(0, columns=['TP', 'FP', 'TN', 'FN'], index=list_of_stocks['stocks'])
cnfig_result = np.array([])
mat_acc_epoch = np.array([])

# 65 days from the last day in the data are test data
for day in range(1, 66):
    # To be repeatable code, the random seed is initialized
    seed = get_random_seed(day)
    seed_tf = seed
    np.random.seed(seed)
    print('day = ', day, 'Date = ',data.index[-day])

    X_GCN = pd.DataFrame()
    Y_GCN = pd.DataFrame()
    train_samples_weight = np.array([])
    test_samples_weight = np.array([])
    daily_max_score_of_stock = np.array([])
    Y_TRUE = np.array([])
    Y_prediction_Classic_models = np.array([])
    number_of_correct_predictions_of_classic_models = 0

    list_stocks_roz = []
    list_stocks_without_data = []
    for col in list_of_stocks['stocks']:

        result_classical_models_on_validation_each_stock = pd.DataFrame(columns=['score', 'best_k', 'model'])

        Stock_data = pd.DataFrame()
        for col2 in data.columns:
            if col2[1] == col:
                Stock_data[col2[0]] = data[col2]


        X = Stock_data.drop(['Adj Close'], axis=1, inplace=False)
        X = X.drop(['Y_label'], axis=1, inplace=False)
        Y = Stock_data['Y_label']


        test_x = X[len(Stock_data) - day: len(Stock_data) - day + 1]
        test_y = Y[len(Stock_data) - day: len(Stock_data) - day + 1]

        Stock_data = Stock_data.dropna()
        X = Stock_data.drop(['Adj Close'], axis=1, inplace=False)
        X = X.drop(['Y_label'], axis=1, inplace=False)
        Y = Stock_data['Y_label']


        train_x = X[: len(Stock_data) - (10 + day)]
        train_y = Y[: len(Stock_data) - (10 + day)]

        validation_x = X[len(Stock_data) - (10 + day): len(Stock_data) - day]
        validation_y = Y[len(Stock_data) - (10 + day): len(Stock_data) - day]

        train_x_plus = X[: len(Stock_data) - (day)]
        train_y_plus = Y[: len(Stock_data) - (day)]


        # ----------------------------------------------------------------------------------------------------------
        if np.isnan(test_y.iloc[-1].item()) != True and len(train_x) >= 20 and len(validation_x) >= 10:
            # Classic_models
            try:
                score, best_k, model = LDA(train_x, train_y, validation_x, validation_y, [4])
                dic = {'score': score, 'best_k': best_k, 'model': model}
                result_classical_models_on_validation_each_stock = \
                    result_classical_models_on_validation_each_stock.append(dic, ignore_index=True)
            except Exception as e:
                pass
            try:
                score, best_k, model = FDA(train_x, train_y, validation_x, validation_y, [4])
                dic = {'score': score, 'best_k': best_k, 'model': model}
                result_classical_models_on_validation_each_stock = \
                    result_classical_models_on_validation_each_stock.append(dic, ignore_index=True)
            except Exception as e:
                pass
            try:
                score, best_k, model = NB(train_x, train_y, validation_x, validation_y, [4])
                dic = {'score': score, 'best_k': best_k, 'model': model}
                result_classical_models_on_validation_each_stock = \
                    result_classical_models_on_validation_each_stock.append(dic, ignore_index=True)
            except Exception as e:
                pass
            try:
                score, best_k, model = AdaBoost(train_x, train_y, validation_x, validation_y, [4])
                dic = {'score': score, 'best_k': best_k, 'model': model}
                result_classical_models_on_validation_each_stock = \
                    result_classical_models_on_validation_each_stock.append(dic, ignore_index=True)
            except Exception as e:
                pass
            try:
                score, best_k, model = DT(train_x, train_y, validation_x, validation_y, [4])
                dic = {'score': score, 'best_k': best_k, 'model': model}
                result_classical_models_on_validation_each_stock = \
                    result_classical_models_on_validation_each_stock.append(dic, ignore_index=True)
            except Exception as e:
                pass
            # ----------------------------------------------------------------------------------------------------------
            # print(result_classical_models_on_validation_each_stock)
            best_model = result_classical_models_on_validation_each_stock.loc[result_classical_models_on_validation_each_stock['score'].idxmax()]
            prediction = get_prediction(train_x_plus, train_y_plus, test_x, int(best_model['best_k']), best_model['model'])
        # ----------------------------------------------------------------------------------------------------------
        if np.isnan(test_y.iloc[-1].item()) != True  :
            list_stocks_roz.append(col)
            dic1 = {
                'S_Close': test_x['S_Close'].iloc[-1],
                'S_Volume': test_x['S_Volume'].iloc[-1],
                'S_cloes_az_open': test_x['S_cloes_az_open'].iloc[-1],
                'S_RSI': test_x[('S_RSI')].iloc[-1],
                'S_BB': test_x['S_BB'].iloc[-1],
                'S_MACD': test_x['S_MACD'].iloc[-1],
                'S_SAR': test_x['S_SAR'].iloc[-1],
                'S_ADX_DMI': test_x['S_ADX_DMI'].iloc[-1],
                'S_Stochastic': test_x['S_Stochastic'].iloc[-1],
                'S_MFI': test_x['S_MFI'].iloc[-1],
                'S_CCI': test_x['S_CCI'].iloc[-1],
            }
            X_GCN = X_GCN.append(dic1, ignore_index=True)

            if np.isnan(test_y.iloc[-1]) :
                dic2 = {
                    'Y_+': 0,
                    'Y_-': 0,
                }
                Y_GCN = Y_GCN.append(dic2, ignore_index=True)
                train_samples_weight = np.append(train_samples_weight, 0)
                test_samples_weight = np.append(test_samples_weight, 0)
                daily_max_score_of_stock = np.append(daily_max_score_of_stock, 0)
                Y_TRUE = np.append(Y_TRUE, 0)
                Y_prediction_Classic_models = np.append(Y_prediction_Classic_models, 0)

            else:
                dic2 = {
                    'Y_+': 1 if test_y.iloc[-1] == 1 else 0,
                    'Y_-': 1 if test_y.iloc[-1] == -1 else 0,
                }
                Y_GCN = Y_GCN.append(dic2, ignore_index=True)
                train_samples_weight = np.append(train_samples_weight, 0)
                if best_model['score'] <= threshold:
                    test_samples_weight = np.append(test_samples_weight, 1)
                    daily_max_score_of_stock = np.append(daily_max_score_of_stock, 0)
                else:
                    test_samples_weight = np.append(test_samples_weight, 0)
                    daily_max_score_of_stock = np.append(daily_max_score_of_stock, best_model['score'])
                Y_TRUE = np.append(Y_TRUE, test_y.iloc[-1])
                Y_prediction_Classic_models = np.append(Y_prediction_Classic_models, prediction)
                # ---------------------------------------------------------------------------
        else:
            list_stocks_without_data.append(col)

    if day == 1:
        if os.path.isfile('Network_0.1.npz'):
            adj = sparse.load_npz("Network_0.1.npz")
            adj = adj.todense() + np.identity(100)
            adj = np.where(adj > 0, 1, 0)
            Network = nx.DiGraph(adj)
            nodes_labels = {}
            for i in range(len(list_of_stocks)):
                nodes_labels[i] = list_of_stocks['stocks'].iloc[i]
            Network = nx.relabel_nodes(Network, nodes_labels)
            # Network_print(Network)

    Network_daily = Network.copy()
    for i in list_stocks_without_data:
        Network_daily.remove_node(i)
    temp = nx.adjacency_matrix(Network_daily)
    A = sparse.csr_matrix(temp)
    A = A.astype('float32')


    # ---------------------------------------------------------------------------------------------------------------
    number_of_test_sample_in_GCN = test_samples_weight.sum()
    test_samples_weight = test_samples_weight / test_samples_weight.sum()

    Graph_test = spektral.data.graph.Graph(x=X_GCN.astype('float32').values, a=A.astype('float32'), e=None,
                                           y=Y_GCN.astype('float32').values)
    list_of_Graphs = []
    list_of_Graphs.append(Graph_test)
    set_of_graphs_test = MyDataset()
    set_of_graphs_test.read()

    loader_test = SingleLoader(set_of_graphs_test, sample_weights=test_samples_weight)

    number_of_test_sample_in_classic = 0
    Confusion_matrix_Classic = pd.DataFrame(0, columns=['TP', 'FP', 'TN', 'FN'], index=list_stocks_roz)
    # Multiplying the threshold by the number of stocks for each day, high-precision labels are generated for network training.
    for max in range(0,len(daily_max_score_of_stock)):
        if daily_max_score_of_stock[max] >= threshold:
            Y_GCN['Y_+'].iloc[max] = 1 if Y_prediction_Classic_models[max] == 1 else 0
            Y_GCN['Y_-'].iloc[max] = 1 if Y_prediction_Classic_models[max] == -1 else 0
            train_samples_weight[max] = 1
            number_of_test_sample_in_classic += 1

            if Y_prediction_Classic_models[max] == Y_TRUE[max]:
                number_of_correct_predictions_of_classic_models += 1

            if Y_prediction_Classic_models[max] == 1 and Y_TRUE[max] == 1:
                Confusion_matrix_Classic['TP'].loc[list_stocks_roz[max]] += 1
            if Y_prediction_Classic_models[max] == -1 and Y_TRUE[max] == -1:
                Confusion_matrix_Classic['TN'].loc[list_stocks_roz[max]] += 1
            if Y_prediction_Classic_models[max] == 1 and Y_TRUE[max] == -1:
                Confusion_matrix_Classic['FP'].loc[list_stocks_roz[max]] += 1
            if Y_prediction_Classic_models[max] == -1 and Y_TRUE[max] == 1:
                Confusion_matrix_Classic['FN'].loc[list_stocks_roz[max]] += 1

        else:
            # The rest of the stocks take a value of 0 and are ineffective due to the weight of the samples in the training data.
            Y_GCN['Y_+'].iloc[max] = 0
            Y_GCN['Y_-'].iloc[max] = 0


    train_samples_weight = train_samples_weight / train_samples_weight.sum()

    Graph_train = spektral.data.graph.Graph(x=X_GCN.astype('float32').values, a=A.astype('float32'), e=None,
                                            y=Y_GCN.astype('float32').values)
    list_of_Graphs = []
    list_of_Graphs.append(Graph_train)
    set_of_graphs_train = MyDataset()
    set_of_graphs_train.read()

    loader_train = SingleLoader(set_of_graphs_train, sample_weights=train_samples_weight)
    # ---------------------------------------------------------------------------------------------------------------

    cnfig_result = np.append(cnfig_result, number_of_test_sample_in_GCN)
    cnfig_result = np.append(cnfig_result, number_of_test_sample_in_classic + number_of_test_sample_in_GCN)
    cnfig_result = np.append(cnfig_result, number_of_test_sample_in_classic)
    cnfig_result = np.append(cnfig_result,
                             number_of_correct_predictions_of_classic_models / number_of_test_sample_in_classic)



    my_eval = MyEval()
    tf.random.set_seed(seed=seed_tf)

    model = GNN(n_labels=set_of_graphs_train.n_labels, n_input_channels=set_of_graphs_train.n_node_features)
    model.compile(optimizer=Adam(1e-2), loss=CategoricalCrossentropy(reduction="sum"), weighted_metrics=["acc"], run_eagerly=True)
    model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch,
                        callbacks=[my_eval], shuffle=False, epochs=epoch, verbose=0)


    predictian_GCN = model.predict(loader_test.load(), steps=loader_test.steps_per_epoch, verbose=0)
    for i in range(len(predictian_GCN)):
        if predictian_GCN[i, 0] >= predictian_GCN[i, 1]:
            predictian_GCN[i, 0] = 1
        else:
            predictian_GCN[i, 0] = -1
    Confusion_matrix_daily = perf_measure(Y_TRUE, predictian_GCN[:, 0], sample_weight=test_samples_weight)


    Confusion_matrix_daily['Stock'] = list_stocks_roz
    Confusion_matrix_daily.set_index('Stock', inplace=True, drop=True)

    for index_stock in Confusion_matrix_daily.index:
        Confusion_matrix_daily['TP'].loc[index_stock] += Confusion_matrix_Classic['TP'].loc[index_stock]
        Confusion_matrix_daily['FP'].loc[index_stock] += Confusion_matrix_Classic['FP'].loc[index_stock]
        Confusion_matrix_daily['TN'].loc[index_stock] += Confusion_matrix_Classic['TN'].loc[index_stock]
        Confusion_matrix_daily['FN'].loc[index_stock] += Confusion_matrix_Classic['FN'].loc[index_stock]

    for index_stock in Confusion_matrix_daily.index:
        Confusion_matrix_total['TP'].loc[index_stock] += Confusion_matrix_daily['TP'].loc[index_stock]
        Confusion_matrix_total['FP'].loc[index_stock] += Confusion_matrix_daily['FP'].loc[index_stock]
        Confusion_matrix_total['TN'].loc[index_stock] += Confusion_matrix_daily['TN'].loc[index_stock]
        Confusion_matrix_total['FN'].loc[index_stock] += Confusion_matrix_daily['FN'].loc[index_stock]


    mat_acc_epoch = np.append(mat_acc_epoch, my_eval.acc_set)



    # ---------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------------

print('mean :DGAT_samples, Total_samples, PLD_samples, acc_PLD')
print('mean :', np.mean(cnfig_result.reshape(int(len(cnfig_result)/4),4), axis=0))

c = Confusion_matrix_total.copy()

c['sum'] = 0
c['acc'] = 0
c['mcc'] = 0
for i in range(len(c)):
    c['sum'].iloc[i] = c['TP'].iloc[i] + c['TN'].iloc[i] + c['FP'].iloc[i] + c['FN'].iloc[i]
    c['acc'].iloc[i] = (c['TP'].iloc[i] + c['TN'].iloc[i]) / c['sum'].iloc[i]
    c['mcc'].iloc[i] = ((c['TP'].iloc[i] * c['TN'].iloc[i]) - (c['FP'].iloc[i] * c['FN'].iloc[i])) / np.sqrt(
        (c['TP'].iloc[i] + c['FP'].iloc[i]) * (c['TP'].iloc[i] + c['FN'].iloc[i]) * (
                    c['TN'].iloc[i] + c['FP'].iloc[i]) * (c['TN'].iloc[i] + c['FN'].iloc[i]))

# Weighted average given the number of samples predicted for each stock
print('**** ACC_MEAN =', (c['sum'] * c['acc']).sum() / c['sum'].sum())
print('**** MCC_MEAN =', (c['sum'] * c['mcc']).sum() / c['sum'].sum())

c.set_index(list_of_stocks['stocks'], inplace=False, drop=True).to_csv('Confusion_matrix_total_IRAN_2.csv', index=True, header=True)


# -------------------------------------------------------------------------------------------------------------------


