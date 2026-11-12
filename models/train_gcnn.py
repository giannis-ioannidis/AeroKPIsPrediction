"""Train the GCNN (graph of flight-phase agents + multi-head attention) on the
normalised Ppr or Opr feature CSVs.

Each flight row holds 9 sub-phases; every 3 consecutive sub-phases form one
agent (Climbing, Cruising, Descending), so each agent sees len_feature columns.
Agents encode their features with a shared MLP, exchange information through
two multi-head attention layers restricted to neighbouring phases, and each
regresses the target; the final prediction is the mean of the agents.

Usage:
    python models/train_gcnn.py --set ppr --target ci --epochs 50
"""
import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Concatenate, Dense, Dropout, Flatten, Input, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam

#define constants
SETS = {
    # features per agent, dropout on the regressor (as in the original scripts)
    'ppr': {'len_feature': 51, 'dropout': None},
    'opr': {'len_feature': 69, 'dropout': 0.2},
}
TARGETS = {'ci': 'Cost_Index', 'payload': 'Max_Payload'}
max_dist = 1
n_phases = 3
classes = ['Climbing', 'Crusing', 'Descending']


def Adjacency(class_list):
	"""One n_phases x n_phases matrix per agent that selects the agent itself
	and its neighbours within max_dist."""
	out = []
	for phase in class_list:
		tmp1 = []
		for cls in class_list:
			tmp = [0] * len(class_list)
			if abs(class_list.index(phase) - class_list.index(cls)) <= max_dist:
				tmp[class_list.index(cls)] = 1
			tmp1.append(tmp)
		out.append(tmp1)
	return np.asarray(out, dtype='float32')


def MLP(len_feature):

	In_0 = Input(shape=(len_feature,))
	h = Dense(128, activation='relu', kernel_initializer='random_normal')(In_0)
	h = Dense(128, activation='relu', kernel_initializer='random_normal')(h)
	h = Reshape((1, 128))(h)
	return Model(inputs=In_0, outputs=h, name='encoder')


def MultiHeadsAttModel(l=2, d=128, dv=16, dout=128, nv=8, name=None):

	v1 = Input(shape=(l, d))
	q1 = Input(shape=(l, d))
	k1 = Input(shape=(l, d))
	ve = Input(shape=(1, l))

	v2 = Dense(dv*nv, activation="relu", kernel_initializer='random_normal')(v1)
	q2 = Dense(dv*nv, activation="relu", kernel_initializer='random_normal')(q1)
	k2 = Dense(dv*nv, activation="relu", kernel_initializer='random_normal')(k1)

	v = Reshape((l, nv, dv))(v2)
	q = Reshape((l, nv, dv))(q2)
	k = Reshape((l, nv, dv))(k2)
	v = Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(v)
	k = Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(k)
	q = Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(q)

	# scaled dot-product attention per head: (nv, l, l)
	att = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True) / np.sqrt(dv))([q, k])
	att = Lambda(lambda x: tf.nn.softmax(x))(att)
	out = Lambda(lambda x: tf.matmul(x[0], x[1]))([att, v])
	out = Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(out)

	out = Reshape((l, dv*nv))(out)

	# keep only the row of the current agent
	T = Lambda(lambda x: tf.matmul(x[0], x[1]))([ve, out])

	out = Dense(dout, activation="relu", kernel_initializer='random_normal')(T)
	return Model(inputs=[q1, k1, v1, ve], outputs=out, name=name)


def Regressor(dropout):

	I1 = Input(shape=(1, 128))
	I2 = Input(shape=(1, 128))
	I3 = Input(shape=(1, 128))

	h = Concatenate()([Flatten()(I1), Flatten()(I2), Flatten()(I3)])

	V = Dense(128, activation="relu", kernel_initializer='random_normal')(h)
	if dropout:
		V = Dropout(dropout)(V)
	V = Dense(48, activation="relu", kernel_initializer='random_normal')(V)
	if dropout:
		V = Dropout(dropout)(V)
	V = Dense(1, kernel_initializer='random_normal')(V)

	return Model(inputs=[I1, I2, I3], outputs=V, name='regressor')


def out_round(x):
	"""Round to 1 decimal in the forward pass, identity gradient in the backward pass."""
	rounded = tf.round(x * 10) / 10
	return x - tf.stop_gradient(x - rounded)


def build_model(len_feature, dropout):

	encoder = MLP(len_feature)
	m1 = MultiHeadsAttModel(l=n_phases, name='attention_1')
	m2 = MultiHeadsAttModel(l=n_phases, name='attention_2')
	reg = Regressor(dropout)

	In = []
	for j in range(n_phases):
		In.append(Input(shape=(len_feature,), name=f'features_{j}'))
		In.append(Input(shape=(n_phases, n_phases), name=f'adjacency_{j}'))
	In.append(Input(shape=(1, n_phases), name='selector'))

	feature = [encoder(In[j*2]) for j in range(n_phases)]
	feature_ = Concatenate(axis=1)(feature)

	relation1 = []
	for j in range(n_phases):
		T = Lambda(lambda x: tf.matmul(x[0], x[1]))([In[j*2+1], feature_])
		relation1.append(m1([T, T, T, In[n_phases*2]]))
	relation1_ = Concatenate(axis=1)(relation1)

	relation2 = []
	for j in range(n_phases):
		T = Lambda(lambda x: tf.matmul(x[0], x[1]))([In[j*2+1], relation1_])
		relation2.append(m2([T, T, T, In[n_phases*2]]))

	V = [reg([feature[j], relation1[j], relation2[j]]) for j in range(n_phases)]

	#mean of agents
	Out = Lambda(lambda x: tf.reduce_mean(tf.stack(x), axis=0))(V)
	Rounded_Out = Lambda(out_round)(Out)

	model = Model(inputs=In, outputs=Rounded_Out)
	model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
	return model


def make_inputs(df, len_feature, target_col):
	"""Build the 2*n_phases+1 model inputs and the target vector for every row."""
	n = len(df)
	X = df.iloc[:, :n_phases * len_feature].to_numpy(dtype='float32')
	adj = Adjacency(classes)
	vec = np.zeros((n, 1, n_phases), dtype='float32')
	vec[:, 0, 0] = 1

	inputs = []
	for j in range(n_phases):
		inputs.append(X[:, j * len_feature:(j + 1) * len_feature])
		inputs.append(np.repeat(adj[j][None], n, axis=0))
	inputs.append(vec)

	y = df[target_col].to_numpy(dtype='float32').reshape(-1, 1)
	return inputs, y


def main():
	parser = argparse.ArgumentParser(description=__doc__,
									 formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--set', required=True, choices=sorted(SETS))
	parser.add_argument('--target', default='ci', choices=sorted(TARGETS),
						help='ci = Cost_Index, payload = Max_Payload (MPL in the file names)')
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--data-dir', default='sample_data/processed')
	parser.add_argument('--out-dir', default='sample_data/runs')
	parser.add_argument('--seed', type=int, default=31)
	args = parser.parse_args()

	random.seed(args.seed)
	keras.utils.set_random_seed(args.seed)

	cfg = SETS[args.set]
	len_feature = cfg['len_feature']
	target_col = TARGETS[args.target]
	prefix = os.path.join(args.data_dir, args.set.capitalize())

	train_df = pd.read_csv(prefix + 'TrainNorm.csv').sample(frac=1, random_state=args.seed)
	test_df = pd.read_csv(prefix + 'TestNorm.csv')
	assert train_df.shape[1] == n_phases * len_feature + 2, \
		f'expected {n_phases * len_feature} features + 2 labels, found {train_df.shape[1]} columns'

	x_train, y_train = make_inputs(train_df, len_feature, target_col)
	x_test, y_test = make_inputs(test_df, len_feature, target_col)

	model = build_model(len_feature, cfg['dropout'])
	history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
						validation_data=(x_test, y_test), verbose=2)

	os.makedirs(args.out_dir, exist_ok=True)
	run_name = f'{args.set}_{args.target}'

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title(f'{args.set.upper()} - {target_col} (MAE)')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.savefig(os.path.join(args.out_dir, run_name + '_loss.png'))

	# Lambda layers make full-model files fragile to reload; the architecture is
	# rebuilt with build_model(), so only the weights are stored.
	model.save_weights(os.path.join(args.out_dir, run_name + '.weights.h5'))

	print(f"Final MAE  train: {history.history['loss'][-1]:.3f}  test: {history.history['val_loss'][-1]:.3f}")
	print(f'Baseline MAE (always predict train mean): '
		  f'{np.abs(y_test - y_train.mean()).mean():.3f}')
	print(f'Saved {run_name}_loss.png and {run_name}.weights.h5 to {args.out_dir}')


if __name__ == '__main__':
	main()
