# ============================================================================
# import libraries
# ============================================================================
print("=====Import libraries=====")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', 20)
import tensorflow as tf

# ============================================================================
# prepare data
# ============================================================================
print("=====Prepare data=====")

df = pd.read_csv('resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv')

# convert data to numeric types which includes one hot encoding
df['months_past_2017_Jan'] = df['month'].apply(lambda x: (int(x.split("-")[0]) - 2017) * 12 + int(x.split("-")[1]))
df = pd.concat([df, pd.get_dummies(df['flat_type'], prefix='flat_type')], axis=1)
df = pd.concat([df, pd.get_dummies(df['town'], prefix='town')], axis=1)
df['storey'] = df['storey_range'].apply(lambda x: np.random.randint(low=int(x.split(" ")[0]), high=int(x.split(" ")[2])+1, size=1)[0])
df = pd.concat([df, pd.get_dummies(df['flat_model'], prefix='flat_model')], axis=1)
def lease_helper(x):
    lst = x.split(" ")
    if len(lst) == 2:
        return int(lst[0]) * 12
    else:
        return int(lst[0]) * 12 + int(lst[2])
df['remaining_lease_months'] = df['remaining_lease'].apply(lease_helper)

# plots various graph to look at relationship between features and label
plt.scatter(df['floor_area_sqm'], df['resale_price'], alpha=0.75, s=0.6)
plt.ylabel('resale_price')
plt.xlabel('floor_area_sqm')
plt.savefig('floor_area_sqm.png')
plt.clf()

month_against_price = df.groupby('months_past_2017_Jan')['resale_price'].mean()
plt.plot(month_against_price.index.values, month_against_price.values)
plt.ylabel('resale_price')
plt.xlabel('months_past_2017_Jan')
plt.savefig('months_past_2017_Jan.png')
plt.clf()

storey_against_price = df.groupby('storey')['resale_price'].mean()
plt.plot(storey_against_price.index.values, storey_against_price.values)
plt.ylabel('resale_price')
plt.xlabel('storey')
plt.savefig('storey.png')
plt.clf()

lease_months_against_price = df.groupby('remaining_lease_months')['resale_price'].mean()
plt.plot(lease_months_against_price.index.values, lease_months_against_price.values)
plt.ylabel('resale_price')
plt.xlabel('remaining_lease_months')
plt.savefig('remaining_lease_months.png')
plt.clf()

flat_type = df['flat_type'].unique()
flat_type_against_price = []
for flat in flat_type:
    flat_type_against_price.append(df[df['flat_type'] == flat]['resale_price'].values)
plt.figure(figsize=(10, 10))
plt.boxplot(flat_type_against_price, labels=flat_type)
plt.ylabel('resale_price')
plt.xticks(rotation=75)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('flat_type.png')
plt.clf()

towns = df['town'].unique()
town_against_price = []
for town in towns:
    town_against_price.append(df[df['town'] == town]['resale_price'].values)
plt.figure(figsize=(20, 10))
plt.boxplot(town_against_price, labels=towns)
plt.ylabel('resale_price')
plt.xticks(rotation=75)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('town.png')
plt.clf()

models = df['flat_model'].unique()
models_against_price = []
for flat_model in models:
    models_against_price.append(df[df['flat_model'] == flat_model]['resale_price'].values)
plt.figure(figsize=(20, 10))
plt.boxplot(models_against_price, labels=models)
plt.ylabel('resale_price')
plt.xticks(rotation=75)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('flat_model.png')
plt.clf()

df = df.drop(columns=['month', 'flat_type', 'town', 'block', 'street_name', 'storey_range', 'flat_model', 'lease_commence_date', 'remaining_lease'])

# normalize features
for col in df.columns:
    d = df[col]
    max_d = d.max()
    min_d = d.min()
    if col == 'resale_price':
        resale_max = max_d
        resale_min = min_d
    if max_d and min_d == 0:
        continue
    else:
        df[col] = (df[col] - min_d) / (max_d - min_d)

# split data into training and test data with ratio 8:1 respectively
train_data = df.sample(frac=0.9, random_state=0)
test_data = df.drop(train_data.index)

print('training data shape:', train_data.shape)
print('test data shape:', test_data.shape)

# split into features and label
X_cols = df.columns[df.columns != 'resale_price']
Y_col = 'resale_price'
print('Features:', X_cols.values)
print('Label:', Y_col)

X_train, Y_train = train_data[X_cols], train_data[Y_col]
X_test, Y_test = test_data[X_cols], test_data[Y_col]

# ============================================================================
# train model
# ============================================================================
print("=====Train model=====")

def plot_loss(history, name):
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(name + '_loss.png')
    plt.clf()

# linear regression model
print('linear model:')
try:
    linear_model = tf.keras.models.load_model('linear_model.h5')
except:
    linear_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])

    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_absolute_error'
    )

    history = linear_model.fit(
        X_train,
        Y_train,
        epochs=100,
        verbose=2,
        validation_split=(1/8)
    )

    plot_loss(history, 'linear_model')

    linear_model.save('linear_model.h5')

# deep neural network model
print('deep neural network model:')
try:
    nnm = tf.keras.models.load_model('nnm.h5')
except:
    nnm = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(units=8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(units=1)
    ])

    nnm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_absolute_error'
    )

    history = nnm.fit(
        X_train,
        Y_train,
        epochs=100,
        verbose=2,
        validation_split=(1/8)
    )

    plot_loss(history, 'neural_network_model')

    nnm.save('nnm.h5')

# ============================================================================
# optimize deep neural network model
# ============================================================================
print("=====optimize model=====")

# looks for best learning rate for optimizer
data = {}
for rate in [0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003]:
    nnm = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(units=8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(units=1)
    ])

    nnm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=rate),
        loss='mean_absolute_error'
    )

    history = nnm.fit(
        X_train,
        Y_train,
        epochs=50,
        verbose=2,
        validation_split=(1/8)
    )

    data[rate] = history.history['val_loss'][-1]

print(data)
best_learning_rate = 0
lowest_cost = min(data.values())
for k, v in data.items():
    if v == lowest_cost:
        best_learning_rate = k
print('best_learning_rate:', best_learning_rate) # 0.0003

# looks for best lambda for regularizer
data = {}
for lambda_ in [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.01, 0.1]:
    nnm = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dense(units=8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
        tf.keras.layers.Dense(units=1)
    ])

    nnm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate),
        loss='mean_absolute_error'
    )

    history = nnm.fit(
        X_train,
        Y_train,
        epochs=50,
        verbose=2,
        validation_split=(1/8)
    )

    data[lambda_] = history.history['val_loss'][-1]

print(data)
best_lambda_ = 0
lowest_cost = min(data.values())
for k, v in data.items():
    if v == lowest_cost:
        best_lambda_ = k
print('best_lambda_:', best_lambda_) # 0.00001

# ============================================================================
# Final fit and predict with test data
# ============================================================================
print("=====Final fit and predict with test data=====")

try:
    final_nnm = tf.keras.models.load_model('final_nnm.h5')
except:
    best_learning_rate = 0.0003
    best_lambda_ = 0.0001
    final_nnm = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(best_lambda_)),
            tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(best_lambda_)),
            tf.keras.layers.Dense(units=8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(best_lambda_)),
            tf.keras.layers.Dense(units=1)
        ])

    final_nnm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_learning_rate),
        loss='mean_absolute_error'
    )

    history = final_nnm.fit(
        X_train,
        Y_train,
        epochs=100,
        verbose=2,
        validation_split=(1/8)
    )

    plot_loss(history, 'final_neural_network_model')

    final_nnm.save('final_nnm.h5')

test_predictions = final_nnm.predict(
    X_test,
    verbose=0
).flatten()


pred = test_predictions[:10] * (resale_max - resale_min) + resale_min
act = Y_test[:10] * (resale_max - resale_min) + resale_min
predictions = pd.DataFrame({
    'predicted': pred,
    'actual': act
})
print(predictions)
#        predicted    actual
# 2   295602.06250  262000.0
# 10  302733.93750  288500.0
# 21  338846.15625  321000.0
# 27  354275.84375  335000.0
# 30  351532.12500  373000.0
# 43  478019.40625  518000.0
# 46  686999.00000  688000.0
# 55  802243.06250  888000.0
# 71  317393.68750  300000.0
# 75  319095.18750  310000.0
