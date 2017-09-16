import numpy as np

def fit_model(model, EPOCH_NR, MODEL_NAME, train_data, IMG_SIZE):
    train = train_data[:-5000]
    test = train_data[-5000:]

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, EPOCH_NR, validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=2000, show_metric=True, run_id=MODEL_NAME)