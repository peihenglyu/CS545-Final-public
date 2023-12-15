from keystroke_libfiles import *


'''
This file build the cnn model by taking one or more .npz file
'''
data_paths = ["data/kb1_noise0.5_1100ms.npz", "data/kb1_noise1_1100ms.npz", "data/kb1_noise1.1_1100ms.npz", "data/kb1_noise3_1100ms.npz", "data/kb1_noise4_1100ms.npz", ]
# data_paths = ["data/kb1_noise0.5_1100ms.npz", "data/kb1_noise1_1100ms.npz","data/kb1_noise1.1_1100ms.npz", "data/kb1_noise3_1100ms.npz", "data/kb1_noise4_1100ms.npz", "data/kb1_noise9_1100ms.npz"]
# data_paths = ["data/kb1_noise0.5_1100ms_denoised.npz", "data/kb1_noise1_1100ms_denoised.npz","data/kb1_noise1.1_1100ms_denoised.npz", "data/kb1_noise3_1100ms_denoised.npz", "data/kb1_noise4_1100ms_denoised.npz"]
# data_paths = ["data/kb2_noise1_1100ms_heavy_center.npz", "data/kb2_noise1_1100ms_light_center.npz", "data/kb2_noise1_1100ms_median_center.npz", "data/kb2_noise1_1100ms_median_corner.npz" ]
# mode 0 for .npz file, 1 for .wav file (not implemented yet)
mode = 0
# train_num_list = [12, 10, 40, 7, 7]
train_num_list = [12, 10, 40, 7, 7]



# stft step
dft_size, hop_size, zero_pad, stft_window = get_stft_param()

epsilon = 1e-7


def make_data(index_list, data_x, prepro_type):
    prepro_x_list = []
    for index in index_list:
        data = data_x[index]
        prepro_x_list.append(np.array(preprocessing_data(data, prepro_type)))

    return prepro_x_list


def train_model(data_paths):

    if not os.path.exists("cnn_model/"):
        os.makedirs("cnn_model/")

    if not os.path.exists("label_match/"):
        os.makedirs("label_match/")

    preprocessed_train_list = []
    preprocessed_test_list = []
    train_y = []
    test_y = []
    

    dict_key_label, dict_label_key = get_label_29keys()


    dict_count = 0
    if len(data_paths) != len(train_num_list):
        print("\nlength of train_num_list doesn't match the number of file in data_paths, please modify train_num_list to match the length\n")
    for file in data_paths:
        dict_class = {}
        # train_num = train_num_list[dict_count]
        print(f"loaded:{file}")
        data_x = np.load(file)['x']
        data_y = np.load(file)['y']

        # record the index of each data and its label for later train and test dataset assignment
        for i in range(len(data_y)):
            y = data_y[i]
            if dict_key_label[y] in dict_class:
                dict_class[dict_key_label[y]].append(i)
            else:
                dict_class[dict_key_label[y]] = [i]

        for key in dict_class:
            # select random file for training and testing
            temp_class = []
            temp_train = []
            temp_test = []

            temp_class.extend(dict_class[key])
            train_num = train_num_list[dict_count]
            if train_num > len(temp_class):
                train_num = len(temp_class)
            temp_train = random.sample(temp_class,train_num)
            train_y.extend([key]*train_num)
            # print(f"temp_class:{temp_class}")
            for index in temp_class:
                if index not in temp_train:
                    temp_test.append(index)
                    test_y.append(key)

            preprocessed_train = make_data(temp_train, data_x, "stft")
            preprocessed_test = make_data(temp_test, data_x, "stft")
            preprocessed_train_list.extend(preprocessed_train)
            preprocessed_test_list.extend(preprocessed_test)

        
        dict_count += 1

    preprocessed_train_x = np.asarray(preprocessed_train_list)
    preprocessed_test_x = np.asarray(preprocessed_test_list)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    print("np.shape(preprocessed_train_x):",np.shape(preprocessed_train_x))
    print("np.shape(preprocessed_test_x):",np.shape(preprocessed_test_x))
    print("np.shape(train_y):",np.shape(train_y))
    print("np.shape(test_y):",np.shape(test_y))


    # prepare the data for PyTorch
    train_x = torch.from_numpy(preprocessed_train_x).float()
    train_y = torch.from_numpy(train_y).long()
    test_x = torch.from_numpy(preprocessed_test_x).float()
    # test_y = torch.from_numpy(test_y).long()


    # create TensorDataset and DataLoader for the training data
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    # initialize the CNN model, loss function, and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # train the model
    num_epochs = 200
    count = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            # forward pass, backward pass, and optimization
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print the average loss
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}")

        # print("Finished training")

        # make predictions on the test data
        with torch.no_grad():
            test_outputs = model(test_x)
            _, predicted = torch.max(test_outputs, 1)

        predictions = predicted.numpy()

        # show accuracy
        accuracy = np.sum(predictions == test_y) / len(test_y)
        print(f"Accuracy: {accuracy}")
    

        if epoch >= num_epochs-3:
            torch.save(model, f"cnn_model/cnn{count}")
            count += 1

train_model(data_paths)