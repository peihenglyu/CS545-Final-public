from keystroke_libfiles import *


'''
This file build the denoise model by taking .npz form file of sound data. The denoise model will be saved in /denoise_model
'''

filepath = "data/kb1_noise1.1_1100ms.npz"




def fre_axis(sr_sound,x_sound,title):
    fig, ax = plt.subplots(1,1)
    ax.plot(np.linspace(0,np.shape(x_sound)[0],np.shape(x_sound)[0]), x_sound)
    ax.set_title(title)

# function to train the w and h based on given stft data
def standard_train(x_stft,k):
    # initialize f, w ,and h matrices
    epsilon = 1e-7
    rng = np.random.default_rng()
    f = np.abs(x_stft).T
    m,n = np.shape(f)
    w = rng.random((m,k))+10
    h = rng.random((k,n))+10

    # iterate the learning process
    iter_num = 250
    for i in range(iter_num):
        reconstruction = np.dot(w, h)
        v = f/(reconstruction+epsilon)
        h = h*np.dot(w.T,v)
        w = w*np.dot(v,h.T)
        w = w / np.sum(w, axis=0)

        error = np.linalg.norm(f - reconstruction, 'fro')  # Frobenius norm
        if i%2 == 0:
            print(f"round {i}/{iter_num}. error:{error}")
    return w,h

def build_denoise(filepath):


    if not os.path.exists("denoise_model/"):
        os.makedirs("denoise_model/")

    data_x = np.load(filepath)['x']
    print(f"data_x.shape:{data_x.shape}")
    shape = data_x.shape
    data_x = data_x.reshape(shape[0]*shape[2],shape[1])
    print(f"data_x.shape:{data_x.shape}")

    # stft step
    dft_size = 1024
    hop_size = int(1/8 * dft_size)
    zero_pad = dft_size
    window = np.hamming(dft_size)
    temp_list = []
    for data in data_x:
        temp_list.append(stft(data, dft_size, hop_size, zero_pad, window)) 

    stft_data = np.array(temp_list)
    print(f"stft_data.shape:{stft_data.shape}")

    x_shape = data_x.shape
    data_x = data_x.reshape(x_shape[0]*x_shape[1])
    stft_shape = stft_data.shape
    stft_data = stft_data.reshape(stft_shape[0]*stft_shape[1], stft_shape[2])
    stft_phase = np.exp(1j * np.angle(stft_data))

    # stft_axis(44100, data_x, stft_data, "random stft_img")
    # istft_rand_data = istft(stft_data, dft_size, hop_size, zero_pad, window)
    # sound( data_x, rate=44100, label='')
    # sound( istft_rand_data, rate=44100, label='')


    # rand_index = np.random.randint(low = 0, high = stft_data.shape[0]-1, size = 1)[0]
    # print(rand_index)
    # stft_axis(44100, data_x[rand_index], stft_data[rand_index], "random stft_img")
    # istft_rand_data = istft(stft_data[rand_index], dft_size, hop_size, zero_pad, window)

    # def pad_sound(x):
    #     return np.pad(x, (20000, 20000), 'constant', constant_values=(0))
    
    # sound( pad_sound(data_x[rand_index]), rate=44100, label='')
    # sound( pad_sound(istft_rand_data), rate=44100, label='')
    
    plt.show()

    k = 120
    w_data,h_data = standard_train(stft_data,k)

    # reconstruct_stft = np.dot(w_data, h_data) * stft_phase
    # reconstruct_istft = istft(reconstruct_stft, dft_size, hop_size, zero_pad, window)
    # sound(reconstruct_istft, rate = 44100, label='reconstruct sound')

    file_list = filepath.split("/")
    np.savez(f"denoise_model/w_{file_list[-1]}", x=w_data)


    return 0


build_denoise(filepath)