from keystroke_libfiles import *


'''
This file can transform a folder with .wav files to a numpy dataset, or another folder with modified .wav files
'''

# mode: 0 for single file testing, 1 for entire folder executing
mode = 1
# data_type: 0 to store as .wav file, 1 for store as .npz file
data_type = 1
# filepath = 'records/kb1_noise0.5_1100ms/.381.wav'
# filepath = 'records/kb1_noise9_1100ms/'
filepath = 'record_denoised/kb1_noise9_1100ms_denoised/'


'''
update 2.0: enable combination of keystroke and noise
'''
# noise: When set to 1, the ouput will be the noise part of the given data instead of the keystroke part .
noise = 0
# combine: set to 0 has no influence. When set to 1, the noise part in the data from 'combine_path' will mix with the keystroke part in the data from 'filepath'
combine = 0
combine_path = 'records/kb1_noise4_1100ms/'




def single_file(filepath):
    sr, x = wavreadlocal(filepath)

    thre = np.max(x)/5

    print(f"x.shape:{np.shape(x)}")

    keystroke_x = locate_keystroke_maxdiv(x, 5, 9000)
    print(f"keystroke_x.shape:{keystroke_x.shape}")

    line_x = np.array([thre]*np.shape(x)[0])
    line_key = np.array([thre]*np.shape(keystroke_x)[0])

    fig, ax = plt.subplots(2,1)
    fig.tight_layout(h_pad=2, w_pad=1)

    ax[0].plot(np.linspace(0,np.shape(keystroke_x)[0],np.shape(keystroke_x)[0]), line_key)
    ax[0].plot(np.linspace(0,np.shape(keystroke_x)[0],np.shape(keystroke_x)[0]), keystroke_x)
    ax[0].set_title('extracted keystroke')

    ax[1].plot(np.linspace(0,np.shape(x)[0],np.shape(x)[0]), line_x)
    ax[1].plot(np.linspace(0,np.shape(x)[0],np.shape(x)[0]), x)
    ax[1].set_title(f'original sound data {filepath}')

    plt.show()


def multi_file(filepath):

    data_x = []
    label_y = []

    file_list = filepath.split("/")

    if not data_type:
        if not os.path.exists("record_cut/"):
            os.makedirs("record_cut/")

        if not os.path.exists(f"record_cut/{file_list[-2]}/"):
            os.makedirs(f"record_cut/{file_list[-2]}/")
        

    # count = 0
    for file in os.listdir(filepath):

        sr, x = wavreadlocal(filepath+file)

        if noise:
            keystroke_x = locate_noise_maxdiv(x, 1.01, 9000)
        else:
            keystroke_x = locate_keystroke_maxdiv(x, 1.01, 9000)

        print(len(keystroke_x))
        data_x.append(keystroke_x)
        if file[0].isnumeric():
            label_y.append(' ')
        else:
            label_y.append(file[0])

        if not data_type:
            wavwritelocal(f'record_cut/{file_list[-2]}/{file}', sr = sr, data = keystroke_x)

        # count+=1
        # if count > 3:
        #     break

    np_data_x = np.array(data_x)
    np_data_y = np.array(label_y)

    print(f"np_data_x:{np_data_x.shape}")
    print(f"np_data_y:{np_data_y.shape}")

    
    if not os.path.exists("data/"):
        os.makedirs("data/")
    
    if data_type:
        np.savez(f"data/{file_list[-2]}", x=np_data_x, y=np_data_y)

    plt.show()


def multi_file_combine(filepath,combine_path):

    data_x = []
    label_y = []

    file_list = filepath.split("/")

    if not data_type:
        if not os.path.exists("record_cut/"):
            os.makedirs("record_cut/")

        if not os.path.exists(f"record_cut/comb_{file_list[-2]}/"):
            os.makedirs(f"record_cut/comb_{file_list[-2]}/")
        
    file_dir = os.listdir(filepath)
    noise_dir = os.listdir(combine_path)

    count = 0
    for file in file_dir:

        sr, x_data = wavreadlocal(filepath+file)
        sr, noise_data = wavreadlocal(combine_path+random.choice(noise_dir))
        
        keystroke_x = locate_keystroke_maxdiv(x_data, 3, 9000)
        noise_x = locate_noise_maxdiv(noise_data, 1.001, 9000)

        print(f"keystroke_x.shape:{keystroke_x.shape}")
        print(f"noise_x.shape:{noise_x.shape}")

        keystroke_x = keystroke_x + noise_x
        print(keystroke_x.shape)

        data_x.append(keystroke_x)
        if file[0].isnumeric():
            label_y.append(' ')
        else:
            label_y.append(file[0])

        if not data_type:
            wavwritelocal(f'record_cut/comb_{file_list[-2]}/{file}', sr = sr, data = keystroke_x)

        count+=1
        # if count > 3:
        #     break


    np_data_x = np.array(data_x)
    np_data_y = np.array(label_y)

    print(f"np_data_x:{np_data_x.shape}")
    print(f"np_data_y:{np_data_y.shape}")

    
    # print(f"data:{file_list[-2]}")
    if not os.path.exists("data/"):
        os.makedirs("data/")
    
    if data_type:
        np.savez(f"data/comb_{file_list[-2]}", x=np_data_x, y=np_data_y)

    plt.show()


if not mode:
    single_file(filepath)
elif not combine:
    multi_file(filepath)
else:
    multi_file_combine(filepath,combine_path)