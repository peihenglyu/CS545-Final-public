from keystroke_libfiles import *


'''
This file can capture keystroke in real time and save it at data/ in .wav form
'''

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text
    

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1,2], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=1100, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()
q_raw = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])
    q_raw.put(indata[::, mapping])


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    global plotdata2

    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data

    for column, line in enumerate(lines1):
        line.set_ydata(plotdata[:, column])
    for column2, line2 in enumerate(lines2):
        line2.set_ydata(plotdata2[:, column2])
    return lines1 + lines2


def user_input(child_conn):
    """

    This function capture the user input on keyboard in real time
    The keystroke in plotdata will be found and extract as data when user press letter keys

    """
    global plotdata
    global plotdata2
    global key_char
    global counter
    print(f"\n user input thread started, press function key such as \'Delete\', \'Insert\' to end this thread")
    while True:
        try:
            # msvcrt.getch() return byte string, so decode it into normal string
            key_char = msvcrt.getch().decode("utf-8")
        except:
            print(f"key press detection failed")
            break
        # any special function key will exit the function
        if key_char in ['\000','\xe0','\x00','\x08']:
            print("exit user input")
            return 1
        else:
            print(key_char)
            time.sleep(0.4)

            plotdata2 = plotdata
            plotdata = np.zeros_like(plotdata)
            wavwritelocal(f'records/{key_char}{counter}.wav', sr = int(args.samplerate), data = plotdata2)
            counter += 1
            

    return 1


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']
        print(f"sample rate:{args.samplerate}")

    if not os.path.exists("records/"):
            os.makedirs("records/")

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))
    plotdata2 = np.zeros((length, len(args.channels)))
    # plotdata2 = np.zeros((int(args.window * args.samplerate/1000), len(args.channels)))
    raw_audio_data = np.zeros((int(args.window * args.samplerate/1000), len(args.channels)))

    # number of figure
    fig_num = 2
    fig, (ax1,ax2) = plt.subplots(fig_num,1)
    axs = [ax1,ax2]
    
    lines1 = (ax1.plot(plotdata))
    lines2 = (ax2.plot(plotdata2))
    line_list = [lines1, lines2]


    if len(args.channels) > 1:
        ax1.legend([f'channel {c}' for c in args.channels],
                loc='lower left', ncol=len(args.channels))
    ax1.axis((0, len(plotdata), -1, 1))
    ax1.set_yticks([0])
    ax1.yaxis.grid(True)
    ax1.tick_params(bottom=False, top=False, labelbottom=False,
                right=False, left=False, labelleft=False)
    
    if len(args.channels) > 1:
        ax2.legend([f'channel {c}' for c in args.channels],
                loc='lower left', ncol=len(args.channels))
    ax2.axis((0, len(plotdata), -0.1, 0.25))
    # ax2.set_yticks([0])
    # ax2.yaxis.grid(True)
    ax2.tick_params(bottom=False, top=False, labelbottom=False,
                right=False, left=False, labelleft=False)

    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)

    # The thread to capture user input
    key_char = None
    data_conn, data_conn2 = Pipe()
    key_conn, key_conn2 = Pipe()
    counter = 0
    t1 = threading.Thread(target = user_input, args = (data_conn,))
    t1.start()

    with stream:
        plt.show()

    t1.join()

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))