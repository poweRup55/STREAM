import ctypes
import threading

import cv2.cv2 as cv2
import imutils
import numpy as np
import pyaudio
from scipy import ndimage

CHANGING_IMAGE_COEF = 150
STILL_IMAGE_COEF = 200
FRAME_COMP_HEIGHT = 600
EXIT_KEY = 27
REMOVE_NOISE_SIGMA = 2
BYTE_MAX = 255
ERROR_RETURN = -1
NUM_OF_CHANNELS = 1
MBOX_STYLE = 0
GAUSSIAN_SIGMA = 10
SAMPLING_RATE = 8000
SINE_FREQUENCY = 50.0
SINE_BEEP_DURATION = 1 / 10
ERROR = 'ERROR'
WINDOW = "window"
CANNOT_OPEN_WEBCAM_ERROR = "Cannot open webcam"
NO_AUDIO_DEVICE_FOUND_ERROR = 'No audio device found!'


def image_2_sound(frame_on_screen, frame_diff_a, frame_diff_b, screen_size, audio_stream, mutex):
    """
    Makes a sin wave sound in respect to the shown image and the change between two frames
    :param frame_on_screen: The image that is shown on screen - 2d numpy matrix
    :param frame_diff_a: An image from the camera - 2d numpy matrix
    :param frame_diff_b: A similar but different image from the camera - 2d numpy matrix
    :param screen_size: Screen size tuple
    :param audio_stream: An output audio stream
    :param mutex: a mutex
    """
    frame_on_screen = np.round(frame_on_screen / BYTE_MAX).astype(np.int8)
    diff, diff_size = find_diff(frame_diff_a, frame_diff_b)
    changing_image_sin_x = (np.sum(diff) / diff_size) * CHANGING_IMAGE_COEF
    shown_image_sin_x = (np.sum(frame_on_screen) / screen_size) * STILL_IMAGE_COEF
    sin_x = shown_image_sin_x + changing_image_sin_x
    x = np.linspace(0, SINE_BEEP_DURATION, int(SAMPLING_RATE * SINE_BEEP_DURATION), endpoint=False)
    frequencies = x * SINE_FREQUENCY
    samples = (np.sin(sin_x * frequencies)).astype(np.float32).tobytes()
    audio_stream.write(samples)
    mutex.release()


def water_stream_loop(audio_stream, cap):
    """
    Makes the frame watery and makes an appropriate sine wave sound
    :param audio_stream: An audio output stream
    :param cap: web camera cv2 stream
    :return:
    """
    screen_size = get_screen_size()
    num_of_pixels = screen_size[0] * screen_size[1]
    mutex = threading.Lock()
    ret, cam_frame = cap.read()  # Gets image from webcam
    cam_frame_copy = imutils.resize(cam_frame, height=FRAME_COMP_HEIGHT)
    frame_hold = cam_frame_copy.copy()
    copy_frame = False  # A flag to mark if an input image should be copied
    while True:
        # Show image
        watered_frame = filter_to_water(cam_frame, screen_size)
        cv2.imshow(WINDOW, watered_frame)
        c = cv2.waitKey(1)
        if c == EXIT_KEY:
            break

        # Sound Handler
        if not mutex.locked():
            mutex.acquire()
            sound_thread = threading.Thread(target=image_2_sound, args=(
                watered_frame, cam_frame_copy, frame_hold, num_of_pixels, audio_stream, mutex))
            sound_thread.start()
            frame_hold = cam_frame_copy.copy()
            copy_frame = True

        # Get new image
        ret, cam_frame = cap.read()
        if copy_frame:
            cam_frame_copy = imutils.resize(cam_frame, height=FRAME_COMP_HEIGHT)
            copy_frame = False


def filter_to_water(frame, screen_size):
    """
    Unique image filtration to make it look like digital water
    :param frame: An image - 2d numpy matrix
    :param screen_size: screen shape tuple
    :return: The filtered frame as a numpy 2d matrix
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Turns image to grayscale
    frame = ndimage.gaussian_filter(frame, GAUSSIAN_SIGMA)  # Gaussian filter
    frame = ndimage.sobel(frame)  # Image derivative
    frame = cv2.resize(frame, screen_size)  # Resize to screen size
    return frame


def find_diff(frame_a, frame_b):
    """
    Does an image comparison
    :param frame_a: a numpy 2d matrix
    :param frame_b: a numpy 2d matrix (must be same size as frame_a)
    :return: A heatmap of the difference between the images and the total number of pixels in the heatmap
    """
    # Gaussian to remove noise
    frame_b = ndimage.gaussian_filter(frame_b, REMOVE_NOISE_SIGMA)
    frame_a = ndimage.gaussian_filter(frame_a, REMOVE_NOISE_SIGMA)
    diff = frame_b.copy()
    cv2.absdiff(frame_b, frame_a, diff)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # increasing the size of differences
    for i in range(0, 3):
        dilated = cv2.dilate(diff_gray.copy(), None, iterations=i + 1)
    # Threshold to make the heatmap
    (T, heatmap) = cv2.threshold(dilated, 3, BYTE_MAX, cv2.THRESH_BINARY)
    heatmap = (heatmap / BYTE_MAX).astype(np.int8)
    return heatmap, (heatmap.shape[0] * heatmap.shape[1])


def set_fullscreen():
    """
    Sets output window to full-screen
    """
    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def init_audio_stream(pyaudio_inst):
    """
    Initializes audio stream
    :param pyaudio_inst: pyaudio instance
    :return: audio stream
    """
    try:
        audio_stream = pyaudio_inst.open(format=pyaudio.paFloat32,
                                         channels=NUM_OF_CHANNELS,
                                         rate=SAMPLING_RATE,
                                         output=True)
    except OSError:
        return ERROR_RETURN
    return audio_stream


def get_screen_size():
    """
    :return: Screen shape tuple
    """
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return screensize


def close_program(cap, pyaudio_inst):
    """
    Closes program and terminates properly
    :param cap: web camera cv2 stream
    :param pyaudio_inst: pyaudio instance
    """
    pyaudio_inst.terminate()
    cap.release()
    cv2.destroyAllWindows()


def message_box_gui(title, text, style):
    """
    Message box interface
    :param title: Message box title
    :param text: Main text of the message box
    :param style: An integer
    """
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)


def error_shutdown(cap, pyaudio_inst, error_message):
    """
    Shutdowns program when error occurs. Shows error message.
    :param cap: web camera cv2 stream
    :param pyaudio_inst: pyaudio instance
    :param error_message: A string error message
    :return:
    """
    message_box_gui(ERROR, error_message, MBOX_STYLE)
    close_program(cap, pyaudio_inst)
    exit(-1)


def init_video_stream():
    """
    Initializes web camera stream.
    :return: web camera cv2 stream
    """
    cap = cv2.VideoCapture(1)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return ERROR_RETURN
    return cap


def main():
    """
    Initializes all dependencies, starts main loop and terminates
    """
    # initialize video and audio stream and other variables
    pyaudio_inst = pyaudio.PyAudio()
    cap = init_video_stream()
    if cap == ERROR_RETURN:
        error_shutdown(cap, pyaudio_inst, CANNOT_OPEN_WEBCAM_ERROR)
    audio_stream = init_audio_stream(pyaudio_inst)
    if audio_stream == ERROR_RETURN:
        error_shutdown(cap, pyaudio_inst, NO_AUDIO_DEVICE_FOUND_ERROR)
    set_fullscreen()
    # Starts main loop
    water_stream_loop(audio_stream, cap)
    # Terminate successfully
    audio_stream.stop_stream()
    audio_stream.close()
    close_program(cap, pyaudio_inst)
    exit(0)


if __name__ == '__main__':
    main()
