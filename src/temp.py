from pynput import keyboard

def on_press(key):
    if key == keyboard.Key.esc:
        return False  # stop listener
    
    try:
        if key.char == 'q':
            print("The 'q' key was pressed.")
        elif key.char == 'w':
            print("The 'w' key was pressed.")
    except AttributeError:
        pass

        

listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread
listener.join()  # remove if main thread is polling self.keys  