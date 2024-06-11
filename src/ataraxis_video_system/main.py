from ataraxis_video_system import VideoSystem, Camera

if __name__ == "__main__" :
    vs = VideoSystem('imgs', Camera())
    vs.start(True)
    input() # Need this line because subprocesses are daemon, as soon as you leave the __main__ scope the processes stop

    

    
    










