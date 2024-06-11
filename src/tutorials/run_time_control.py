from high_precision_timer.precision_timer import PrecisionTimer

d = {"ns": 10**9, "us": 10**6, "ms": 10**3, "s": 1}

unit = "ms"

def run_time_control(func, fps, verbose=False):
    run_timer = PrecisionTimer(unit)
    check_timer = PrecisionTimer(unit)

    frames = 0

    while True:
        if not fps or run_timer.elapsed / d[unit] >= 1 / fps:
            func()
            frames += 1
            run_timer.reset()
        if verbose and check_timer.elapsed > d[unit]:
            print("fps:", frames)
            check_timer.reset()
            frames = 0