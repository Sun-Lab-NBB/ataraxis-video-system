# First, import the timer class.
from high_precision_timer.precision_timer import PrecisionTimer
import time as tm

timer = PrecisionTimer('s')
timer.reset()
tm.sleep(3)
print(timer.elapsed)

