from multiprocessing import Queue

q = Queue()

print('q empty:', q.empty())

q.put(1)

print('q empty:', q.empty())
