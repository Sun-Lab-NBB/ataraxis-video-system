from multiprocessing import Process, Queue


# Define a function that will run in a separate process
def process1(q):
    # Put a string into the queue
    q.put("hello")


# Define a function that will run in a separate process
def process2(q):
    # Get a message from the queue and print it to the console
    print(q.get())


# The following code will only run if the script is run directly
if __name__ == "__main__":
    # Create an instance of the Queue class
    q = Queue()

    # Create two instances of the Process class, one for each function
    p1 = Process(target=process1, args=(q,))
    p2 = Process(target=process2, args=(q,))

    # Start both processes
    p1.start()
    p2.start()

    # Wait for both processes to finish
    p1.join()
    p2.join()
