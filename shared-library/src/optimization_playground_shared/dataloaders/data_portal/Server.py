"""
The idea is this, sometimes we have larger dataset which are annoying to send over to the cloud. 
We run the data portal on a VPS and the then Cloud GPU just connects to that over zmq.

This server expects the files to already been pre-processed and the dataloader only serves them.
"""
import zmq
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import threading

DATASET_SIZE = 1_000


def fetch_files(files_queue: queue.Queue):
    global DATASET_SIZE
    actual_size = 0
    predicted_dataset_size = 1_000
    is_first_round = True
    DATASET_SIZE = min(predicted_dataset_size, args.limit) if args.limit != None else predicted_dataset_size
    while True:
        files = glob.iglob(args.path, recursive=True)
        next_item = None
        try:
            next_item = next(files)
        except StopIteration as e:
            print("Stop iterator ", e)
            files = glob.iglob(args.path, recursive=True)                
            next_item = next(files)
            is_first_round = False
            DATASET_SIZE = min(actual_size, args.limit) if args.limit != None else predicted_dataset_size
        
     #   print(next_item)
        with open(next_item, "rb") as file:
            files_queue.put(file.read())
        # So we can compare it later on.
        if is_first_round:
            actual_size += 1

def serve_files(files_queue: queue.Queue):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("Serving files now ")
    while True:
        message = socket.recv()
        if message == b"get":
            next_item = files_queue.get()
            socket.send(next_item)
        elif message == b"size":
            # Do I even want to do this ? I could just return a huge item while we figure it out
            socket.send(str(DATASET_SIZE).encode())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Glob file format of the files you want to return", type=str, required=True)
    parser.add_argument("-l", "--limit", help="Limit the dataset size", type=int, required=False)
    
    args = parser.parse_args()
    files_queue = queue.Queue(maxsize=100)
    thread = threading.Thread(target=fetch_files, args=(files_queue,  ))
    thread.start()

    thread = threading.Thread(target=serve_files, args=(files_queue, ))
    thread.start()
    thread.join()
