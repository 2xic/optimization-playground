"""
The idea is this, sometimes we have larger dataset which are annoying to send over to the cloud. 
We run the data portal on a VPS and the then Cloud GPU just connects to that over zmq.

This server expects the files to already been pre-processed and the dataloader only serves them.
"""
import zmq
import argparse
import glob
import queue
import threading
import time 
import json

DATASET_SIZE = 1_000

def fetch_files(files_queue: queue.Queue):
    global DATASET_SIZE
    actual_size = 0
    predicted_dataset_size = 1_000
    is_first_round = True
    DATASET_SIZE = min(predicted_dataset_size, args.limit) if args.limit != None else predicted_dataset_size
    last_printed = time.time()
    files = glob.iglob(args.path, recursive=True)
    is_glob_mode = True
    unique_files = set()
    while True:
        next_item = None
        try:
            next_item = next(files)
            if not is_glob_mode:
                unique_files.add(next_item)
        except StopIteration as e:
            print("Stop iterator ", e)
            # Disable glob and use the unique files!
            is_glob_mode = False
            files = unique_files
            next_item = next(files)
            is_first_round = False
            DATASET_SIZE = min(actual_size, args.limit) if args.limit != None else predicted_dataset_size
        if (time.time() - last_printed) > 30:
            print(actual_size)
            last_printed = time.time()
        # print(next_item)
        with open(next_item, "rb") as file:
            files_queue.put(file.read())
        # So we can compare it later on.
        if is_first_round:
            actual_size += 1

def serve_files(files_queue: queue.Queue):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind("tcp://*:5555")
    print("Serving files now ")
    lookup = {}
    while True:
        try:
            request = socket.recv_multipart()
            [clientId, payload] = request
           # print(clientId.hex())
           # print(payload)
            payload = json.loads(payload)
            command = payload["command"]
            arguments = payload.get("arguments", {})
            if command == "get":
                index = arguments["index"]
                if index in lookup:
                    socket.send_multipart((clientId, lookup[index]))
                else:
                    next_item = files_queue.get()
                    lookup[index] = next_item
                    socket.send_multipart((clientId, next_item))
            elif command == "size":
                # Do I even want to do this ? I could just return a huge item while we figure it out
                socket.send_multipart((clientId, str(DATASET_SIZE).encode()))
            else:
                print("Unknown command")
        except Exception as e:
            print(e)

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
