#!/usr/bin/python

from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import caffe2.python.onnx.backend as backend
import numpy as np
import onnx


PORT_NUMBER = 8080
start = time.time()

# Load the ONNX model
model = onnx.load("model.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

rep = backend.prepare(model, device="CPU")  # or "CUDA:0"
end = time.time()
print("Loading time: {0:f} secs)".format(end - start))


class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Handler for GET requests"""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
        # To run networks with more than one input, pass a tuple
        # rather than a single numpy ndarray.
        self.wfile.write(bytes(str(outputs[0]), "utf8"))


try:
    server = HTTPServer(('', PORT_NUMBER), MyHandler)
    print('Started httpserver on port', PORT_NUMBER)
    server.serve_forever()

except KeyboardInterrupt:
    server.server_close()
    print('Stopping server')
