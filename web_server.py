#!/usr/bin/env python3

# import the libraries
import os
from bottle import route, post, static_file, run, request
from test import *


@post("/upload")
def submit_image():
    # do some cleanup
    if os.path.isfile("upload.jpg"):
        os.remove("upload.jpg")

    # get the file
    upload = request.files.get('fileToUpload')
    name, extension = os.path.splitext(upload.filename)
    
    # save the file
    upload_filename = "upload" + extension
    upload.save(upload_filename)

    # lets do detection
    do_processing(upload_filename)

    return static_file("output.jpg", root="./")


@route("/")
def home_page():
    return static_file("home.html", root="html")


# run the server
run(host="0.0.0.0", port=9000, debug=True)
