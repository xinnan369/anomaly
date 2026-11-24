"""Reject Python network connections and use only already checked-out Git objects."""
import os
import socket
import sys


def enable_offline():
    def guard(event, args):
        if event == "socket.connect" and args[0].family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("TASK-005 prohibits network access")
        if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
            raise RuntimeError("TASK-005 prohibits DNS lookups")

    sys.addaudithook(guard)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", GIT_NO_LAZY_FETCH="1")
