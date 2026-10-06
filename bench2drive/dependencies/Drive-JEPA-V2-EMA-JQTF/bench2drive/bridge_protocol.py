"""Size-limited protocol used only between trusted localhost processes."""

import pickle
import struct


PROTOCOL_VERSION = 1
_HEADER = struct.Struct("!I")
_MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def _recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("Drive-JEPA bridge connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock, message):
    """Send one pickle frame. Never expose this protocol outside loopback."""
    payload = pickle.dumps(message, protocol=4)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError("bridge message exceeds 32 MiB")
    sock.sendall(_HEADER.pack(len(payload)))
    sock.sendall(payload)


def receive_message(sock):
    """Receive one size-limited pickle frame."""
    (size,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if size <= 0 or size > _MAX_MESSAGE_BYTES:
        raise ValueError("invalid bridge message size: {}".format(size))
    return pickle.loads(_recv_exact(sock, size))

