import asyncio
import json

import numpy as np

from server import routes


class _Avatar:
    def __init__(self):
        self.frames = []

    def put_audio_frame(self, frame, datainfo):
        self.frames.append((frame.copy(), datainfo))


class _Request:
    def __init__(self, body):
        self.query = {"sessionid": "test-session", "sample_rate": "16000"}
        self.app = {}
        self.body = body

    async def read(self):
        return self.body


def test_humanpcm_buffers_partial_frames(monkeypatch):
    avatar = _Avatar()
    monkeypatch.setattr(routes, "get_session", lambda request, sessionid: avatar)
    request = _Request(b"\x00\x40" * 250)

    first = asyncio.run(routes.humanpcm(request))
    request.body = b"\x00\xc0" * 390
    second = asyncio.run(routes.humanpcm(request))

    assert json.loads(first.text)["data"] == {"frames": 0, "buffered_bytes": 500}
    assert json.loads(second.text)["data"] == {"frames": 2, "buffered_bytes": 0}
    assert len(avatar.frames) == 2
    assert all(frame.shape == (320,) and frame.dtype == np.float32 for frame, _ in avatar.frames)
    assert avatar.frames[0][0][0] == 0.5
