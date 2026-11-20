"""Batched observation uploads independent of policy action caching."""

import base64
import io
import json
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np
from PIL import Image
import requests


class EpisodeUpload:
    """Upload all sampled frames in bounded batches using a dedicated session."""

    def __init__(self, base_url, metadata, interval=2., batch_bytes=64 * 1024**2,
                 max_bytes=512 * 1024**2):
        self.base_url = base_url
        self.episode_id = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:12]
        self.session = requests.Session()
        self.interval, self.batch_bytes, self.max_bytes = interval, batch_bytes, max_bytes
        self.pending_bytes = 0
        self.lock = threading.Lock()
        self.error = None
        self.frames = queue.Queue()
        self.last_frame_id = -1
        self.closed = False
        self._post('start', dict(metadata, episode_id=self.episode_id))
        self.worker = threading.Thread(target=self._run, daemon=True, name='episode-upload')
        self.worker.start()

    def _post(self, operation, payload):
        for attempt in range(3):
            try:
                response = self.session.post(f'{self.base_url}/recording/{operation}',
                    json=payload, timeout=(5, 120))
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(0.5)

    def check(self):
        """Surface recording failures on the robot loop before another command."""
        if self.error is not None:
            raise RuntimeError(f'Episode recording failed: {self.error}') from self.error

    def add(self, frame):
        """Own a copy of one complete control-cycle observation and command."""
        self.check()
        if self.closed:
            raise RuntimeError('Recording already stopped')
        frame = {key: value.copy() if isinstance(value, np.ndarray) else value
                 for key, value in frame.items()}
        size = sum(value.nbytes for value in frame.values() if isinstance(value, np.ndarray)) + 1024
        with self.lock:
            if self.pending_bytes + size > self.max_bytes:
                self.error = RuntimeError('Episode upload queue full; current frame was not enqueued')
                raise self.error
            self.pending_bytes += size
        self.last_frame_id = frame['frame_id']
        self.frames.put((frame, size))

    @staticmethod
    def _encode(frame):
        encoded = {}
        for key, value in frame.items():
            if key in ('image', 'gripper'):
                buffer = io.BytesIO()
                # Match StepRequestFromEvaluator's default PIL JPEG encoding.
                Image.fromarray(value).save(buffer, format='JPEG')
                encoded[key] = base64.b64encode(buffer.getvalue()).decode('ascii')
            else:
                encoded[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return encoded

    def _run(self):
        batch, size, deadline = [], 0, time.monotonic() + self.interval
        try:
            while True:
                try:
                    item = self.frames.get(timeout=max(0., deadline - time.monotonic()))
                except queue.Empty:
                    item = 'flush'
                closing = item is None
                if item != 'flush' and not closing:
                    frame, frame_size = item
                    batch.append(self._encode(frame))
                    size += frame_size
                if closing or size >= self.batch_bytes or time.monotonic() >= deadline:
                    if batch:
                        self._post('frames', dict(episode_id=self.episode_id, frames=batch))
                        with self.lock:
                            self.pending_bytes -= size
                        batch, size = [], 0
                    deadline = time.monotonic() + self.interval
                if closing:
                    return
        except Exception as exc:
            self.error = exc

    def drain(self):
        """Finish uploads before collecting optional operator annotations."""
        if not self.closed:
            self.closed = True
            self.frames.put(None)
        self.worker.join()
        self.check()

    def finish(self, stop, result='unknown', note='', reason='completed'):
        """Finalize remotely, retaining an end receipt for retry after network failure."""
        recording_error = ''
        try:
            self.drain()
        except Exception as exc:
            recording_error = str(exc)
        payload = dict(episode_id=self.episode_id, final_frame_id=self.last_frame_id,
            result=result, note=note, reason=reason, recording_error=recording_error, **stop)
        receipt = Path(f'{self.episode_id}_end.json')
        receipt.write_text(json.dumps(payload, indent=2))
        response = self._post('end', payload)
        receipt.unlink()
        self.session.close()
        return response
