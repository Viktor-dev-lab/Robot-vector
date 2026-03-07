import queue
import threading
import json

class MQTTWorker:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None

    def start(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        print("[System] MQTT Worker thread started.")

    def stop(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)

    def publish(self, topic, payload):
        message = {
            "topic": topic,
            "payload": payload
        }
        self.message_queue.put(message)

    def _process_queue(self):
        while self.is_running:
            try:
                msg = self.message_queue.get(timeout=1)
                topic = msg["topic"]
                payload = msg["payload"]
                
                payload_str = json.dumps(payload) if isinstance(payload, dict) else str(payload)
                
                print(f"[Queue -> MQTT] Topic: {topic} | Payload: {payload_str}")
                # TODO: client.publish(topic, payload_str)
                
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Error] MQTT Worker: {e}")