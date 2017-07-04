import cv2
import time

from threading import Thread
from Queue import Queue

class VideoStream(object):
	"""
		Desc:	A faster video streaming class which uses Python's threading module 
				for creating and maintaining threads.
	"""
	def __init__	(self, path, queue_size = 128):
		self.stream = cv2.VideoCapture(path)
		self.exit = False

		self.queue = Queue(maxsize=queue_size)

	def start(self):
		thread = Thread(name="Video Reader", target=self.__read_frames, args=())
		thread.daemon = True
		thread.start()
		time.sleep(1.0) # So that the thread can get enough time for building up initial queue.
		return

	def __read_frames(self):
		"""
		Desc:	This function is dedicated to reading frames and pushing them into the Queue. 
				It will run in a separate daemon thread.
		"""
		while True:
			if self.exit:
				self.stream.release()
				return
	
			if not self.queue.full():
				success, frame = self.stream.read()

				if not success:
					self.stream.release()
					self.stop()					
					return

				self.queue.put(frame)

	def read(self):
		return self.queue.get()

	def is_empty(self):
		return self.queue.qsize() == 0

	def stop(self):
		self.exit = True



			 