import os
import cv2
import magic
import shutil
import asyncio
import aiofiles
import numpy as np
import subprocess as sp
import concurrent.futures as parallel

from functools import reduce
from typing import TypeVar, List, Any, Callable

T = TypeVar('T')

class Reader:
	
	def __init__(self):
		self.task:str = None
  
	def __call__(self, paths: List[str], batch_size: int = 32) -> Any:
		self.isMediaFile(paths)
		batches = [paths[i:i+batch_size] for i in range(0, len(paths), batch_size)]
		call_tasks = self.task_map(self.task)
		outs = call_tasks(batches)
		return outs
	
	def task_map(self, task: str) -> Callable:
		return {
			"image"	: self.process_image,
			"video"	: self.process_video,
			}[task]

	def isMediaFile(self, paths):
		
		if not isinstance(paths, list):
			raise TypeError("paths must be list")
		
		mime = magic.Magic(mime=True)
		mediafile_set = set(map(lambda x: mime.from_file(x).split('/')[0], paths))
		if len(mediafile_set) > 1:
			raise TypeError("All elements of the list must be images or videos")
		
		self.task = list(mediafile_set)[0]
  
	def pipe(self, data: T, *functions: Callable[[Any], Any]) -> Any:
		return reduce(lambda x, f: f(x), functions, data)

	def process_image(self, paths: List[List[str]]) -> List[List[np.ndarray]]:
		processed_batches = []
		for batch_paths in paths:
			processed_batch = asyncio.run(self._process_image(batch_paths))
			processed_batches.append(processed_batch)
		return processed_batches
		#yield

	async def _process_image(self, batch_paths: List[str]) -> List[np.ndarray]:
		tasks = list(map(lambda x: self._read_async(x), batch_paths))
		images = await asyncio.gather(*tasks)
		return images

	async def _read_async(self, path:str) -> np.ndarray:
		async with aiofiles.open(path, "rb") as file:
			content = await file.read()
			image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
			return image

	def process_video(self):
		return
	
		

		

		