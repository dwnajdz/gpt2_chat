import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class GPT2:
	model: any
	tokenizer: any

	def __init__(self,	model_name='gpt2'):
		if torch.cuda.is_available():
			print('using cuda')
			torch.set_default_device('cuda')

		self.model = AutoModelForCausalLM.from_pretrained(model_name)
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)

	def answer_prompt(self, prompt: str, max_length=500, temperature=0.9) -> str:
		input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
		gen_tokens = self.model.generate(
			input_ids,
			do_sample=True,
			temperature=temperature,
			max_length=max_length,
		)
		gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
		return gen_text

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

app = FastAPI()

@app.get('/')
def index():
	return FileResponse('./index.html')

@app.get('/ask/{prompt}')
def ask_gpt(prompt: str):
	llm = GPT2(model_name='gpt2') 
	return {'answer': llm.answer_prompt(prompt)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
	await websocket.accept()

	llm = GPT2(model_name='gpt2')
	while True:
		prompt = await websocket.receive_text()
		answer = llm.answer_prompt(prompt)

		print('\n---------------------ANSWER------------------------:\n')
		print(answer)
		print('\n---------------------END OF ANSWER------------------\n')
		await websocket.send_text(f'GPT2: {answer}')
	return
