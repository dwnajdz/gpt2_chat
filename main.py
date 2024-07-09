from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from llm_model import GPT2

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
