import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

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

def start():
	while True:
		try:
			prompt = input('Your prompt: ')
			if type(prompt) != str:
				raise Error('wrong type')
			answer_prompt(prompt)
		except:
			print('Some error happened')
			break
