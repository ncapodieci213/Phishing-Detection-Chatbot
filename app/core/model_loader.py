# Load or cache ML & LLM models
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import json
import os

print("Initializing phishing detection system...")

def device_check():
	if torch.cuda.is_available():
		print(f" CUDA available: {torch.cuda.get_device_name(0)}")
		print(f" CUDA version: {torch.version.cuda}")
		total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
		print(f"  GPU Memory: {total_vram:.2f} GB")
		device = "cuda"
	else:
		print(" CUDA not available, using CPU (slower)")
		device = "cpu"
	
	return device

def load_model():
	# Check GPU availability
	device = device_check()

	model_name = "Qwen/Qwen2.5-0.5B-Instruct"
	
	print("\nLoading tokenizer...")
	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

	print("Loading model...")

	try:
		if device == "cuda":
			# Attempt to load model in FP16 first
			print("Attempting to load model in FP16 on GPU...")

			# Clear cache first
			torch.cuda.empty_cache()

			model = AutoModelForCausalLM.from_pretrained(
				model_name,
				dtype=torch.float16,
				device_map="auto",  # Automatically handles GPU/CPU split if GPU VRAM is not enough
				trust_remote_code=True,
				low_cpu_mem_usage=True,
				max_memory={0: "5GB", "cpu": "8GB"}  # Reserve 5GB for GPU, rest for CPU
			)

			print(" Model loaded successfully on GPU")
			print(f" Device map: {model.hf_device_map}")

			# Check VRAM usage
			allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
			reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
			print(f"  VRAM allocated: {allocated:.2f} GB")
			print(f"  VRAM reserved: {reserved:.2f} GB")

		else:
			# CPU loading
			print("Loading model on CPU...")
			model = AutoModelForCausalLM.from_pretrained(
				model_name,
				dtype=torch.float32,
				trust_remote_code=True,
				low_cpu_mem_usage=True
			)
			model = model.to(device)
			print(" Model loaded on CPU")

	except Exception as e:
		print(f" Error loading model: {e}")
		print("\nAttempting fallback: CPU offloading...")

		try:
			# Fallback: More CPU offloading
			model = AutoModelForCausalLM.from_pretrained(
				model_name,
				dtype=torch.float16 if device == "cuda" else torch.float32,
				device_map="auto",
				trust_remote_code=True,
				low_cpu_mem_usage=True,
				max_memory={0: "4GB", "cpu": "12GB"} if device == "cuda" else None
			)
			print(" Model loaded with CPU offloading")
		except Exception as e2:
			print(f" Fallback failed: {e2}")
			raise
	
	return model, tokenizer, device