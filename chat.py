import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your model directory
MODEL_PATH = "./tiny_transformer_model"  # Update this path to your model directory

# Load tokenizer and model
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Model loaded on {device}")

conversation_history = ""

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    conversation_history += f"User: {user_input}\nAssistant: "

    inputs = tokenizer(
        conversation_history,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    response = full_response[len(conversation_history):].strip()

    print(f"Bot: {response}")

    conversation_history += response + "\n"