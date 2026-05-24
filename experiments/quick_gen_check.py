import torch
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt

def main():
    ckpt_path = "checkpoints/dn_baseline_30L_217M_for_oracle.pt"
    device = "cuda"
    
    print(f"Loading model: {ckpt_path}")
    model, _ = build_model_from_ckpt(ckpt_path)
    model = model.to(device)
    model.eval()
    
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    prompts = [
        "def fibonacci(n):",
        "import torch\nimport torch.nn as nn\n\nclass",
        "def factorial(n):",
        "# A function that adds two numbers"
    ]
    
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt} ---")
        input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        
        # Simple greedy generation
        generated = input_ids
        for _ in range(50):
            with torch.no_grad():
                logits = model(generated)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
            if next_token.item() == tok.eos_token_id:
                break
                
        print(tok.decode(generated[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
