import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

import json
from transformers import AutoTokenizer, AutoConfig
from models.configuration_qwen3_dsa import Qwen3DSAConfig
from models.modeling_qwen3_dsa import Qwen3DSAForCausalLM

AutoConfig.register("qwen3_dsa", Qwen3DSAConfig)

MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.do.mirr.one",
]

def try_with_mirrors(fn, model_name):
    last_error = None
    for mirror in MIRRORS:
        os.environ["HF_ENDPOINT"] = mirror
        print(f"  Trying mirror: {mirror}")
        try:
            result = fn(model_name)
            print(f"  Success with {mirror}")
            return result
        except Exception as e:
            print(f"  Failed with {mirror}: {e}")
            last_error = e
    raise last_error

if __name__ == "__main__":
    model_name = "kaizheng9105/Qwen3-4B-Instruct-2507-DSA"

    print("Loading config...")
    config_dict, _ = try_with_mirrors(Qwen3DSAConfig.get_config_dict, model_name)
    config_dict["model_type"] = "qwen3_dsa"
    config = Qwen3DSAConfig(**config_dict)

    print("Loading tokenizer...")
    tokenizer = try_with_mirrors(
        lambda mn: AutoTokenizer.from_pretrained(mn, trust_remote_code=True),
        model_name,
    )

    print("Loading model...")
    model = try_with_mirrors(
        lambda mn: Qwen3DSAForCausalLM.from_pretrained(
            mn, config=config, torch_dtype="auto", device_map="auto", trust_remote_code=True
        ),
        model_name,
    )
    print("Model loaded successfully!")

    prompt = "Give me a short introduction to sparse attention."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("content:", content)
