import argparse
from datasets import load_dataset
from quick.awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def basic_quantization(model_path, quant_path, quant_config):
    # Load model
    # NOTE: pass safetensors=True to load safetensors
    model = AutoAWQForCausalLM.from_pretrained(
        model_path#, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def load_wikitext():    
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
        return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext())
    
    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='', type=str, help='Path to hf model')
    parser.add_argument('--quant_path', default='./', type=str, help='Path to save quantized AWQ model file')
    parser.add_argument('--version', default='QUICK', type=str, help='Kernel Version: QUICK, GEMM, or GEMV')
    args = parser.parse_args()
    
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": args.version}

    basic_quantization(model_path=args.model_path, quant_path=args.quant_path, quant_config=quant_config)
