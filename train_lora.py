from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

model_name = "Qwen/Qwen1.5-0.5B"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.config.use_cache = False  # 防止报错

# ✅ 设置 pad_token，防止 loss 报错
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# 加载中文数据集（JSON 格式，每行为 {"instruction": "...", "output": "..."}）
dataset = load_dataset("json", data_files="train_zh.json")

# 编码数据
def tokenize(example):
    prompt = f"指令：{example['instruction']}\n回答：{example['output']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

# ✅ 自定义 Trainer（手动计算 loss）
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./qwen_lora_output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    gradient_accumulation_steps=2,
    fp16=True  # 如果你支持 CUDA 和 FP16，否则改为 False
)

# 启动训练
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=default_data_collator
)

trainer.train()
