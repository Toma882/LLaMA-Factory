#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
这个脚本用于增强LLaMA-Factory中的中文参数说明。
使用方法：
1. 将此脚本放在LLaMA-Factory项目根目录
2. 运行：python enhance_chinese_descriptions.py
3. 脚本会备份原始locales.py文件并创建增强版本
"""

import os
import re
import shutil
from pathlib import Path

# 定义增强后的中文参数说明
ENHANCED_DESCRIPTIONS = {
    "model_name": {
        "label": "模型名称",
        "info": "输入模型名称以检索相应模型。支持从Hugging Face库中选择预训练模型，如'llama2'、'mistral'、'qwen'等。输入关键字后系统会自动搜索匹配的模型。例如：输入'llama2'可找到所有LLaMA-2系列模型；输入'mistral'可找到Mistral系列模型。"
    },
    "model_path": {
        "label": "模型路径",
        "info": "预训练模型的本地路径或Hugging Face模型标识符。本地路径格式如：'/path/to/model'或'C:\\path\\to\\model'；Hugging Face模型标识符格式如：'meta-llama/Llama-2-7b'。若使用Hugging Face模型，请确保您已登录并有访问权限。推荐使用绝对路径以避免路径解析错误。"
    },
    "finetuning_type": {
        "label": "微调方法",
        "info": "选择模型微调的方法。可选项包括：'lora'（参数高效微调，推荐用于资源受限环境）、'qlora'（量化版LoRA，进一步降低内存需求）、'full'（全参数微调，需要较多计算资源但效果可能更好）等。对于大多数用户，建议从'lora'开始尝试，在效果不满意时再考虑其他方法。"
    },
    "checkpoint_path": {
        "label": "检查点路径",
        "info": "微调模型保存的检查点路径。可以选择已有的检查点继续训练，或指定新路径保存未来的检查点。多个检查点可以组合使用（如合并多个LoRA权重）。格式示例：'checkpoints/my-llama-lora'或绝对路径。注意：若继续训练，请确保选择与当前模型和微调方法兼容的检查点。"
    },
    "quantization_bit": {
        "label": "量化等级",
        "info": "模型量化位数，可减少内存使用和加速推理。'none'表示不量化；'8'表示8位量化，平衡性能和质量；'4'表示4位量化，大幅降低内存需求但可能影响生成质量。推荐：对于>7B参数的大模型，首选尝试8位量化；对于资源极其有限的环境，可考虑4位量化。注意：量化会轻微影响模型性能，但显著减少内存占用。"
    },
    "quantization_method": {
        "label": "量化方法",
        "info": "选择量化算法。'bitsandbytes'：通用量化方案，适用于大多数情况；'hqq'：高质量量化，追求更高精度；'eetq'：高效嵌入式量化，适合边缘设备。不同算法在速度和精度上有所权衡。推荐：首先尝试'bitsandbytes'，若对模型性能有更高要求，可尝试'hqq'；在资源受限设备上可考虑'eetq'。需注意安装相应的库支持。"
    },
    "template": {
        "label": "对话模板",
        "info": "构建提示词时使用的模板格式。不同模型支持不同的对话模板，如'llama2'、'vicuna'、'alpaca'等。正确选择与模型匹配的模板至关重要，否则可能导致生成质量下降。例如：LLaMA-2模型应选择'llama2'模板；Vicuna模型应选择'vicuna'模板。如果不确定，可选择'default'让系统自动检测，或查阅模型文档确认推荐模板。自定义模板可通过修改配置文件实现。"
    },
    "rope_scaling": {
        "label": "RoPE 插值方法",
        "info": "旋转位置编码(RoPE)的缩放方法，用于扩展模型的上下文窗口长度。'none'：不使用扩展；'linear'：线性插值，适合中等长度扩展；'dynamic'：动态插值，根据输入自适应调整；'yarn'：Yet Another RoPE Extension，提供更好的长文本理解能力；'llama3'：专为LLaMA-3模型优化的方法。推荐：对于处理超出原始训练长度的长文本，选择'yarn'可能效果最佳；对于LLaMA-3模型应选择'llama3'。"
    },
    "booster": {
        "label": "加速方式",
        "info": "用于提升训练和推理速度的加速方法。'auto'：自动选择最适合的方法；'flashattn2'：Flash Attention 2加速注意力计算；'unsloth'：使用Unsloth库加速训练；'liger_kernel'：Liger内核优化，针对特定硬件优化。不同加速方法对硬件要求不同，且可能需要安装额外依赖。推荐：首选'auto'让系统自动选择；针对注意力计算密集的场景，可尝试'flashattn2'；针对LoRA训练，'unsloth'可能提供最佳加速效果。"
    },
    "training_stage": {
        "label": "训练阶段",
        "info": "选择训练阶段和方法。'pt'：预训练，从头训练模型；'sft'：监督微调，使用高质量数据进行指令跟随训练；'rm'：奖励模型训练，用于偏好学习；'dpo'：直接偏好优化，直接基于人类偏好优化模型；'ppo'：近端策略优化，基于强化学习方法优化模型。对于大多数用户，建议从'sft'开始，根据需求再尝试高级阶段。每个阶段需要不同格式的数据和训练策略。"
    },
    "dataset_dir": {
        "label": "数据路径",
        "info": "训练数据所在的文件夹路径。可以是相对路径（如'data/'）或绝对路径。该文件夹应包含预处理好的数据集文件，通常是JSON或JSONL格式。对于自定义数据集，建议参考项目文档组织数据结构。正确的数据格式对训练至关重要，不同训练阶段（SFT、RM、DPO等）要求不同的数据格式。"
    },
    "dataset": {
        "label": "数据集",
        "info": "选择用于训练的数据集。可以选择内置数据集或自定义数据集。多个数据集可以同时选择，系统会自动合并。对于指令微调(SFT)，推荐使用高质量的指令数据；对于奖励模型(RM)或DPO，需要包含偏好信息的数据。数据集质量直接影响模型性能，建议仔细筛选和清洗训练数据。"
    },
    "learning_rate": {
        "label": "学习率",
        "info": "AdamW优化器的初始学习率，控制模型参数更新的步长大小。数值过大可能导致训练不稳定，过小则可能导致训练缓慢或陷入局部最优。对于LoRA微调，推荐范围：2e-5至2e-4；对于全参数微调，建议使用较小值如5e-6至1e-5。特别建议：初次训练时可从适中值开始（如1e-4），根据训练曲线调整；对于更精细的调整任务，可降低至5e-5或更低。"
    },
    "num_train_epochs": {
        "label": "训练轮数",
        "info": "需要执行的训练总轮数（epoch）。一个epoch表示模型遍历整个训练数据集一次。训练轮数过少可能导致模型欠拟合，过多则可能导致过拟合。对于大规模预训练数据，通常3-5轮足够；对于特定任务微调，可能需要5-10轮。建议使用早停策略（验证损失不再下降时停止）避免过拟合。对于高质量但数量较少的数据集，可适当增加轮数。"
    },
    "max_grad_norm": {
        "label": "最大梯度范数",
        "info": "梯度裁剪的阈值，用于防止梯度爆炸。当梯度的L2范数超过此值时，会按比例缩小。默认值1.0适用于大多数情况。如果训练中出现'nan'值或训练不稳定，可尝试降低此值（如0.5）；如果训练进展缓慢，可适当增大（如1.5-2.0）。梯度裁剪是稳定训练的重要技术，特别是在全参数微调或处理复杂任务时。"
    },
    "compute_dtype": {
        "label": "计算精度",
        "info": "混合精度训练使用的数据类型。'fp16'：半精度浮点，可显著减少内存使用并加速训练，但可能在某些情况下导致数值不稳定；'bf16'：脑浮点格式，介于fp16和fp32之间，提供更好的数值稳定性；'fp32'：全精度浮点，最稳定但最慢且内存消耗最大。推荐：首选尝试'bf16'（如果硬件支持），其次是'fp16'。A100/H100等新一代GPU对bf16有硬件加速。"
    },
    "cutoff_len": {
        "label": "截断长度",
        "info": "输入序列的最大token数。超过此长度的序列会被截断，过长的序列会增加内存使用和计算时间。此值应根据模型的上下文窗口大小和任务需求设置。对于大多数LLM，建议值为1024-2048。对于处理长文本的任务，可适当增加（如4096），但需确保硬件内存足够。注意：过大的截断长度会显著增加内存消耗，过小则可能导致模型无法捕获完整上下文。"
    },
    "batch_size": {
        "label": "批处理大小",
        "info": "每个GPU处理的样本数量。较大的批处理大小可提高训练速度和稳定性，但需要更多GPU内存。对于消费级GPU（如RTX 3090、4090），LoRA微调推荐值：4-8；使用量化（QLoRA）时可尝试增加至8-16；专业级GPU可相应提高。当出现CUDA内存不足错误时，应首先减小此值。注意：实际批大小 = 批处理大小 × 梯度累积步数，可通过增加梯度累积步数来模拟更大批次。"
    },
    "gradient_accumulation_steps": {
        "label": "梯度累积",
        "info": "在执行优化器更新前累积梯度的步数。通过增加此值，可在内存受限的情况下模拟更大的批处理大小。有效批大小 = 批处理大小 × 梯度累积步数。适用场景：当需要更大的批大小但GPU内存不足时；当训练不稳定需要更大批量时。推荐：对于稳定性要求高的任务，可设置为4-8；对于内存受限的情况，可设置为8-16。注意增加此值会相应增加训练时间。"
    },
    "val_size": {
        "label": "验证集比例",
        "info": "从训练数据中划分出作为验证集的比例。验证集用于评估模型性能和防止过拟合。推荐值通常为0.05-0.1(5%-10%)，对于大型数据集，可设置较小值（如0.02）；对于小型数据集，可适当增加（如0.15）。设置为0表示不使用验证集。建议除非数据集非常小，否则始终使用验证集以监控训练过程和检测过拟合。验证集的质量和代表性应与训练集相当。"
    },
    "logging_steps": {
        "label": "日志间隔",
        "info": "每两次日志输出间的更新步数。较小的值（如5-10）可以更频繁地看到训练进度，但可能略微影响训练速度；较大的值（如50-100）适合长时间训练，减少日志输出频率。日志包含当前损失、学习率等关键信息，对监控训练过程非常重要。推荐：对于短期实验，可设置较小值以密切监控；对于长期训练，可设置较大值以减少日志量。"
    },
    "save_steps": {
        "label": "保存间隔",
        "info": "每隔多少步保存一次检查点。较小的值提供更多恢复点但占用更多磁盘空间；较大的值减少IO操作和存储需求。建议值：对于不稳定环境或重要训练，可设置为较小值（如100-500）；对于稳定环境下的长期训练，可设置为较大值（如1000-5000）。检查点文件包含模型权重和优化器状态，可用于恢复训练或部署模型。"
    },
    "warmup_steps": {
        "label": "预热步数",
        "info": "学习率预热的步数。在这些步骤中，学习率从很小的值逐渐增加到设定值。对于不稳定的训练或大批次训练，设置为总步数的5%-10%可能有帮助；对于稳定训练，可设置为0（不使用预热）。预热有助于稳定初始训练阶段，避免学习率过大导致的发散。推荐：首次尝试特定任务时，从适当的预热步数开始（如100-500），根据训练表现进行调整。"
    },
    "neftune_alpha": {
        "label": "NEFTune 噪声参数",
        "info": "NEFTune技术中嵌入向量所添加的噪声大小。设置为0表示不使用NEFTune；推荐范围为0.1-1.0。NEFTune通过在嵌入层添加噪声来提高模型的泛化能力，特别适合小规模数据集训练。较大的值（如0.5-1.0）增加多样性但可能降低稳定性；较小的值（如0.1-0.3）提供轻微改进同时保持稳定。建议根据任务复杂度和数据集大小进行调整。"
    },
    "max_new_tokens": {
        "label": "最大生成长度",
        "info": "模型生成的最大token数。应根据任务需求设置合适的值：对于简短回答或分类任务，64-256通常足够；对于中等长度解释，256-512较为合适；对于长篇生成如文章或代码，可设置为1024-2048或更高。注意：较大的值会增加生成时间和资源消耗，但过小的值可能导致生成的回答不完整。生成长度还应考虑模型的上下文窗口大小限制。"
    },
    "top_p": {
        "label": "核采样概率",
        "info": "核采样（Nucleus Sampling）参数，控制生成文本的多样性。原理是仅从累积概率达到该值的词汇中采样。较小的值（如0.3-0.5）使生成更确定和保守，适合需要准确性的任务；较大的值（如0.7-0.95）使生成更多样和创造性，适合开放性任务。推荐：对于事实性问答设置较低值；对于创意写作设置较高值；对于一般对话，0.7是较好的平衡点。"
    },
    "temperature": {
        "label": "温度系数",
        "info": "控制生成文本随机性的温度参数。降低温度会使模型更倾向于选择高概率词，生成更确定性的文本；提高温度会使词的选择更均匀，生成更多样化的文本。常用值范围：0.1-0.5（高确定性，适合事实回答）；0.7-0.9（均衡性，适合一般对话）；1.0-1.2（高创造性，适合创意内容）。温度与top_p配合使用时，建议在调整一个参数时保持另一个参数在中等值。"
    },
    "output_dir": {
        "label": "输出目录",
        "info": "模型训练结果和检查点的保存路径。可以是相对路径或绝对路径。该目录将存储模型权重、训练日志、评估结果等。建议为每次训练使用独特的目录名，如包含模型名称、日期或实验编号的组合。若目录不存在会自动创建；若目录已存在且包含先前训练的检查点，则可能继续从该检查点训练（取决于其他设置）。确保路径有足够的磁盘空间。"
    },
    "packing": {
        "label": "序列打包",
        "info": "是否将多个短样本打包到一个训练序列中。启用此选项可提高训练效率，特别是当数据集包含大量短样本时。打包减少了填充所需的计算资源，但可能在样本边界处导致上下文混淆。推荐：对于平均长度远小于截断长度的数据集，启用此选项可显著提高训练速度；对于需要严格保持样本完整性的任务，应禁用此选项。通常与neat_packing结合使用效果更佳。"
    },
    "neat_packing": {
        "label": "优化打包",
        "info": "使用改进的数据打包策略，在保证样本边界清晰的同时实现高效打包。与普通packing相比，neat_packing添加了额外的边界标记，帮助模型识别不同样本的边界。这种方法可以减轻样本混淆问题，同时保持打包带来的计算效率。推荐：当同时需要效率和样本边界清晰度时，应启用此选项；通常作为普通packing的替代选择，而不是同时使用两者。"
    },
    "train_on_prompt": {
        "label": "训练提示部分",
        "info": "是否在训练中同时对提示词部分进行训练。启用时，模型会学习理解并生成提示词的模式；禁用时，模型只学习生成回答部分。启用此选项可能帮助模型更好地理解指令和上下文，但也可能导致模型在推理时混淆用户输入和预期输出的边界。推荐：对于理解复杂指令的任务可能有帮助；对于严格的指令跟随任务，建议禁用以避免模型生成提示词。"
    },
    "mask_history": {
        "label": "屏蔽历史对话",
        "info": "是否在训练中屏蔽历史对话内容。启用此选项时，模型只关注当前回合的提问和回答，而不考虑之前的对话历史。适用于训练单轮问答能力，或当您希望模型对每个问题独立响应而不过度依赖上下文时。禁用此选项则允许模型学习使用完整对话历史进行回答，对多轮对话场景更为适合。对于通用助手类模型，通常应禁用此选项以保持对话连贯性。"
    },
    "lora_rank": {
        "label": "LoRA 秩",
        "info": "LoRA适配器的秩参数，决定了LoRA矩阵的维度和模型的参数量。较小的秩（如4-8）产生更紧凑的模型，训练更快但表达能力较弱；较大的秩（如16-64）提供更强的表达能力但需要更多计算资源。推荐：对于简单任务或资源有限情况，使用8-16；对于复杂任务且有足够资源时，可使用32-64。增加秩通常会提高模型性能，但收益呈现边际递减，超过64的值通常效果提升有限。"
    },
    "lora_alpha": {
        "label": "LoRA 缩放因子",
        "info": "控制LoRA更新对原始权重的影响程度。通常设置为LoRA秩的2倍（经验法则）。增大alpha会增强LoRA的效果，但值过大可能导致训练不稳定。较大的alpha/rank比值会放大LoRA的影响，可以在使用较小秩的情况下获得更强的调优效果。推荐：通常保持alpha=2*rank的关系；对于需要强调特定特征的任务，可以适当增大比值至3-4倍；对于稳定性要求高的任务，可保持默认的2倍关系或略低。"
    }
}

def backup_file(file_path):
    """备份原始文件"""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"已备份原始文件到: {backup_path}")

def enhance_chinese_descriptions(file_path):
    """增强中文参数描述"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 创建正则表达式模式来匹配要替换的中文部分
    for param_name, param_data in ENHANCED_DESCRIPTIONS.items():
        pattern = rf'"{param_name}":\s*{{[^{{}}]*"zh":\s*{{'
        if pattern in content:
            # 查找原来的中文部分
            zh_pattern = rf'"{param_name}":\s*{{[^{{}}]*"zh":\s*{{[^}}]*}}'
            zh_match = re.search(zh_pattern, content, re.DOTALL)
            if zh_match:
                old_zh = zh_match.group(0)
                
                # 创建新的中文部分
                new_zh = f'"{param_name}": {{'
                new_zh += old_zh.split('"zh": {')[0] + '"zh": {'
                new_zh += f'\n            "label": "{param_data["label"]}",\n'
                new_zh += f'            "info": "{param_data["info"]}",\n'
                new_zh += '        }'
                
                # 替换内容
                content = content.replace(old_zh, new_zh)
    
    # 写入增强后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已增强 {len(ENHANCED_DESCRIPTIONS)} 个中文参数描述")

def modify_component_files(component_files):
    """修改组件文件，添加info参数"""
    # 加载已有的中文参数描述
    description_map = {k.lower(): v for k, v in ENHANCED_DESCRIPTIONS.items()}
    
    for file_path in component_files:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            continue
            
        # 备份原始文件
        backup_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有Gradio组件定义
        modified_content = content
        
        # 查找以gr.开头的组件定义
        component_pattern = r'(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*gr\.(Dropdown|Textbox|Slider|Checkbox)(\([^)]*\))'
        matches = re.finditer(component_pattern, content, re.DOTALL)
        
        for match in matches:
            indent, var_name, component_type, params = match.groups()
            var_name_lower = var_name.lower()
            
            # 检查是否已经有info参数
            if 'info=' in params:
                continue
                
            # 查找匹配的描述
            desc = None
            if var_name_lower in description_map:
                desc = description_map[var_name_lower]
            
            # 如果找到匹配的描述，添加info参数
            if desc:
                # 确保参数结束前添加info
                new_params = params[:-1] + f', info="{desc["info"]}"' + params[-1:]
                new_component = f'{indent}{var_name} = gr.{component_type}{new_params}'
                modified_content = modified_content.replace(match.group(0), new_component)
        
        # 写入修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"已修改 {file_path} 添加info参数")

def main():
    """主函数"""
    # 定义要处理的文件路径
    component_files = [
        "src/llamafactory/webui/components/top.py",
        "src/llamafactory/webui/components/train.py",
        "src/llamafactory/webui/components/eval.py",
        # 可以添加更多组件文件
    ]
    
    locales_path = "src/llamafactory/webui/locales.py"
    
    # 检查文件是否存在
    if not os.path.exists(locales_path):
        print(f"错误: 文件 {locales_path} 不存在")
        return
    
    # 备份原始文件
    backup_file(locales_path)
    
    # 增强中文参数描述
    enhance_chinese_descriptions(locales_path)
    
    # 修改组件文件添加info
    modify_component_files(component_files)
    
    print("完成! 请重新启动LLaMA-Factory以应用更改。")
    print("如果需要恢复原始文件，请使用'.bak'后缀的备份文件。")

if __name__ == "__main__":
    main()