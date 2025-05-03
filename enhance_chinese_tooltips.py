#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
这个脚本用于为LLaMA-Factory的Web界面组件添加中文info提示。
使用方法：
1. 将此脚本放在LLaMA-Factory项目根目录
2. 运行：python enhance_chinese_infos.py
3. 脚本会备份原始组件文件并创建增强版本
"""

import os
import re
import shutil
from pathlib import Path

def backup_file(file_path):
    """备份原始文件"""
    backup_path = f"{file_path}.bak"
    # 如果备份已存在，不再重复备份
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"已备份原始文件到: {backup_path}")
    else:
        print(f"备份文件已存在: {backup_path}")

def add_infos_to_components(file_path, info_map):
    """为组件添加info属性"""
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return False
        
    # 备份原始文件
    backup_file(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有Gradio组件定义
        modified_content = content
        
        # 查找以gr.开头的组件定义
        # 匹配模式: 变量名 = gr.组件类型(参数)
        component_pattern = r'(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*gr\.(Dropdown|Textbox|Slider|Checkbox|Button)(\([^)]*\))'
        matches = list(re.finditer(component_pattern, content, re.DOTALL))
        
        # 从后向前替换，避免位置偏移
        for match in reversed(matches):
            indent, var_name, component_type, params = match.groups()
            var_name_lower = var_name.lower()
            
            # 检查是否已经有info参数
            if 'info=' in params:
                continue
                
            # 查找匹配的提示信息
            info_text = None
            for key, info in info_map.items():
                if key == var_name_lower or key in var_name_lower:
                    info_text = info
                    break
            
            # 如果找到匹配的提示信息，添加info参数
            if info_text:
                # 确保参数结束前添加info
                new_params = params[:-1] + f', info="{info_text}"' + params[-1:]
                new_component = f'{indent}{var_name} = gr.{component_type}{new_params}'
                modified_content = modified_content[:match.start()] + new_component + modified_content[match.end():]
        
        # 如果内容有变化，写入修改后的内容
        if modified_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"已为 {file_path} 添加info参数")
            return True
        else:
            print(f"文件 {file_path} 没有需要修改的组件")
            return False
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False

def main():
    """主函数"""
    # 定义要处理的文件路径
    component_files = [
        "src/llamafactory/webui/components/top.py",
        "src/llamafactory/webui/components/train.py",
        "src/llamafactory/webui/components/eval.py",
        "src/llamafactory/webui/components/infer.py",
        "src/llamafactory/webui/components/export.py",
        "src/llamafactory/webui/components/chatbot.py",
    ]
    
    # 定义组件的提示信息映射
    info_map = {
        # 顶部组件
        "lang": "选择您希望使用的语言。",
        "model_name": "输入模型名称以检索相应模型。支持从Hugging Face库中选择预训练模型，如'llama2'、'mistral'、'qwen'等。",
        "model_path": "预训练模型的本地路径或Hugging Face模型标识符。本地路径格式如：'/path/to/model'；Hugging Face模型标识符格式如：'meta-llama/Llama-2-7b'。",
        "finetuning_type": "选择模型微调的方法。可选项包括：'lora'（参数高效微调，推荐用于资源受限环境）、'qlora'（量化版LoRA）、'full'（全参数微调）等。",
        "checkpoint_path": "微调模型保存的检查点路径。可以选择已有的检查点继续训练，或指定新路径保存未来的检查点。多个检查点可以组合使用。",
        "quantization_bit": "模型量化位数，可大幅减少内存使用和加速推理。'none'表示不量化；'8'表示8位量化，可减少约50%内存使用；'4'表示4位量化，可减少约75%内存使用但可能略微影响生成质量；'3'/'2'表示更极端的压缩比例，适合资源极度受限的环境。",
        "quantization_method": "选择量化算法。'bitsandbytes'：通用量化方案，平衡速度和精度；'hqq'：高质量量化，在保持精度的同时实现高压缩率；'eetq'：高效嵌入式量化，适合边缘设备和移动设备；'gptq'：针对生成模型优化的静态量化方法，提供较好的速度-精度平衡。",
        "template": "构建提示词时使用的模板格式。不同模型支持不同的对话模板，如'llama2'、'vicuna'、'alpaca'等。正确选择与模型匹配的模板至关重要。",
        "rope_scaling": "旋转位置编码(RoPE)的缩放方法，用于扩展模型的上下文窗口长度。'none'：不使用扩展；'linear'：线性插值；'dynamic'：动态插值。",
        "booster": "用于提升训练和推理速度的加速方法。'auto'：自动选择；'flashattn2'：Flash Attention 2加速；'unsloth'：使用Unsloth库加速。",

        # 训练组件
        "training_stage": "选择训练阶段和方法。'pt'：预训练，从头训练模型；'sft'：监督微调，使用高质量数据进行指令跟随训练；'rm'：奖励模型训练，用于偏好学习；'dpo'：直接偏好优化，直接基于人类偏好优化模型；'ppo'：近端策略优化，基于强化学习方法优化模型。",
        "dataset_dir": "训练数据所在的文件夹路径。可以是相对路径（如'data/'）或绝对路径。该文件夹应包含预处理好的数据集文件，通常是JSON或JSONL格式。",
        "dataset": "选择用于训练的数据集。可以选择内置数据集或自定义数据集。多个数据集可以同时选择，系统会自动合并。",
        "learning_rate": "AdamW优化器的初始学习率，控制模型参数更新的步长大小。对于LoRA微调，推荐范围：2e-5至2e-4；对于全参数微调，建议使用较小值如5e-6至1e-5。",
        "num_train_epochs": "需要执行的训练总轮数（epoch）。一个epoch表示模型遍历整个训练数据集一次。训练轮数过少可能导致模型欠拟合，过多则可能导致过拟合。",
        "max_grad_norm": "梯度裁剪的阈值，用于防止梯度爆炸。当梯度的L2范数超过此值时，会按比例缩小。默认值1.0适用于大多数情况。",
        "max_samples": "训练过程中使用的最大样本数量。设置为较大值（如100000）表示使用全部数据集；设置为较小值可用于快速验证训练流程。",
        "compute_type": "混合精度训练使用的数据类型。'bf16'：脑浮点格式，介于fp16和fp32之间，提供良好的数值稳定性和内存效率；'fp16'：半精度浮点，可减少约50%的显存使用但可能出现溢出问题；'fp32'：全精度浮点，最稳定但显存占用最高。推荐：首选'bf16'（如果硬件支持），其次是带有动态缩放的'fp16'。",
        "cutoff_len": "输入序列的最大token数。超过此长度的序列会被截断。较短的序列长度可显著降低显存使用，对于大多数LLM，建议值为1024-2048。长度每翻倍，显存使用量通常增加1.5-2倍。对于处理长文本的任务，可适当增加（如4096），但需确保硬件内存足够。",
        "batch_size": "每个GPU处理的样本数量。较大的批处理大小可提高训练速度和稳定性，但线性增加显存使用。对于消费级GPU（如RTX 3090，24GB），LoRA微调推荐值：4-8；使用量化（QLoRA）时可尝试增加至8-16；对于较小显存GPU（如RTX 3060，12GB），建议使用1-2并配合梯度累积。",
        "gradient_accumulation_steps": "在执行优化器更新前累积梯度的步数。通过增加此值，可在内存受限的情况下模拟更大的批处理大小而不增加显存使用。有效批大小 = 批处理大小 × 梯度累积步数。对于16GB显存GPU，推荐批大小1-2，累积步数8-16来达到有效的训练效果。",
        "val_size": "从训练数据中划分出作为验证集的比例。验证集用于评估模型性能和防止过拟合。推荐值通常为0.05-0.1(5%-10%)，设置为0表示不使用验证集。",
        "lr_scheduler_type": "学习率调度器类型，控制学习率在训练过程中的变化方式。'cosine'：余弦衰减，逐渐降低学习率，最为常用；'linear'：线性衰减；'constant'：保持恒定学习率；'constant_with_warmup'：先预热再保持恒定。",
        "logging_steps": "每两次日志输出间的更新步数。设置较小的值（如5）可以更频繁地看到训练进度，但可能略微影响训练速度。对于大型数据集，可以设置较大的值（如50-100）。",
        "save_steps": "保存检查点的步数间隔。较小的值保存更频繁，提供更多恢复点但占用更多磁盘空间；较大的值减少磁盘IO和存储需求。建议值：100-500。",
        "warmup_steps": "学习率预热的步数。在这些步骤中，学习率从很小的值逐渐增加到设定值。对于不稳定的训练，设置为总步数的5%-10%可能有帮助。设置为0表示不使用预热。",
        "neftune_alpha": "NEFTune技术中嵌入向量所添加的噪声大小。设置为0表示不使用NEFTune；推荐范围为0.1-1.0。NEFTune可以提高模型的泛化能力，尤其适合小规模数据集训练。",
        "extra_args": "以JSON格式传递给训练器的额外参数。常用设置包括优化器选择{'optim': 'adamw_torch'}、SGD动量设置{'momentum': 0.9}等。高级用户可根据需要添加Transformers库支持的任何训练参数。",
        "packing": "启用数据打包，允许在同一序列中包含多个训练样本，提高训练效率。适合固定长度的小样本，但可能影响样本边界的上下文理解。",
        "neat_packing": "使用改进的数据打包策略，尝试在保证样本边界清晰的情况下实现高效打包。比普通打包有更好的边界处理，但会增加一定计算开销。",
        "train_on_prompt": "在训练过程中同时对提示词和回答进行训练。启用此选项可能帮助模型更好地理解指令，但也可能导致模型在生成时混淆用户输入和预期输出的边界。",
        "mask_history": "在训练过程中屏蔽历史对话内容，只关注当前回合。适用于希望模型更专注于当前问题而不过度依赖对话历史的情况。",
        "resize_vocab": "根据训练数据动态调整词汇表大小。对于包含特殊词汇或非英语语言的数据集可能有帮助，但可能增加训练时间和内存使用。",
        "use_llama_pro": "使用LLaMA Pro优化策略，可能提高模型的训练效率和生成质量。这是一种实验性功能，可能不适用于所有模型架构。",
        "report_to": "选择训练指标的报告工具。'wandb'：Weight & Biases，提供丰富的可视化和实验跟踪；'tensorboard'：TensorBoard，轻量级本地可视化；'mlflow'/'neptune'：其他实验跟踪平台。选择'none'表示不使用外部报告工具。",

        # 冻结相关参数
        "freeze_trainable_layers": "可训练的层数。正值表示从末尾开始计算的层数；负值表示从开头计算的层数。例如：设置为2表示只训练最后2层；设置为-2表示只训练前2层。通常模型后面的层与生成能力关系更大。",
        "freeze_trainable_modules": "指定要训练的模块名称，使用英文逗号分隔。'all'表示所有模块；也可以指定具体模块如'q_proj,k_proj,v_proj'。可通过模型结构分析确定关键模块名称。",
        "freeze_extra_modules": "额外冻结的模块名称，使用英文逗号分隔。指定的模块将不参与训练。常用于冻结不需要微调的组件，如嵌入层(embed)或特定头部(head)。",

        # LoRA相关参数
        "lora_rank": "LoRA适配器的秩，控制LoRA层的参数量和表达能力。较小的值（如4-8）显存占用极低，适合显存受限环境；中等值（如16-32）平衡效果和资源消耗；较大的值（如64-128）可能提供更好的性能但显存占用较高。每翻倍rank值，额外参数量和显存占用大约增加一倍。",
        "lora_alpha": "LoRA缩放因子，通常设置为LoRA秩的2倍。控制LoRA更新对原始权重的影响程度。增大此值会增强LoRA的效果但不影响显存使用，但值过大可能导致训练不稳定。",
        "lora_dropout": "LoRA层的Dropout率，用于防止过拟合。对于小数据集，推荐设置为0.05-0.1；对于大数据集或需要完全拟合的场景，可设为0。",
        "loraplus_lr_ratio": "LoRA Plus中学习率比例因子。设置为0表示不使用LoRA Plus；大于0时启用，建议值为1-8。LoRA Plus可能在某些任务上提供更好的性能，特别是长文本生成。",
        "create_new_adapter": "创建新的LoRA适配器而不是继续训练现有适配器。当您想要从头开始训练一个新的LoRA模型时选择此项，而不是在现有适配器的基础上继续训练。",
        "use_rslora": "使用Rank-stabilized LoRA (RSLoRA)，一种改进的LoRA变体，在训练过程中通过额外的正则化保持秩的稳定性。可能提高模型性能，特别是对于较高秩值或长时间训练。",
        "use_dora": "使用DoRA (Weight-Decomposed Low-Rank Adaptation)，可以提高LoRA的表达能力而不增加额外参数。通过分解原始权重矩阵，DoRA可能在某些任务上取得更好的效果。",
        "use_pissa": "使用PISSA (Part-of-Speech Informed Sensitivity Analysis)，一种关注语言模型中词性信息的微调方法。可能提高模型对语法和语义的理解能力。",
        "lora_target": "LoRA将应用的目标模块名称，使用英文逗号分隔。常用值：'q_proj,v_proj'（只对查询和值投影应用LoRA）或'all'（所有线性层）。不同模型架构的模块名可能不同。",
        "additional_target": "额外的LoRA目标模块，使用英文逗号分隔。可用于指定非标准模块应用LoRA，如'lm_head'（语言模型头部）或特定于模型架构的其他模块。",
        
        # RLHF参数设置
        "pref_beta": "Beta参数控制KL散度的权重，平衡模型探索与利用已学习行为的程度。较高的值(如0.1-0.5)使模型更保守，较低的值允许更多探索。一般从0.1开始调整，DPO/PPO训练中非常重要的超参数。",
        "pref_ftx": "FTX参数用于控制奖励函数的温度缩放。较高的值(如0.5-2.0)使奖励分布更平滑；较低的值(如0.1-0.3)使奖励差异更明显。对于有明确偏好的场景，可使用较低值；对于偏好差异较小的场景，推荐使用较高值。",
        "pref_loss": "偏好学习使用的损失函数类型。'sigmoid'：标准二元交叉熵，适合一般场景；'hinge'：边界损失，对错误更敏感；'ipo'：隐式偏好优化；'kto_pair'：KTO配对损失；'orpo'：在线强化偏好优化；'simpo'：简化偏好优化。对于初次尝试，推荐使用'sigmoid'。",
        "reward_model": "用于评估生成回答质量的奖励模型。可以选择预训练的奖励模型或自己训练的模型。对于偏好学习(DPO)不需要显式奖励模型；对于PPO训练，好的奖励模型对最终效果至关重要。",
        "ppo_score_norm": "启用奖励分数归一化，将奖励转换为标准分布(均值0，标准差1)。这有助于稳定PPO训练过程，减少异常奖励值的影响。对于奖励分布不均匀的场景特别有用，几乎所有PPO训练都推荐启用。",
        "ppo_whiten_rewards": "对奖励进行白化处理，降低奖励间的相关性，进一步提高训练稳定性。此选项在处理高度相关奖励或复杂任务时特别有用。与score_norm结合使用效果更佳。",
        
        # GaLore参数设置
        "use_galore": "启用GaLore(梯度低秩投影)优化方法，通过低秩投影减少显存占用并加速大模型训练。对于内存受限环境非常有用，可以在保持大部分性能的同时减少高达70-80%的显存需求。特别适合在消费级GPU(如RTX 3060/3070)上训练大型模型，可以用8-12GB显存训练原本需要20-24GB显存的模型。",
        "galore_rank": "GaLore投影空间的维度，控制梯度压缩率和保留信息量。较小的值(如4-8)可以减少高达90%显存使用但可能损失精度；中等值(如16-32)提供良好的平衡点，减少约70-80%显存使用；较大的值(如64-128)保留更多信息但显存节省较少。建议从16开始，根据训练效果和可用资源调整。",
        "galore_update_interval": "GaLore重新计算投影子空间的步数间隔。较小的值(如50-100)更新更频繁，通常有更好的性能但计算开销大；较大的值(如200-500)更节省计算资源但可能略微影响效果。通常设置为200左右较为平衡。",
        "galore_scale": "控制GaLore投影梯度的缩放比例。推荐值为1.0-5.0，较大的值增强更新强度，但可能导致训练不稳定；较小的值更保守稳定但收敛可能较慢。建议从2.0开始，根据训练损失曲线调整。",
        "galore_target": "指定GaLore应用的网络层。'all'表示应用于所有适用层；也可以选择特定层组合如'attention'(仅应用于注意力相关层)或'mlp'(仅应用于前馈层)。不同模型架构的层名称可能有所不同。",
        
        # APOLLO参数设置
        "use_apollo": "启用APOLLO优化器，结合了自适应学习率调整和低秩梯度近似，在某些任务上比GaLore有更好的性能。APOLLO可以减少约60-70%的显存使用，特别适合训练不稳定或梯度噪声较大的场景，能提供更稳定的收敛过程。适合在中等规格GPU(如RTX 3080/4070，10-16GB显存)上训练大型模型。",
        "apollo_rank": "APOLLO投影空间的维度。通常设置为16-64之间，较高的值(如32-64)提供更精确的梯度近似但消耗更多内存；较低的值(如8-16)更节省内存但近似精度较低。复杂任务或大型模型通常需要更高的秩。",
        "apollo_update_interval": "APOLLO重新计算投影子空间的步数间隔。类似于GaLore的相应参数，但APOLLO通常对更新间隔更敏感。推荐值为100-300，较小值通常能提供更好的性能，但计算开销更大。",
        "apollo_scale": "控制APOLLO更新强度的缩放因子。推荐值范围为16-64，显著高于GaLore。较大的值可能有助于加速收敛，但增加训练不稳定的风险；较小的值提供更稳定的训练过程。建议从32.0开始调整。",
        "apollo_target": "选择APOLLO优化器应用的网络模块。与GaLore类似，'all'表示应用于所有合适的模块；也可以限制在特定模块上以平衡性能和计算资源使用。APOLLO对不同模块的选择可能比GaLore更敏感。",
        
        # BAdam参数设置
        "use_badam": "启用Block-wise Adam优化器，通过分块处理参数减少内存使用并提高训练效率。BAdam可以减少约40-60%的优化器状态显存占用，特别适合参数量大且优化难度高的模型训练，能在有限内存条件下实现接近全参数更新的效果。与量化技术(如QLoRA)结合使用，可以在8GB显存GPU上训练10-20B参数的模型。",
        "badam_mode": "选择BAdam的工作模式。'layer'：按网络层分块处理，适合常规训练和大多数模型；'ratio'：按参数比例分块，更灵活但计算开销可能更大。一般情况下推荐使用'layer'模式。",
        "badam_switch_mode": "决定BAdam在不同块之间切换的策略。'ascending'：从浅层到深层顺序更新，适合渐进式训练；'descending'：从深层到浅层，适合细调输出层；'random'：随机选择，增加训练随机性；'fixed'：固定更新某些块。一般推荐使用'ascending'。",
        "badam_switch_interval": "BAdam在不同参数块之间切换的步数。较小的值(如10-50)使不同层更均匀地更新，但增加切换开销；较大的值(如100-200)减少切换开销但可能导致部分层更新不均衡。建议从50开始尝试。",
        "badam_update_ratio": "每次迭代更新的参数块比例。值范围为0-1，较小的值(如0.05-0.1)极大节省内存但延长收敛时间；较大的值(如0.3-0.5)更新更多参数但需要更多内存。典型值为0.05-0.1，视可用显存情况调整。",
        
        # SwanLab参数设置
        "use_swanlab": "启用SwanLab实验跟踪工具，记录训练过程中的指标、参数和结果。SwanLab提供直观的可视化界面，帮助跟踪和比较不同实验，适合需要系统化管理实验结果的场景。",
        "swanlab_project": "在SwanLab中创建的项目名称，用于组织相关实验。建议使用简洁且描述性的名称，如'llama2-sft'或'mistral-rlhf'，便于后续在仪表板中快速识别项目。",
        "swanlab_run_name": "当前实验的名称。建议包含关键参数信息，如'lora-r16-b8-ep3'（表示秩为16、批次为8、训练3轮的LoRA实验），这有助于在不查看详细配置的情况下比较不同实验。",
        "swanlab_workspace": "SwanLab存储实验数据的工作区路径。默认使用本地路径，也可设置为网络共享位置以便团队协作。指定绝对路径如'/path/to/workspace'或相对路径如'./swanlab_workspace'。",
        "swanlab_api_key": "用于SwanLab云端同步的API密钥。启用云端功能时需要提供，可从SwanLab账户设置获取。保持为空则仅使用本地功能，不进行云端同步。",
        "swanlab_mode": "选择SwanLab的运行模式。'cloud'：将实验数据同步到云端，便于远程访问和团队协作；'local'：仅在本地存储数据，适合处理敏感数据或无需远程访问的场景。云端模式需要有效的API密钥。",
        
        # DeepSpeed配置
        "ds_stage": "DeepSpeed优化阶段，控制内存优化和计算效率的平衡。'none'：不使用DeepSpeed；'1'：基础优化，节省约20%显存；'2'：使用ZeRO-2优化，平衡速度和内存，节省约50%显存；'3'：使用ZeRO-3优化，最大程度节省内存(高达70-80%)但可能略微影响速度。大模型训练建议使用'2'或'3'，单卡训练大模型或多卡训练超大模型时尤为有效。",
        "ds_offload": "启用DeepSpeed优化器状态卸载到CPU。这可以显著减少GPU内存使用(约30-40%)，但会增加额外的CPU-GPU数据传输开销(可能降低10-20%训练速度)。在GPU内存受限但CPU内存充足的情况下特别有用，通常与ZeRO-2或ZeRO-3一起使用。对于消费级PC，配合NVMe硬盘的Offload可以进一步扩展可训练的模型大小。",
        
        # 其他配置
        "config_path": "训练配置文件的路径。可以加载之前保存的配置以重复或修改已有实验。配置文件通常以YAML格式存储，包含所有训练参数设置。留空则使用当前界面中的设置。",
        "device_count": "可用于训练的设备(GPU)数量。系统自动检测，显示为只读值。多GPU训练会自动应用数据并行策略，加速训练过程。",
        
        # 评估组件
        "max_new_tokens": "生成回答的最大token数量。设置合适的值避免生成过长或过短的回答。简短任务可设置较小值(如128-256)；需要详细解释的任务可设置较大值(如512-2048)。", 
        "top_p": "核采样参数，控制生成的多样性。较小的值(如0.5)使生成更确定和保守；较大的值(如0.9)使生成更多样和创造性。测试时推荐使用0.7作为平衡值，实际应用可根据需要调整。",
        "temperature": "温度参数，控制生成的随机性。较小的值(如0.1-0.5)使生成更确定性；较大的值(如0.8-1.2)使生成更多样化。评估特定能力时建议使用较低温度(0.2-0.5)，创意生成可使用较高温度(0.7-1.0)。",
        "predict": "是否生成预测结果。启用时将使用模型生成响应并计算评估指标；禁用时仅计算损失等基本指标而不生成文本，评估速度更快。对于大规模评估，如只需计算准确率时，可禁用以提高效率。",
        "output_dir": "保存评估或训练结果的目录路径。结果将包含详细的指标和生成的文本（如果启用）。留空则使用默认路径。定期查看此目录下的结果有助于监控和分析模型性能。",
        
        # 额外补充的显存优化参数
        "flash_attention": "启用Flash Attention优化，显著减少自注意力机制的内存占用和计算复杂度。可减少约20-30%的显存使用，同时提高计算速度约30-50%。特别适合长上下文训练和大批量训练。对于支持的模型架构(如Llama、Mistral、Qwen等)，几乎总是应该启用此选项。",
        "gradient_checkpointing": "启用梯度检查点技术，通过在前向传播中放弃某些中间激活值并在反向传播时重新计算它们来节省内存。可减少约40-50%的激活值内存占用，但会增加约20-30%的计算量。对于大模型几乎是必选的内存优化技术，尤其是在消费级GPU上训练时。",
        "packed_adam": "使用优化的Adam优化器实现，通过紧凑存储优化器状态以减少内存使用。可节省约25-30%的优化器状态内存，几乎不影响训练精度和收敛速度。适合大规模模型训练，尤其是结合LoRA等技术时效果更佳。",
        "paged_adamw": "使用分页内存管理的AdamW优化器，允许优化器状态在GPU和CPU内存之间动态迁移。可以处理远超GPU显存容量的模型训练，但会增加一定的计算开销。与量化方法结合使用时，甚至可以在单个消费级GPU上训练超过30B参数的模型。",
        "max_memory": "手动设置每个GPU设备可用的最大显存。以'GPU设备ID:显存大小'格式指定，如'0:10GiB,1:10GiB'。对于多GPU系统特别有用，可以为系统保留部分显存，防止OOM错误。也可用于模拟低显存环境进行测试。"
    }
    
    # 检查并创建组件文件的info
    modified_count = 0
    for file_path in component_files:
        if add_infos_to_components(file_path, info_map):
            modified_count += 1
    
    print(f"\n完成! 已成功修改 {modified_count} 个文件。")
    print("请重新启动LLaMA-Factory以应用更改。")
    print("如果需要恢复原始文件，请使用'.bak'后缀的备份文件。")

if __name__ == "__main__":
    main() 