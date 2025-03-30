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
        "quantization_bit": "模型量化位数，可减少内存使用和加速推理。'none'表示不量化；'8'表示8位量化；'4'表示4位量化，大幅降低内存需求但可能影响生成质量。",
        "quantization_method": "选择量化算法。'bitsandbytes'：通用量化方案；'hqq'：高质量量化；'eetq'：高效嵌入式量化，适合边缘设备。",
        "template": "构建提示词时使用的模板格式。不同模型支持不同的对话模板，如'llama2'、'vicuna'、'alpaca'等。正确选择与模型匹配的模板至关重要。",
        "rope_scaling": "旋转位置编码(RoPE)的缩放方法，用于扩展模型的上下文窗口长度。'none'：不使用扩展；'linear'：线性插值；'dynamic'：动态插值。",
        "booster": "用于提升训练和推理速度的加速方法。'auto'：自动选择；'flashattn2'：Flash Attention 2加速；'unsloth'：使用Unsloth库加速。",

        # 训练组件
        "training_stage": "选择训练阶段和方法。'pt'：预训练；'sft'：监督微调；'rm'：奖励模型训练；'dpo'：直接偏好优化；'ppo'：近端策略优化。",
        "dataset_dir": "训练数据所在的文件夹路径。可以是相对路径或绝对路径。该文件夹应包含预处理好的数据集文件，通常是JSON或JSONL格式。",
        "dataset": "选择用于训练的数据集。可以选择内置数据集或自定义数据集。多个数据集可以同时选择，系统会自动合并。",
        "learning_rate": "AdamW优化器的初始学习率，控制模型参数更新的步长大小。对于LoRA微调，推荐范围：2e-5至2e-4；对于全参数微调，建议使用较小值。",
        "num_train_epochs": "需要执行的训练总轮数（epoch）。一个epoch表示模型遍历整个训练数据集一次。训练轮数过少可能导致欠拟合，过多则可能过拟合。",
        "max_grad_norm": "梯度裁剪的阈值，用于防止梯度爆炸。当梯度的L2范数超过此值时，会按比例缩小。默认值1.0适用于大多数情况。",
        "cutoff_len": "输入序列的最大token数。超过此长度的序列会被截断。对于大多数LLM，建议值为1024-2048。过大的值会显著增加内存消耗。",
        "batch_size": "每个GPU处理的样本数量。较大的批处理大小可提高训练速度和稳定性，但需要更多GPU内存。",
        "gradient_accumulation_steps": "在执行优化器更新前累积梯度的步数。通过增加此值，可在内存受限的情况下模拟更大的批处理大小。",
        "val_size": "从训练数据中划分出作为验证集的比例。验证集用于评估模型性能和防止过拟合。推荐值通常为0.05-0.1(5%-10%)。",
        "lr_scheduler_type": "学习率调度器类型，控制学习率在训练过程中的变化方式。'cosine'：余弦衰减，最为常用；'linear'：线性衰减；'constant'：保持恒定。",
        "logging_steps": "每两次日志输出间的更新步数。设置较小的值可以更频繁地看到训练进度，但可能略微影响训练速度。",
        "save_steps": "保存检查点的步数间隔。较小的值保存更频繁，提供更多恢复点但占用更多磁盘空间。建议值：100-500。",
        "warmup_steps": "学习率预热的步数。在这些步骤中，学习率从很小的值逐渐增加到设定值。对于不稳定的训练，设置为总步数的5%-10%可能有帮助。",
        "neftune_alpha": "NEFTune技术中嵌入向量所添加的噪声大小。设置为0表示不使用NEFTune；推荐范围为0.1-1.0。可以提高模型的泛化能力。",
        "packing": "启用数据打包，允许在同一序列中包含多个训练样本，提高训练效率。适合固定长度的小样本，但可能影响样本边界的上下文理解。",
        "neat_packing": "使用改进的数据打包策略，尝试在保证样本边界清晰的情况下实现高效打包。比普通打包有更好的边界处理。",
        "train_on_prompt": "在训练过程中同时对提示词和回答进行训练。启用此选项可能帮助模型更好地理解指令，但也可能导致混淆输入和输出的边界。",
        "mask_history": "在训练过程中屏蔽历史对话内容，只关注当前回合。适用于希望模型更专注于当前问题而不过度依赖对话历史的情况。",
        "resize_vocab": "根据训练数据动态调整词汇表大小。对于包含特殊词汇或非英语语言的数据集可能有帮助，但可能增加训练时间和内存使用。",
        "report_to": "选择训练指标的报告工具。'wandb'：Weight & Biases，提供丰富的可视化；'tensorboard'：轻量级本地可视化；'none'：不使用外部报告工具。",

        # LoRA相关
        "lora_rank": "LoRA适配器的秩，控制LoRA层的参数量和表达能力。较小的值（如8-16）适合大多数任务；较大的值（如32-64）可能提供更好的性能。",
        "lora_alpha": "LoRA缩放因子，通常设置为LoRA秩的2倍。控制LoRA更新对原始权重的影响程度。增大此值会增强LoRA的效果。",
        "lora_dropout": "LoRA层的Dropout率，用于防止过拟合。对于小数据集，推荐设置为0.05-0.1；对于大数据集或需要完全拟合的场景，可设为0。",
        "lora_target": "LoRA将应用的目标模块名称，使用英文逗号分隔。常用值：'q_proj,v_proj'（只对查询和值投影应用LoRA）或'all'（所有线性层）。",
        
        # 评估组件
        "max_new_tokens": "生成回答的最大token数量。设置合适的值避免生成过长或过短的回答。简短任务可设置较小值；需要详细解释的任务可设置较大值。", 
        "top_p": "核采样参数，控制生成的多样性。较小的值使生成更确定和保守；较大的值使生成更多样和创造性。测试时推荐使用0.7作为平衡值。",
        "temperature": "温度参数，控制生成的随机性。较小的值使生成更确定性；较大的值使生成更多样化。评估特定能力时建议使用较低温度。",
        "predict": "是否生成预测结果。启用时将使用模型生成响应并计算评估指标；禁用时仅计算损失等基本指标而不生成文本，评估速度更快。",
        "output_dir": "保存评估或训练结果的目录路径。结果将包含详细的指标和生成的文本（如果启用）。留空则使用默认路径。",

        # RLHF参数设置
        "beta": "Beta参数控制KL散度的权重，平衡模型探索与利用已学习行为的程度。较高的值(如0.1-0.5)使模型更保守，较低的值允许更多探索。一般从0.1开始调整。",
        "fix_gamma": "修正因子gamma，用于稳定RLHF训练。较高的值可以减少训练波动，但可能减慢收敛速度。通常设置为0-10之间的值，根据训练稳定性调整。",
        "损失类型": "选择RLHF中使用的损失函数类型。'sigmoid'适合一般场景；'hinge'对错误更敏感；'log'适合概率输出。推荐初学者使用sigmoid。",
        "奖励模型": "选择用于评估生成回答质量的奖励模型类型。不同模型对不同方面（如准确性、有用性、无害性）的权重不同。选择与训练目标一致的模型。",
        "归一化分数": "启用后将奖励分数标准化，有助于稳定训练并减少极端奖励值的影响。推荐在奖励分布不均匀时启用。",
        "白化奖励": "对奖励进行白化处理，降低奖励间的相关性，提高训练稳定性。适合奖励分布有偏或噪声较大的情况。",
        
        # GaLore参数设置
        "使用galore": "启用GaLore（梯度低秩投影）优化方法，通过低秩投影减少内存占用，加速大模型训练。适合内存受限的环境。",
        "galore秩": "GaLore投影空间的维度，控制梯度压缩率。较小的值(如8-16)节省更多内存但可能降低精度；较大的值(如32-64)保留更多信息。根据GPU内存和模型大小调整。",
        "更新间隔": "GaLore重新计算投影子空间的步数间隔。较小的值更新更频繁但计算开销更大；较大的值(如200-500)更节省计算资源。",
        "galore缩放系数": "控制GaLore投影梯度的缩放比例。推荐值为2-8，较大的值增强更新强度，但可能导致不稳定；较小的值更稳定但收敛可能较慢。",
        "galore作用模块": "指定GaLore应用的网络层。'all'表示应用于所有适用层；也可选择特定层组合，如仅应用于注意力层或前馈层。",
        
        # APOLLO参数设置
        "使用apollo": "启用APOLLO优化器，它结合了自适应学习率和低秩近似，比GaLore更适合某些任务。对梯度噪声较大的场景特别有效。",
        "apollo秩": "APOLLO投影空间的维度。通常设置为16-64，较高的值(如32-64)提供更好的近似但消耗更多内存。复杂任务通常需要更高的秩。",
        "apollo缩放系数": "控制APOLLO更新强度的参数。推荐值为16-64，较大的值可能加速收敛但增加不稳定风险；较小的值更稳定但收敛较慢。",
        "apollo作用模块": "选择APOLLO优化器应用的网络模块。'all'表示应用于所有适合的模块；也可以限制在特定模块上以平衡性能和资源使用。",
        
        # BAdam参数设置
        "使用badam": "启用Block-wise Adam优化器，通过分块处理参数减少内存使用并提高训练效率。适合参数量大且优化难度高的模型训练。",
        "badam模式": "选择BAdam的工作模式。'layer'按层分块，适合常规训练；'parameter'按参数分块，更精细但计算开销更大；'mixed'在两者间平衡。",
        "切换策略": "决定BAdam何时在不同块之间切换的策略。'ascending'从浅层到深层；'descending'从深层到浅层；'random'随机切换。影响模型不同部分的学习进度。",
        "切换频率": "BAdam在不同参数块之间切换的步数。较小的值(如10-50)使不同块更均匀地更新；较大的值(如100-500)减少切换开销但可能导致更新不平衡。",
        "block更新比例": "每次迭代更新的参数块比例。值范围为0-1，较小的值(如0.05-0.2)节省内存但延长收敛时间；较大的值更新更多参数但需要更多内存。",
        
        # SwanLab参数设置
        "使用swanlab": "启用SwanLab实验跟踪工具，记录训练过程中的指标、参数和结果。适合需要系统化管理和可视化实验结果的场景。",
        "swanlab项目名": "在SwanLab中创建的项目名称，用于组织相关实验。建议使用简洁且描述性的名称，如'llama2-sft'或'mistral-rlhf'。",
        "swanlab实验名": "当前实验的名称(非必填)。建议包含关键参数信息，如'lora-r16-sft'，方便后续比较不同实验。",
        "swanlab工作区": "SwanLab存储实验数据的工作区路径(非必填)。默认使用本地路径，也可设置为共享位置以便团队协作。",
        "swanlab_api密钥": "用于SwanLab云端同步的API密钥(非必填)。启用云端功能时需要提供，可从SwanLab账户设置获取。",
        "swanlab模式": "选择SwanLab的运行模式。'cloud'将数据同步到云端，便于远程访问；'local'仅在本地存储，适合处理敏感数据或无需远程访问的场景。"
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