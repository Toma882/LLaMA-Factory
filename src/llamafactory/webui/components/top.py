# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...data import TEMPLATES
from ...extras.constants import METHODS, SUPPORTED_MODELS
from ...extras.packages import is_gradio_available
from ..common import save_config
from ..control import can_quantize, can_quantize_to, get_model_info, list_checkpoints


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


def create_top() -> dict[str, "Component"]:
    with gr.Row():
        lang = gr.Dropdown(choices=["en", "ru", "zh", "ko", "ja"], value=None, scale=1, info="选择您希望使用的语言。")
        available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]
        model_name = gr.Dropdown(choices=available_models, value=None, scale=3, info="输入模型名称以检索相应模型。支持从Hugging Face库中选择预训练模型，如'llama2'、'mistral'、'qwen'等。输入关键字后系统会自动搜索匹配的模型。")
        model_path = gr.Textbox(scale=3, info="预训练模型的本地路径或Hugging Face模型标识符。本地路径格式如：'/path/to/model'或'C:\\path\\to\\model'；Hugging Face模型标识符格式如：'meta-llama/Llama-2-7b'。")

    with gr.Row():
        finetuning_type = gr.Dropdown(choices=METHODS, value="lora", scale=1, info="选择模型微调的方法。可选项包括：'lora'（参数高效微调，推荐用于资源受限环境）、'qlora'（量化版LoRA，进一步降低内存需求）、'full'（全参数微调，需要较多计算资源但效果可能更好）等。")
        checkpoint_path = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=6, info="微调模型保存的检查点路径。可以选择已有的检查点继续训练，或指定新路径保存未来的检查点。多个检查点可以组合使用（如合并多个LoRA权重）。")

    with gr.Row():
        quantization_bit = gr.Dropdown(choices=["none", "8", "4"], value="none", allow_custom_value=True, info="模型量化位数，可减少内存使用和加速推理。'none'表示不量化；'8'表示8位量化，平衡性能和质量；'4'表示4位量化，大幅降低内存需求但可能影响生成质量。")
        quantization_method = gr.Dropdown(choices=["bitsandbytes", "hqq", "eetq"], value="bitsandbytes", info="选择量化算法。'bitsandbytes'：通用量化方案，适用于大多数情况；'hqq'：高质量量化，追求更高精度；'eetq'：高效嵌入式量化，适合边缘设备。")
        template = gr.Dropdown(choices=list(TEMPLATES.keys()), value="default", info="构建提示词时使用的模板格式。不同模型支持不同的对话模板，如'llama2'、'vicuna'、'alpaca'等。正确选择与模型匹配的模板至关重要，否则可能导致生成质量下降。")
        rope_scaling = gr.Dropdown(choices=["none", "linear", "dynamic", "yarn", "llama3"], value="none", info="旋转位置编码(RoPE)的缩放方法，用于扩展模型的上下文窗口长度。'none'：不使用扩展；'linear'：线性插值；'dynamic'：动态插值；'yarn'：Yet Another RoPE Extension；'llama3'：专为LLaMA-3模型优化的方法。")
        booster = gr.Dropdown(choices=["auto", "flashattn2", "unsloth", "liger_kernel"], value="auto", info="用于提升训练和推理速度的加速方法。'auto'：自动选择最适合的方法；'flashattn2'：Flash Attention 2加速；'unsloth'：使用Unsloth库加速；'liger_kernel'：Liger内核优化。")

    model_name.change(get_model_info, [model_name], [model_path, template], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )
    model_name.input(save_config, inputs=[lang, model_name], queue=False)
    model_path.input(save_config, inputs=[lang, model_name, model_path], queue=False)
    finetuning_type.change(can_quantize, [finetuning_type], [quantization_bit], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )
    checkpoint_path.focus(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False)
    quantization_method.change(can_quantize_to, [quantization_method], [quantization_bit], queue=False)

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        finetuning_type=finetuning_type,
        checkpoint_path=checkpoint_path,
        quantization_bit=quantization_bit,
        quantization_method=quantization_method,
        template=template,
        rope_scaling=rope_scaling,
        booster=booster,
    )
