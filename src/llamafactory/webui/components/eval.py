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

from ...extras.packages import is_gradio_available
from ..common import DEFAULT_DATA_DIR
from ..control import list_datasets
from .data import create_preview_box


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_eval_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=2, info="评估数据所在的文件夹路径。可以是相对路径（如'data/'）或绝对路径。该文件夹应包含格式正确的数据集文件，通常是JSON或JSONL格式。")
        dataset = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=4, info="选择用于评估的数据集。可以选择多个数据集进行综合评估，系统会自动合并结果。测试数据集应该与模型训练目标相匹配，以获得有意义的评估结果。")
        preview_elems = create_preview_box(dataset_dir, dataset)

    input_elems.update({dataset_dir, dataset})
    elem_dict.update(dict(dataset_dir=dataset_dir, dataset=dataset, **preview_elems))

    with gr.Row():
        cutoff_len = gr.Slider(minimum=4, maximum=131072, value=1024, step=1, info="评估时输入序列的最大token数。超过此长度的序列会被截断。评估时通常可以设置得比训练时小一些，特别是在仅关注模型对短输入响应质量的情况下。")
        max_samples = gr.Textbox(value="100000", info="评估过程中使用的最大样本数量。设置为较大值（如100000）表示使用全部数据集；设置为较小值（如100）可用于快速验证模型性能。")
        batch_size = gr.Slider(minimum=1, maximum=1024, value=2, step=1, info="每个GPU处理的样本数量。评估时可以使用比训练时更大的批处理大小，因为不需要存储梯度。较大的值可以加速评估过程，但需要更多GPU内存。")
        predict = gr.Checkbox(value=True, info="是否生成预测结果。启用时将使用模型生成响应并计算评估指标；禁用时仅计算损失等基本指标而不生成文本，评估速度更快但信息更少。")

    input_elems.update({cutoff_len, max_samples, batch_size, predict})
    elem_dict.update(dict(cutoff_len=cutoff_len, max_samples=max_samples, batch_size=batch_size, predict=predict))

    with gr.Row():
        max_new_tokens = gr.Slider(minimum=8, maximum=4096, value=512, step=1, info="生成回答的最大token数量。设置合适的值避免生成过长或过短的回答。对于简短的任务（如分类）可设置较小值（如64-128）；对于需要详细解释的任务可设置较大值（如512-1024）。")
        top_p = gr.Slider(minimum=0.01, maximum=1, value=0.7, step=0.01, info="核采样参数，控制生成的多样性。较小的值（如0.3-0.5）使生成更确定和保守；较大的值（如0.7-0.95）使生成更多样和创造性。测试时推荐使用0.7作为平衡值。")
        temperature = gr.Slider(minimum=0.01, maximum=1.5, value=0.95, step=0.01, info="温度参数，控制生成的随机性。较小的值（如0.1-0.5）使生成更确定性；较大的值（如0.8-1.2）使生成更多样化。评估特定能力时，建议使用较低温度；评估创造性时，可使用较高温度。")
        output_dir = gr.Textbox(info="保存评估结果的目录路径。结果将包含详细的评估指标和生成的文本（如果启用预测）。留空则使用默认路径。")

    input_elems.update({max_new_tokens, top_p, temperature, output_dir})
    elem_dict.update(dict(max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature, output_dir=output_dir))

    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        resume_btn = gr.Checkbox(visible=False, interactive=False)
        progress_bar = gr.Slider(visible=False, interactive=False)

    with gr.Row():
        output_box = gr.Markdown()

    elem_dict.update(
        dict(
            cmd_preview_btn=cmd_preview_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            resume_btn=resume_btn,
            progress_bar=progress_bar,
            output_box=output_box,
        )
    )
    output_elems = [output_box, progress_bar]

    cmd_preview_btn.click(engine.runner.preview_eval, input_elems, output_elems, concurrency_limit=None)
    start_btn.click(engine.runner.run_eval, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)
    resume_btn.change(engine.runner.monitor, outputs=output_elems, concurrency_limit=None)

    dataset.focus(list_datasets, [dataset_dir], [dataset], queue=False)

    return elem_dict
