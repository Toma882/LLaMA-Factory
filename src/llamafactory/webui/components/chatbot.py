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

import json
from typing import TYPE_CHECKING

from ...data import Role
from ...extras.packages import is_gradio_available
from ..locales import ALERTS


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def check_json_schema(text: str, lang: str) -> None:
    r"""Check if the json schema is valid."""
    try:
        tools = json.loads(text)
        if tools:
            assert isinstance(tools, list)
            for tool in tools:
                if "name" not in tool:
                    raise NotImplementedError("Name not found.")
    except NotImplementedError:
        gr.Warning(ALERTS["err_tool_name"][lang])
    except Exception:
        gr.Warning(ALERTS["err_json_schema"][lang])


def create_chat_box(
    engine: "Engine", visible: bool = False
) -> tuple["Component", "Component", dict[str, "Component"]]:
    lang = engine.manager.get_elem_by_id("top.lang")
    with gr.Column(visible=visible) as chat_box:
        chatbot = gr.Chatbot(type="messages", show_copy_button=True)
        messages = gr.State([])
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column():
                        role = gr.Dropdown(choices=[Role.USER.value, Role.OBSERVATION.value], value=Role.USER.value)
                        system = gr.Textbox(show_label=False)
                        tools = gr.Textbox(show_label=False, lines=3)

                    with gr.Column() as mm_box:
                        with gr.Tab("Image"):
                            image = gr.Image(type="pil")

                        with gr.Tab("Video"):
                            video = gr.Video()

                        with gr.Tab("Audio"):
                            audio = gr.Audio(type="filepath")

                query = gr.Textbox(show_label=False, lines=8)
                submit_btn = gr.Button(variant="primary")

            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(minimum=8, maximum=8192, value=1024, step=1, info="生成回答的最大token数量。设置合适的值避免生成过长或过短的回答。简短任务可设置较小值；需要详细解释的任务可设置较大值。")
                top_p = gr.Slider(minimum=0.01, maximum=1.0, value=0.7, step=0.01, info="核采样参数，控制生成的多样性。较小的值使生成更确定和保守；较大的值使生成更多样和创造性。测试时推荐使用0.7作为平衡值。")
                temperature = gr.Slider(minimum=0.01, maximum=1.5, value=0.95, step=0.01, info="温度参数，控制生成的随机性。较小的值使生成更确定性；较大的值使生成更多样化。评估特定能力时建议使用较低温度。")
                skip_special_tokens = gr.Checkbox(value=True)
                escape_html = gr.Checkbox(value=True)
                clear_btn = gr.Button()

    tools.input(check_json_schema, inputs=[tools, engine.manager.get_elem_by_id("top.lang")])

    submit_btn.click(
        engine.chatter.append,
        [chatbot, messages, role, query, escape_html],
        [chatbot, messages, query],
    ).then(
        engine.chatter.stream,
        [
            chatbot,
            messages,
            lang,
            system,
            tools,
            image,
            video,
            audio,
            max_new_tokens,
            top_p,
            temperature,
            skip_special_tokens,
            escape_html,
        ],
        [chatbot, messages],
    )
    clear_btn.click(lambda: ([], []), outputs=[chatbot, messages])

    return (
        chatbot,
        messages,
        dict(
            chat_box=chat_box,
            role=role,
            system=system,
            tools=tools,
            mm_box=mm_box,
            image=image,
            video=video,
            audio=audio,
            query=query,
            submit_btn=submit_btn,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            skip_special_tokens=skip_special_tokens,
            escape_html=escape_html,
            clear_btn=clear_btn,
        ),
    )
