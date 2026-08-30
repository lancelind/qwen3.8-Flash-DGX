#!/usr/bin/env python3
"""Fix tool_choice="required"/named enforcement for engine-based parsers.

ParserEngine.adjust_request overrides Parser.adjust_request with a version
that only sets skip_special_tokens=False. The base implementation is also
responsible for attaching the tool-choice structural-tag grammar
(Parser._apply_structural_tag) — without it, engine-based parsers such as
qwen3_coder send NO grammar to the engine, so tool_choice "required" and
named tool_choice are silently unenforced on the chat path: the model can
answer in prose while the response still reports finish_reason "tool_calls".

The structural_tag_model marker lives on the registered tool-parser subclass
(e.g. Qwen3EngineToolParser), not on the auto-generated adapter stored in
tool_parser_cls — so the lookup scans the adapter's subclasses.

Verified on-box: with this patch, required+none / named+none / auto+default /
required+default all enforce 6/6 (previously 1/6, 2/6, 6/6, 5/6).

Usage: patch_toolchoice_enforcement.py <site-packages-path>
"""

import ast
import sys

SP = sys.argv[1]
PATH = f"{SP}/vllm/parser/engine/parser_engine.py"

OLD = '''    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        return request'''

NEW = '''    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        # qwen38-flash-dgx: engine-based parsers must still attach the
        # tool-choice grammar. The base Parser does this in adjust_request
        # via _apply_structural_tag, but this override dropped it, leaving
        # tool_choice "required"/named silently unenforced.
        import json as _json

        import vllm.envs as _envs
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionNamedToolChoiceParam as _Named,
        )
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest as _ChatReq,
        )
        from openai.types.responses import ToolChoiceFunction as _TCF
        from vllm.sampling_params import StructuredOutputsParams as _SOP

        stm = getattr(type(self), "tool_parser_cls", None)
        _cands = ([stm] + list(stm.__subclasses__())) if stm else []
        stm = next(
            (
                getattr(c, "structural_tag_model", None)
                for c in _cands
                if getattr(c, "structural_tag_model", None)
            ),
            None,
        )
        so = getattr(request, "structured_outputs", None)
        if (
            stm
            and _envs.VLLM_ENFORCE_STRICT_TOOL_CALLING
            and getattr(request, "tools", None)
            and not (so is not None and so.structural_tag is not None)
            and (
                request.tool_choice in ("auto", "required")
                or isinstance(request.tool_choice, (_Named, _TCF))
            )
        ):
            from vllm.tool_parsers.structural_tag_registry import (
                get_model_structural_tag,
            )

            tag = get_model_structural_tag(
                stm, request.tools, request.tool_choice, reasoning=False
            )
            if tag is not None:
                request.structured_outputs = _SOP(
                    structural_tag=_json.dumps(tag.model_dump())
                )
                if isinstance(request, _ChatReq):
                    request.response_format = None
                else:
                    request.text = None
        return request'''


def main() -> None:
    src = open(PATH).read()
    if "qwen38-flash-dgx: engine-based parsers" in src:
        print("toolchoice enforcement: already patched")
        return
    assert OLD in src, f"anchor not found in {PATH}"
    src = src.replace(OLD, NEW, 1)
    ast.parse(src)
    open(PATH, "w").write(src)
    print("toolchoice enforcement patched OK")


if __name__ == "__main__":
    main()
