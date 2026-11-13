"""AgentEnv's SlackEnv example, with AgentCore MCP server settings.

Source (Apache-2.0, Copyright 2026 Scale AI):
https://github.com/scaleapi/agentenv-framework/blob/3cf9dd363ce9e63197b79d15834179b450bd295f/packages/agentenv-protocol/README.md
This product includes software developed at Scale AI (https://scale.com).
"""

import base64
import json
import os
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse
from urllib.request import urlopen

from agentenv_protocol import (
    AgentEnvEnvironment,
    DataPart,
    FilePart,
    add_data,
    environment_card,
    get_data,
    reset_data,
    tool,
)
from pydantic import Field


@environment_card(name="slack")
class SlackEnv(AgentEnvEnvironment):
    def __init__(self):
        self.channels, self.messages = {}, []

    @reset_data
    async def _reset(self):
        self.channels.clear()
        self.messages.clear()

    @add_data
    async def _add(self, parts):
        # Seeds arrive as an inline DataPart (live deploy) OR a FilePart whose
        # file carries inline bytes, a file:// URI (a staged file or an exported
        # bundle) or an https:// URL (a signed URL, when agent-env can't stage the
        # file). Handle all four — dropping the FilePart branch makes bundles load empty.
        for p in parts:
            if isinstance(p, DataPart):
                payload = p.data
            elif isinstance(p, FilePart):
                f = p.file
                if getattr(f, "bytes", None) is not None:
                    payload = json.loads(base64.b64decode(f.bytes))
                elif urlparse(f.uri).scheme in ("http", "https"):
                    with urlopen(f.uri) as response:
                        payload = json.loads(response.read())
                else:
                    payload = json.loads(Path(urlparse(f.uri).path).read_bytes())
            else:
                continue  # TextPart / unknown — nothing to load
            self.messages.extend(payload.get("messages", []))

    @get_data
    async def _state(self):
        return [DataPart(data={"channels": list(self.channels.values()), "messages": self.messages})]

    @tool(name="{environment_name}_send_message")
    def send_message(
        self,
        channel: Annotated[str, Field(description="Channel to post to.")],
        text: Annotated[str, Field(description="Message text.")],
    ) -> str:
        """Send a message to a channel."""
        self.messages.append({"channel": channel, "text": text})
        return "ok"


def create_server():
    server = SlackEnv().create_app()
    server.settings.port = int(os.environ.get("MCP_PORT", "8000"))
    server.settings.stateless_http = True
    server.settings.json_response = True
    return server


if __name__ == "__main__":
    create_server().run(transport="streamable-http")
