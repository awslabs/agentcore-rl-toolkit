"""Few-shot example for the AppWorld agent in Strands toolUse/toolResult format.

Adapted from
https://github.com/StonyBrookNLP/appworld/blob/main/experiments/prompts/react_code_agent/instructions.txt.

Teaches the model: API discovery workflow, credential retrieval via supervisor,
pagination handling, and task completion with complete_task().

Uses static dummy values — the example demonstrates the interaction pattern,
not real task data.
"""

import json


def _tool_use(tid, code):
    """Create a toolUse content block."""
    return {"toolUse": {"toolUseId": tid, "name": "execute", "input": {"code": code}}}


def _tool_result(tid, output):
    """Create a toolResult content block."""
    return {"toolResult": {"toolUseId": tid, "status": "success", "content": [{"text": output}]}}


EXAMPLE_APP_DESCRIPTIONS = json.dumps(
    [
        {"name": "spotify", "description": "A music streaming app."},
        {"name": "venmo", "description": "A payments and social networking app."},
        {"name": "phone", "description": "A phone and contacts app."},
        {"name": "supervisor", "description": "An app to manage your personal information."},
        {"name": "file_system", "description": "A file system app to manage files and folders."},
        {"name": "gmail", "description": "An email client app."},
    ],
    indent=1,
)


def build_example_messages():
    """Build 10-turn Spotify playlist example in function-calling format.

    Returns:
        List of Strands-format messages for the worked example.
    """
    return [
        # ── Example task ──
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "My name is: John Doe. "
                        "My personal email is john.doe@gmail.com "
                        "and phone number is 555-0100.\n"
                        "Task: How many playlists do I have in my Spotify playlist library?"
                    )
                }
            ],
        },
        # Turn 1: Discover Spotify APIs
        {
            "role": "assistant",
            "content": [
                {"text": "Okay. Let's first find which APIs are available to use in Spotify."},
                _tool_use("ex_1", "print(apis.api_docs.show_api_descriptions(app_name='spotify'))"),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "ex_1",
                    "[\n ...\n"
                    ' "login : Login to your account.",\n'
                    ' "logout : Logout from your account.",\n'
                    " ...\n]",
                )
            ],
        },
        # Turn 2: Check login API spec
        {
            "role": "assistant",
            "content": [
                {"text": "Okay. Looks like I can use the `login` api. Let's find its specifications."},
                _tool_use("ex_2", "print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))"),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "ex_2",
                    json.dumps(
                        {
                            "app_name": "spotify",
                            "api_name": "login",
                            "path": "/spotify/auth/token",
                            "method": "POST",
                            "description": "Login to your account.",
                            "parameters": [
                                {
                                    "name": "username",
                                    "type": "string",
                                    "required": True,
                                    "description": "Your account email.",
                                    "default": None,
                                    "constraints": [],
                                },
                                {
                                    "name": "password",
                                    "type": "string",
                                    "required": True,
                                    "description": "Your account password.",
                                    "default": None,
                                    "constraints": [],
                                },
                            ],
                            "response_schemas": {
                                "success": {"access_token": "string", "token_type": "string"},
                                "failure": {"message": "string"},
                            },
                        },
                        indent=1,
                    ),
                )
            ],
        },
        # Turn 3: Discover apps for credentials
        {
            "role": "assistant",
            "content": [
                {
                    "text": "I need the supervisor's username and password. "
                    "Let's see if any app can help me get that."
                },
                _tool_use("ex_3", "print(apis.api_docs.show_app_descriptions())"),
            ],
        },
        {"role": "user", "content": [_tool_result("ex_3", EXAMPLE_APP_DESCRIPTIONS)]},
        # Turn 4: Check supervisor APIs
        {
            "role": "assistant",
            "content": [
                {"text": "Looks like the supervisor app could help me with that."},
                _tool_use("ex_4", "print(apis.api_docs.show_api_descriptions(app_name='supervisor'))"),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "ex_4",
                    "[\n ...\n" ' "show_account_passwords : Show your supervisor\'s account passwords."\n' " ...\n]",
                )
            ],
        },
        # Turn 5: Get passwords
        {
            "role": "assistant",
            "content": [
                {
                    "text": "I can use `show_account_passwords` to get the passwords. "
                    "Let me call that and save it in a variable for future reference."
                },
                _tool_use("ex_5", "passwords = apis.supervisor.show_account_passwords()\nprint(passwords)"),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "ex_5",
                    json.dumps(
                        [
                            {"account_name": "spotify", "password": "dummy_spotify_pass"},
                            {"account_name": "file_system", "password": "dummy_fs_pass"},
                        ],
                        indent=1,
                    ),
                )
            ],
        },
        # Turn 6: Login to Spotify
        {
            "role": "assistant",
            "content": [
                {
                    "text": "So the Spotify password is in the `passwords` list. "
                    "I can use it to login now with the provided email address: john.doe@gmail.com."
                },
                _tool_use(
                    "ex_6",
                    "spotify_password = [\n"
                    '    p for p in passwords if p["account_name"] == "spotify"\n'
                    '][0]["password"]\n'
                    "login_result = apis.spotify.login(username='john.doe@gmail.com', "
                    "password=spotify_password)\n"
                    "print(login_result)",
                ),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "ex_6", json.dumps({"access_token": "fake_access_token", "token_type": "Bearer"}, indent=1)
                )
            ],
        },
        # Turn 7: Find playlist API
        {
            "role": "assistant",
            "content": [
                {"text": "First, let's check the available APIs for the Spotify app."},
                _tool_use("ex_7", "print(apis.api_docs.show_api_descriptions(app_name='spotify'))"),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "ex_7",
                    "[\n ...\n"
                    ' "show_playlist_library : Search or show a list of playlists '
                    'in your playlist library.",\n'
                    ' "show_playlist : Get detailed information about a specific playlist.",\n'
                    " ...\n]",
                )
            ],
        },
        # Turn 8: Check playlist API spec
        {
            "role": "assistant",
            "content": [
                {
                    "text": "The `show_playlist_library` API seems to be the one we need. "
                    "Let's check its specifications."
                },
                _tool_use(
                    "ex_8", "print(apis.api_docs.show_api_doc(" "app_name='spotify', api_name='show_playlist_library'))"
                ),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "ex_8",
                    json.dumps(
                        {
                            "app_name": "spotify",
                            "api_name": "show_playlist_library",
                            "path": "/spotify/library/playlists",
                            "method": "GET",
                            "description": "Search or show a list of playlists in your playlist library.",
                            "parameters": [
                                {
                                    "name": "access_token",
                                    "type": "string",
                                    "required": True,
                                    "description": "Access token obtained from spotify app login.",
                                    "default": None,
                                    "constraints": [],
                                },
                                {
                                    "name": "page_index",
                                    "type": "integer",
                                    "required": False,
                                    "description": "The index of the page to return.",
                                    "default": 0,
                                    "constraints": ["value >= 0"],
                                },
                                {
                                    "name": "page_limit",
                                    "type": "integer",
                                    "required": False,
                                    "description": "The maximum number of results to return per page.",
                                    "default": 5,
                                    "constraints": ["value >= 1, <= 20"],
                                },
                            ],
                            "response_schemas": {
                                "success": [{"playlist_id": 1, "title": "string", "is_public": True, "song_ids": [1]}],
                                "failure": {"message": "string"},
                            },
                        },
                        indent=1,
                    ),
                )
            ],
        },
        # Turn 9: Get all playlists with pagination
        {
            "role": "assistant",
            "content": [
                {"text": "I need to page through all the playlists to get the full list."},
                _tool_use(
                    "ex_9",
                    "page_index = 0\n"
                    "playlists = []\n"
                    "while page_index < 10:\n"
                    "    playlist_page = apis.spotify.show_playlist_library(\n"
                    '        access_token=login_result["access_token"],\n'
                    "        page_index=page_index)\n"
                    "    if playlist_page:\n"
                    "        playlists.extend(playlist_page)\n"
                    "        page_index += 1\n"
                    "    else:\n"
                    "        break\n"
                    "num_playlists = len(playlists)\n"
                    "print(num_playlists)",
                ),
            ],
        },
        {"role": "user", "content": [_tool_result("ex_9", "23")]},
        # Turn 10: Complete task
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Now that the task is completed, I can call "
                    "apis.supervisor.complete_task(). Since this task has an "
                    "answer to be returned, I will pass that as an argument."
                },
                _tool_use("ex_10", "apis.supervisor.complete_task(answer=num_playlists)"),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result(
                    "ex_10",
                    "Marked the active task complete." "\n\n[TASK COMPLETED] Stop and do not call any more tools.",
                )
            ],
        },
        # Final response: text only, no tool call → triggers end_turn
        {
            "role": "assistant",
            "content": [
                {"text": "The task is complete. The answer is 23 playlists."},
            ],
        },
    ]
