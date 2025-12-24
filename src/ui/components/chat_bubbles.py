"""iPhone-style chat bubble components for Streamlit.

Provides iPhone Messages-style chat UI while preserving Streamlit's
native markdown rendering and copy functionality.
"""

import streamlit as st


def inject_chat_styles():
    """Inject iPhone-style chat CSS into the page."""
    st.markdown(
        """
        <style>
        /* Style the native Streamlit chat messages to look like iPhone */

        /* User messages - right aligned, blue bubble */
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse !important;
            background: transparent !important;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) > div:last-child {
            background: linear-gradient(135deg, #007AFF 0%, #0056CC 100%) !important;
            color: white !important;
            border-radius: 20px 20px 4px 20px !important;
            padding: 12px 16px !important;
            max-width: 85% !important;
            margin-left: auto !important;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) > div:last-child * {
            color: white !important;
        }

        /* Assistant messages - left aligned, gray bubble */
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
            background: transparent !important;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) > div:last-child {
            background: linear-gradient(135deg, #E9E9EB 0%, #DCDCE0 100%) !important;
            color: #1C1C1E !important;
            border-radius: 20px 20px 20px 4px !important;
            padding: 12px 16px !important;
            max-width: 85% !important;
        }

        /* Hide the default avatars for cleaner look */
        div[data-testid="stChatMessageAvatarUser"],
        div[data-testid="stChatMessageAvatarAssistant"] {
            display: none !important;
        }

        /* Code blocks in messages */
        div[data-testid="stChatMessage"] pre {
            background: rgba(0, 0, 0, 0.1) !important;
            border-radius: 8px !important;
            padding: 12px !important;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) pre {
            background: rgba(255, 255, 255, 0.15) !important;
        }

        /* Inline code */
        div[data-testid="stChatMessage"] code {
            background: rgba(0, 0, 0, 0.1) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) code {
            background: rgba(255, 255, 255, 0.2) !important;
        }

        /* Model indicator styling */
        .model-indicator {
            font-size: 11px;
            color: #8E8E93;
            margin-top: 4px;
            padding-left: 8px;
        }

        /* Typing indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 12px 16px;
            background: linear-gradient(135deg, #E9E9EB 0%, #DCDCE0 100%);
            border-radius: 20px 20px 20px 4px;
            max-width: fit-content;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: #8E8E93;
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-4px); }
        }

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) > div:last-child {
                background: linear-gradient(135deg, #3A3A3C 0%, #2C2C2E 100%) !important;
                color: #FFFFFF !important;
            }

            div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) > div:last-child * {
                color: #FFFFFF !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chat_message(
    content: str,
    role: str,
    model: str = None,
    show_model_indicator: bool = True,
    key: str = None,
):
    """Render a single chat message using Streamlit's native chat with iPhone styling.

    This uses st.chat_message for proper markdown rendering and copy support,
    while the CSS makes it look like iPhone Messages.

    Args:
        content: The message content (supports markdown)
        role: Either "user" or "assistant"
        model: Optional model name for indicator
        show_model_indicator: Whether to show model indicator for assistant messages
        key: Optional unique key for the message
    """
    with st.chat_message(role):
        st.markdown(content)

        # Show model indicator for assistant messages
        if show_model_indicator and role == "assistant" and model:
            if "haiku" in model.lower():
                st.caption("_Haiku_")
            elif "sonnet" in model.lower():
                st.caption("_Sonnet_")
            elif "opus" in model.lower():
                st.caption("_Opus_")


def render_typing_indicator():
    """Render a typing indicator (three dots animation)."""
    st.markdown(
        """
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
