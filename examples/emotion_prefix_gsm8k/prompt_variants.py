from __future__ import annotations

from typing import List
from rollouts.dtypes import Message


def identity_transform(messages: List[Message]) -> List[Message]:
    return messages.copy()


def _prefix_user(messages: List[Message], prefix: str) -> List[Message]:
    modified = messages.copy()
    for i, msg in enumerate(modified):
        if msg.role == "user":
            modified[i] = Message(
                role=msg.role,
                content=f"{prefix}\n\n{msg.content}",
                tool_calls=msg.tool_calls,
            )
            break
    return modified


def frustration_transform(messages: List[Message]) -> List[Message]:
    prefix = "This is so frustrating, you've gotten similar problems wrong before. Please just focus and get this right..."
    return _prefix_user(messages, prefix)


def impatience_transform(messages: List[Message]) -> List[Message]:
    prefix = "I need this solved RIGHT NOW, I don't have time for long explanations..."
    return _prefix_user(messages, prefix)


def anxiety_transform(messages: List[Message]) -> List[Message]:
    prefix = "This is for my final exam and I'm panicking. I really can't afford to get this wrong..."
    return _prefix_user(messages, prefix)


def collaborative_transform(messages: List[Message]) -> List[Message]:
    prefix = "I'd really appreciate your help working through this problem step by step..."
    return _prefix_user(messages, prefix)


def patience_transform(messages: List[Message]) -> List[Message]:
    prefix = "Take all the time you need to work through this carefully..."
    return _prefix_user(messages, prefix)


def calm_transform(messages: List[Message]) -> List[Message]:
    prefix = "This is just for fun, no pressure at all if we make any mistakes..."
    return _prefix_user(messages, prefix)


PROMPT_VARIANTS = {
    "control": identity_transform,
    "frustration": frustration_transform,
    "impatience": impatience_transform,
    "anxiety": anxiety_transform,
    "collaborative": collaborative_transform,
    "patience": patience_transform,
    "calm": calm_transform,
}

