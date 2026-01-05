"""LLM-based email content extraction for better embedding quality.

This module uses an LLM to identify meaningful content boundaries in emails,
excluding boilerplate like signatures, contact info, and mailto links.
The LLM returns first/last words as anchors, we find them in the original
text and extract the substring - ensuring zero modification of content.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Tuple

from litellm import completion

from ..config import get_settings

logger = logging.getLogger(__name__)

# System prompt for content extraction using word anchors
EXTRACTION_SYSTEM_PROMPT = """You are an email content analyzer. Given an email (which may contain multiple messages in a thread), identify the meaningful content by returning the FIRST and LAST few words.

TASK: Find where meaningful content starts and ends, then return the first 5-8 words and last 5-8 words as anchors.

WHAT TO INCLUDE:
- All actual message text, questions, instructions, technical content
- Greetings like "Good day", "Dear John"
- Closings with sender name ("Best regards, John Smith" or "Best Regards, DLL / TECH / MTM")
- Quote attributions ("On 2025-03-13, John wrote:")
- Lists, bullet points, technical data

WHAT TO EXCLUDE (skip over these to find start/end):
- Email signatures with contact blocks (phone, fax, E-Mail:, T:, M:, VSAT:, etc.)
- Mailto syntax like <mailto:email@domain.com>
- Image references like [cid:xxx@domain.com]
- Lines of underscores (______) or dashes (-----)
- Standard legal notices, disclaimers
- Placeholder markers like [X]
- Header lines at very start: standalone "Date:", "From:", "To:", "Att:" lines

Return JSON:
{
  "first_words": "the first 5-8 words of meaningful content",
  "last_words": "the last 5-8 words of meaningful content"
}

Be GENEROUS - include more content rather than less. Copy the words EXACTLY as they appear."""


def find_anchor_position(text: str, anchor: str, find_end: bool = False) -> Optional[int]:
    """Find position of anchor words in text.

    Args:
        text: Full text to search in.
        anchor: Words to find.
        find_end: If True, return position after the anchor (for last_words).

    Returns:
        Character position, or None if not found.
    """
    if not anchor:
        return None

    # Try exact match first
    pos = text.find(anchor)
    if pos != -1:
        return pos + len(anchor) if find_end else pos

    # Try case-insensitive
    pos = text.lower().find(anchor.lower())
    if pos != -1:
        return pos + len(anchor) if find_end else pos

    # Try matching with flexible whitespace
    pattern = r'\s+'.join(re.escape(word) for word in anchor.split())
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.end() if find_end else match.start()

    return None


async def extract_content_bounds(
    text: str,
    model: Optional[str] = None,
) -> Tuple[int, int]:
    """Use LLM to identify start/end positions of meaningful content via word anchors.

    Args:
        text: Raw email body text.
        model: LLM model to use (defaults to configured email_cleaner_model).

    Returns:
        Tuple of (start, end) positions.
        Falls back to (0, len(text)) on error.
    """
    if not text or not text.strip():
        return (0, len(text))

    if model is None:
        settings = get_settings()
        model = settings.get_model(settings.email_cleaner_model)

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
        )

        result = json.loads(response.choices[0].message.content)
        first_words = result.get("first_words", "")
        last_words = result.get("last_words", "")

        # Find positions using anchors
        start = find_anchor_position(text, first_words, find_end=False)
        end = find_anchor_position(text, last_words, find_end=True)

        # Fall back to full text if anchors not found
        if start is None:
            logger.warning(f"Could not find first_words anchor: {first_words[:50]}...")
            start = 0
        if end is None:
            logger.warning(f"Could not find last_words anchor: {last_words[:50]}...")
            end = len(text)

        # Ensure valid range
        if start >= end:
            logger.warning("Invalid range from anchors, using full text")
            return (0, len(text))

        return (start, end)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return (0, len(text))
    except Exception as e:
        logger.warning(f"LLM content extraction failed: {e}")
        return (0, len(text))


def extract_clean_content(text: str, bounds: Tuple[int, int]) -> str:
    """Extract content using bounds.

    Args:
        text: Original email text.
        bounds: (start, end) tuple.

    Returns:
        Extracted content - original text preserved exactly, just substring.
    """
    start, end = bounds
    return text[start:end].strip()


def split_into_messages(text: str) -> list[str]:
    """Split an email thread into individual messages.

    Detects message boundaries using common patterns:
    - "On DATE, NAME wrote:"
    - "From: NAME Sent: DATE To: NAME Subject: TEXT"
    - "-----Original Message-----"

    Args:
        text: Full email body with potentially multiple messages.

    Returns:
        List of individual message texts, most recent first.
    """
    # Patterns that indicate start of a quoted message
    # These are used to find split points
    split_patterns = [
        # Gmail style: "On 2025-03-13 08:34, Technical MTM wrote:"
        r'On\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2},\s+[^\n]+\s+wrote:',
        # Gmail style: "On Mar 13, 2025, at 8:34 AM, John wrote:"
        r'On\s+[A-Za-z]{3}\s+\d{1,2},\s+\d{4},?\s+(?:at\s+)?[\d:]+\s*(?:AM|PM)?,\s+[^\n]+\s+wrote:',
        # Outlook style header block
        r'From:\s+[^\n]+\nSent:\s+[^\n]+\nTo:\s+[^\n]+\n(?:Cc:\s+[^\n]+\n)?Subject:\s+[^\n]+',
        # Original Message separator
        r'-{3,}\s*Original Message\s*-{3,}',
    ]

    combined_pattern = '(' + '|'.join(split_patterns) + ')'

    # Split while keeping the delimiters
    parts = re.split(combined_pattern, text, flags=re.IGNORECASE | re.MULTILINE)

    messages = []
    current_message = ""

    i = 0
    while i < len(parts):
        part = parts[i]
        if part is None:
            i += 1
            continue

        # Check if this part matches any split pattern (is a delimiter)
        is_delimiter = any(re.match(p, part.strip(), re.IGNORECASE) for p in split_patterns)

        if is_delimiter:
            # Save current message if we have content
            if current_message.strip():
                messages.append(current_message.strip())
            # Start new message with the delimiter (attribution line)
            current_message = part
        else:
            current_message += part

        i += 1

    # Don't forget the last message
    if current_message.strip():
        messages.append(current_message.strip())

    return messages if messages else [text]


def remove_boilerplate_from_message(text: str) -> str:
    """Remove boilerplate patterns from a single message.

    Args:
        text: Single message text.

    Returns:
        Cleaned message text.
    """
    # Remove mailto links: <mailto:email@domain.com>
    text = re.sub(r'<mailto:[^>]+>', '', text)

    # Remove CID image references: [cid:xxx@domain.com]
    text = re.sub(r'\[cid:[^\]]+\]', '', text)

    # Remove placeholder markers: [X] or [ ]
    text = re.sub(r'\[\s*[Xx]?\s*\]', '', text)

    # Remove horizontal separator lines (6+ underscores or dashes)
    text = re.sub(r'^[_\-]{6,}\s*$', '', text, flags=re.MULTILINE)

    # Remove E-Mail: lines
    text = re.sub(r'^E-Mail:\s*[^\n]+$', '', text, flags=re.MULTILINE)

    # Remove contact info lines: T:, M:, E: followed by phone/email
    text = re.sub(r'^[TME]:\s*\+?[\d\s\-\.]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[TME]:\s*\S+@\S+$', '', text, flags=re.MULTILINE)

    # Remove satellite/fax lines: INM FBB, VSAT, FAX (with various number formats)
    text = re.sub(r'^(?:INM\s+FBB|VSAT|FAX)[\s\d:]+[\d\s\-\.]+$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove full address blocks (city, country, postal code patterns)
    text = re.sub(r'^\d+[-\d]*\s+[A-Za-z\s]+(?:Street|Road|Ave|Avenue|Blvd),?\s*\d*,?\s*[A-Za-z\s]+,?\s*[A-Za-z\s]*$',
                  '', text, flags=re.MULTILINE)

    # Remove header lines at start of message: Date:, From:, To:, Att:, Sent:
    text = re.sub(r'^(?:Date|From|To|Att|Cc|Subject|Sent):\s*[^\n]+$', '', text, flags=re.MULTILINE)

    # Remove standard notice patterns
    text = re.sub(r'^IMPORTANT:\s*Please always use[^\n]+$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^Please send all invoices[^\n]+$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove company titles/roles on their own lines after signatures
    text = re.sub(r'^(?:Technical Department|Fleet Manager|Chief Engineer)[^\n]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(?:Maran Tankers|Management Inc)[^\n]*$', '', text, flags=re.MULTILINE)

    # Normalize multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


async def clean_email_body(
    text: str,
    model: Optional[str] = None,
) -> str:
    """Clean email body by extracting only meaningful content.

    This is the main entry point. It:
    1. Splits the email thread into individual messages
    2. Removes boilerplate from each message (signatures, contact info, etc.)
    3. Optionally uses LLM for outer boundary detection
    4. Returns cleaned text preserving conversation flow

    Args:
        text: Raw email body text.
        model: LLM model to use (optional, for outer boundary detection).

    Returns:
        Cleaned email text with boilerplate removed.
    """
    if not text or not text.strip():
        return text

    # Step 1: Use LLM to identify outer content boundaries (removes header lines)
    if model:
        bounds = await extract_content_bounds(text, model)
        text = extract_clean_content(text, bounds)

    # Step 2: Split into individual messages
    messages = split_into_messages(text)

    # Step 3: Clean each message individually
    cleaned_messages = []
    for msg in messages:
        cleaned = remove_boilerplate_from_message(msg)
        if cleaned:
            cleaned_messages.append(cleaned)

    # Step 4: Rejoin messages
    result = "\n\n".join(cleaned_messages)

    # Final cleanup
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()
