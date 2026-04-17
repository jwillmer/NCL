"""Image processor using LiteLLM for classification and description."""

from __future__ import annotations

import base64
import logging
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from litellm import acompletion
from pydantic import BaseModel, Field

from ..config import get_settings
from ..llm_privacy import OPENROUTER_PRIVACY_EXTRA_BODY

logger = logging.getLogger(__name__)


class ImageClassification(str, Enum):
    """Classification of image content."""

    LOGO = "logo"
    BANNER = "banner"
    SIGNATURE = "signature"
    ICON = "icon"
    MEANINGFUL = "meaningful"


class ImageClassifyResponse(BaseModel):
    """Structured response for image classification and description."""

    classification: Literal["logo", "banner", "signature", "icon", "meaningful"] = Field(
        description="The category of the image: 'logo' for company logos/brand marks, "
        "'banner' for email headers/footers/promotional banners, "
        "'signature' for email signature images/contact cards, "
        "'icon' for social media/UI icons, "
        "'meaningful' for actual content like screenshots/diagrams/charts/photos"
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed description of the image content. "
        "Only provided if classification is 'meaningful'. "
        "Include any visible text, diagrams, data, or important visual elements.",
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )


class ImageDescribeResponse(BaseModel):
    """Structured response for image description only."""

    description: str = Field(
        description="Detailed description of the image content including: "
        "any visible text (transcribed accurately), diagrams/charts/visual elements, "
        "the overall context and purpose, and key information useful for search"
    )
    contains_text: bool = Field(description="Whether the image contains readable text")
    image_type: str = Field(
        description="Type of image: screenshot, diagram, chart, photo, document, etc."
    )


class ImageAnalysisResult(BaseModel):
    """Result of image analysis."""

    classification: ImageClassification
    description: Optional[str] = None
    skip_reason: Optional[str] = None

    @property
    def should_skip(self) -> bool:
        """Whether this image should be skipped from indexing."""
        return self.classification != ImageClassification.MEANINGFUL


_CLASSIFIER_PROMPT = """\
You are an image classifier for email attachments.
Your job is to determine if an image contains meaningful content or is just decoration.

Classify images as:
- logo: Company logos, brand marks, product logos
- banner: Email header/footer banners, promotional banners, decorative strips
- signature: Email signature images, contact cards, vCard images
- icon: Social media icons, small UI icons, button icons
- meaningful: Screenshots, diagrams, charts, photos of documents, whiteboard photos, \
or any image with substantive information

For meaningful images, provide a detailed description. For non-meaningful images, \
skip the description.

Respond with valid JSON matching this schema:
{
  "classification": "logo" | "banner" | "signature" | "icon" | "meaningful",
  "description": "string or null (only for meaningful images)",
  "reasoning": "brief explanation of why this classification was chosen"
}"""

_DESCRIBER_PROMPT = """\
You are an image analyst. Describe the image in detail for indexing and search purposes.

Include:
- Any visible text (transcribe it accurately)
- Diagrams, charts, or visual elements
- The overall context and purpose of the image
- Key information that would be useful for search and retrieval

Be thorough but concise.

Respond with valid JSON matching this schema:
{
  "description": "detailed description of the image content",
  "contains_text": true | false,
  "image_type": "screenshot" | "diagram" | "chart" | "photo" | "document" | etc.
}"""


class ImageProcessor:
    """Process images using LiteLLM with JSON mode for structured outputs.

    Uses acompletion() with response_format=json_object and Pydantic parsing for:
    - Classification (logo/banner/signature vs meaningful content)
    - Description generation
    """

    # Supported image MIME types for Vision API
    SUPPORTED_TYPES = {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/gif",
        "image/webp",
        "image/tiff",
        "image/bmp",
    }

    # Max image file size (10MB) to prevent memory issues
    MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

    def __init__(self):
        """Initialize the image processor."""
        settings = get_settings()
        self.model = settings.get_model(settings.image_llm_model)

    def is_supported(self, content_type: Optional[str]) -> bool:
        """Check if the image type is supported for Vision API."""
        return content_type in self.SUPPORTED_TYPES

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for API submission.

        Raises:
            ValueError: If image exceeds MAX_IMAGE_SIZE_BYTES.
        """
        file_size = image_path.stat().st_size
        if file_size > self.MAX_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"Image {image_path.name} exceeds max size "
                f"({file_size / 1024 / 1024:.1f}MB > {self.MAX_IMAGE_SIZE_BYTES / 1024 / 1024:.0f}MB)"
            )
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: Path) -> str:
        """Get MIME type from file extension."""
        ext = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_map.get(ext, "image/png")

    def _create_image_message(self, image_path: Path, text_prompt: str) -> dict:
        """Create a message with image input for the Chat Completions API."""
        base64_image = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
            ],
        }

    async def _call_vision(self, system_prompt: str, image_path: Path, user_text: str) -> str:
        """Make a vision API call and return the raw JSON content string."""
        from ..cli._common import _service_counter
        _service_counter.add("vision")

        message = self._create_image_message(image_path, user_text)
        response = await acompletion(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, message],
            response_format={"type": "json_object"},
            drop_params=True,
            extra_body=OPENROUTER_PRIVACY_EXTRA_BODY,
        )
        return response.choices[0].message.content

    async def classify_and_describe(self, image_path: Path) -> ImageAnalysisResult:
        """Classify an email image and describe if meaningful.

        Args:
            image_path: Path to the image file.

        Returns:
            ImageAnalysisResult with classification and optional description.
        """
        if not image_path.exists():
            return ImageAnalysisResult(
                classification=ImageClassification.MEANINGFUL,
                skip_reason="File not found",
            )

        try:
            content = await self._call_vision(
                _CLASSIFIER_PROMPT, image_path, "Analyze this image from an email."
            )
            parsed = ImageClassifyResponse.model_validate_json(content)

            classification = ImageClassification(parsed.classification)
            skip_reason = None
            if classification != ImageClassification.MEANINGFUL:
                skip_reason = (
                    f"Image classified as {classification.value}: {parsed.reasoning}"
                )

            return ImageAnalysisResult(
                classification=classification,
                description=parsed.description,
                skip_reason=skip_reason,
            )

        except Exception as e:
            logger.warning("Failed to classify image %s: %s", image_path, e)
            # On error, assume meaningful to avoid losing content
            return ImageAnalysisResult(
                classification=ImageClassification.MEANINGFUL,
                description=None,
                skip_reason=f"Classification failed: {e}",
            )

    async def describe_only(self, image_path: Path) -> Optional[str]:
        """Describe an image without classification.

        Args:
            image_path: Path to the image file.

        Returns:
            Description string or None on failure.
        """
        if not image_path.exists():
            return None

        try:
            content = await self._call_vision(
                _DESCRIBER_PROMPT, image_path, "Describe this image in detail."
            )
            parsed = ImageDescribeResponse.model_validate_json(content)

            # Enrich description with metadata
            prefix = f"[{parsed.image_type}]"
            if parsed.contains_text:
                prefix += " [contains text]"

            return f"{prefix}\n{parsed.description}"

        except Exception as e:
            logger.warning("Failed to describe image %s: %s", image_path, e)
            return None
