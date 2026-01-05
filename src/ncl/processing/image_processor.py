"""Image processor using OpenAI Agents SDK for classification and description."""

from __future__ import annotations

import base64
import logging
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from agents import Agent, Runner
from pydantic import BaseModel, Field

from ..config import get_settings

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


class ImageProcessor:
    """Process images using OpenAI Agents SDK with structured outputs.

    Uses the Agents SDK with Pydantic models to guarantee valid responses for:
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
        """Initialize the image processor with agents."""
        self.settings = get_settings()

        # Get model for image processing
        image_model = self.settings.get_model(self.settings.image_llm_model)

        # Agent for classifying email images
        self._classifier_agent = Agent(
            name="Image Classifier",
            instructions="""You are an image classifier for email attachments.
Your job is to determine if an image contains meaningful content or is just decoration.

Classify images as:
- logo: Company logos, brand marks, product logos
- banner: Email header/footer banners, promotional banners, decorative strips
- signature: Email signature images, contact cards, vCard images
- icon: Social media icons, small UI icons, button icons
- meaningful: Screenshots, diagrams, charts, photos of documents, whiteboard photos, or any image with substantive information

For meaningful images, provide a detailed description. For non-meaningful images, skip the description.""",
            model=image_model,
            output_type=ImageClassifyResponse,
        )

        # Agent for describing images (no classification)
        self._describer_agent = Agent(
            name="Image Describer",
            instructions="""You are an image analyst. Describe the image in detail for indexing and search purposes.

Include:
- Any visible text (transcribe it accurately)
- Diagrams, charts, or visual elements
- The overall context and purpose of the image
- Key information that would be useful for search and retrieval

Be thorough but concise.""",
            model=image_model,
            output_type=ImageDescribeResponse,
        )

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
        """Create a message with image input for the Responses API.

        The OpenAI Responses API (used by Agents SDK) expects:
        - type: "input_text" for text content
        - type: "input_image" with "image_url" for images
        """
        base64_image = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)
        return {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text_prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{base64_image}",
                },
            ],
        }

    async def classify_and_describe(self, image_path: Path) -> ImageAnalysisResult:
        """Classify an email image and describe if meaningful.

        Uses OpenAI Agents SDK with structured outputs.

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
            # Create message with image in Responses API format
            message = self._create_image_message(
                image_path, "Analyze this image from an email."
            )

            # Run the classifier agent with image input
            result = await Runner.run(
                self._classifier_agent,
                [message],
            )

            # Get the structured output
            parsed = result.final_output_as(ImageClassifyResponse)

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
            logger.warning(f"Failed to classify image {image_path}: {e}")
            # On error, assume meaningful to avoid losing content
            return ImageAnalysisResult(
                classification=ImageClassification.MEANINGFUL,
                description=None,
                skip_reason=f"Classification failed: {e}",
            )

    async def describe_only(self, image_path: Path) -> Optional[str]:
        """Describe an image without classification.

        Uses OpenAI Agents SDK with structured outputs.

        Args:
            image_path: Path to the image file.

        Returns:
            Description string or None on failure.
        """
        if not image_path.exists():
            return None

        try:
            # Create message with image in Responses API format
            message = self._create_image_message(
                image_path, "Describe this image in detail."
            )

            # Run the describer agent with image input
            result = await Runner.run(
                self._describer_agent,
                [message],
            )

            # Get the structured output
            parsed = result.final_output_as(ImageDescribeResponse)

            # Enrich description with metadata
            prefix = f"[{parsed.image_type}]"
            if parsed.contains_text:
                prefix += " [contains text]"

            return f"{prefix}\n{parsed.description}"

        except Exception as e:
            logger.warning(f"Failed to describe image {image_path}: {e}")
            return None
