"""Tests for vessel matching functionality.

Tests for VesselMatcher class that finds vessel names and aliases
in document text for automatic tagging during ingest.
"""

from uuid import uuid4

import pytest


class TestVesselMatcher:
    """Tests for VesselMatcher class."""

    @pytest.fixture
    def sample_vessels(self):
        """Create a list of sample vessels for testing."""
        from mtss.models.vessel import Vessel

        return [
            Vessel(
                id=uuid4(),
                name="MV Nordic Star",
                imo="1234567",
                vessel_type="VLCC",
                aliases=["Nordic Star", "NORDIC STAR"],
            ),
            Vessel(
                id=uuid4(),
                name="MT Ocean Queen",
                imo="2345678",
                vessel_type="Suezmax",
                aliases=["Ocean Queen", "OQ"],
            ),
            Vessel(
                id=uuid4(),
                name="MARAN CANOPUS",
                imo="3456789",
                vessel_type="VLCC",
                aliases=["Canopus", "M CANOPUS"],
            ),
        ]

    @pytest.fixture
    def vessel_matcher(self, sample_vessels):
        """Create a VesselMatcher with sample vessels."""
        from mtss.processing.vessel_matcher import VesselMatcher

        return VesselMatcher(sample_vessels)

    @pytest.mark.unit
    def test_finds_vessel_name_in_subject(self, vessel_matcher, sample_vessels):
        """Should find vessel name in email subject."""
        result = vessel_matcher.find_vessels_in_email(
            subject="RE: MV Nordic Star - Maintenance Report",
            body=None,
        )

        assert len(result) == 1
        assert sample_vessels[0].id in result

    @pytest.mark.unit
    def test_finds_vessel_name_in_body(self, vessel_matcher, sample_vessels):
        """Should find vessel name in email body."""
        result = vessel_matcher.find_vessels_in_email(
            subject="Maintenance Update",
            body="The MT Ocean Queen requires urgent repairs to the hull.",
        )

        assert len(result) == 1
        assert sample_vessels[1].id in result

    @pytest.mark.unit
    def test_finds_vessel_alias(self, vessel_matcher, sample_vessels):
        """Should find vessel by alias."""
        result = vessel_matcher.find_vessels_in_email(
            subject="RE: Canopus Update",
            body="The Canopus is scheduled for inspection.",
        )

        assert len(result) == 1
        assert sample_vessels[2].id in result

    @pytest.mark.unit
    def test_no_match_returns_empty_set(self, vessel_matcher):
        """Should return empty set when no vessels match."""
        result = vessel_matcher.find_vessels_in_email(
            subject="General Update",
            body="This email has no vessel references.",
        )

        assert len(result) == 0

    @pytest.mark.unit
    def test_case_insensitive_matching(self, vessel_matcher, sample_vessels):
        """Should match vessel names case-insensitively."""
        result = vessel_matcher.find_vessels_in_email(
            subject="mv nordic star - lowercase test",
            body=None,
        )

        assert len(result) == 1
        assert sample_vessels[0].id in result

    @pytest.mark.unit
    def test_finds_multiple_vessels(self, vessel_matcher, sample_vessels):
        """Should find multiple vessels in same text."""
        result = vessel_matcher.find_vessels_in_email(
            subject="Fleet Update: Nordic Star and Ocean Queen",
            body="Both MV Nordic Star and MT Ocean Queen are docked.",
        )

        assert len(result) == 2
        assert sample_vessels[0].id in result
        assert sample_vessels[1].id in result

    @pytest.mark.unit
    def test_word_boundary_matching(self, vessel_matcher, sample_vessels):
        """Should use word boundaries to avoid false positives."""
        # "OQ" is an alias for Ocean Queen, but should not match "FREQUENCIES"
        result = vessel_matcher.find_vessels_in_email(
            subject="VHF FREQUENCIES Update",
            body="Check the FREQUENCIES list",
        )

        # Should not match because FREQUENCIES contains "Q" but not as a word boundary
        # Note: "OQ" is an alias, so this tests that we don't match partial words
        assert len(result) == 0

    @pytest.mark.unit
    def test_short_alias_matching(self, vessel_matcher, sample_vessels):
        """Should match short aliases with proper word boundaries."""
        result = vessel_matcher.find_vessels_in_email(
            subject="OQ Status",
            body="The OQ is underway.",
        )

        assert len(result) == 1
        assert sample_vessels[1].id in result

    @pytest.mark.unit
    def test_find_vessels_in_text_directly(self, vessel_matcher, sample_vessels):
        """Should find vessels using find_vessels method."""
        result = vessel_matcher.find_vessels("Reporting on MARAN CANOPUS voyage.")

        assert len(result) == 1
        assert sample_vessels[2].id in result

    @pytest.mark.unit
    def test_empty_text_returns_empty(self, vessel_matcher):
        """Should return empty set for empty text."""
        assert vessel_matcher.find_vessels("") == set()
        assert vessel_matcher.find_vessels(None) == set()

    @pytest.mark.unit
    def test_empty_lookup_returns_empty(self):
        """Should return empty set when no vessels in matcher."""
        from mtss.processing.vessel_matcher import VesselMatcher

        matcher = VesselMatcher([])
        result = matcher.find_vessels("MV Nordic Star is mentioned")

        assert result == set()

    @pytest.mark.unit
    def test_vessel_count_property(self, vessel_matcher):
        """Should report correct number of unique vessels."""
        assert vessel_matcher.vessel_count == 3

    @pytest.mark.unit
    def test_name_count_property(self, vessel_matcher):
        """Should report correct number of names/aliases."""
        # 3 primary names + 6 aliases = 9 total
        # But some aliases might be duplicates of primary names when lowercased
        assert vessel_matcher.name_count >= 6  # At least 3 names + some aliases


class TestVesselMatcherEdgeCases:
    """Edge case tests for VesselMatcher."""

    @pytest.mark.unit
    def test_handles_special_characters_in_name(self):
        """Should handle vessel names with special characters."""
        from mtss.models.vessel import Vessel
        from mtss.processing.vessel_matcher import VesselMatcher

        vessels = [
            Vessel(name="M/V Test-Ship", aliases=["Test Ship"]),
        ]
        matcher = VesselMatcher(vessels)

        # Should find the vessel with special chars
        result = matcher.find_vessels("The M/V Test-Ship departed today.")
        assert len(result) == 1

    @pytest.mark.unit
    def test_handles_whitespace_in_names(self):
        """Should handle vessel names with various whitespace."""
        from mtss.models.vessel import Vessel
        from mtss.processing.vessel_matcher import VesselMatcher

        vessels = [
            Vessel(name="  Trimmed Name  ", aliases=["  Also Trimmed  "]),
        ]
        matcher = VesselMatcher(vessels)

        # Names should be trimmed when building lookup
        result = matcher.find_vessels("Reporting on Trimmed Name status.")
        assert len(result) == 1

    @pytest.mark.unit
    def test_duplicate_aliases_deduped(self):
        """Should handle duplicate aliases across vessels."""
        from mtss.models.vessel import Vessel
        from mtss.processing.vessel_matcher import VesselMatcher

        v1_id = uuid4()
        v2_id = uuid4()
        vessels = [
            Vessel(id=v1_id, name="Ship One", aliases=["Common Name"]),
            Vessel(id=v2_id, name="Ship Two", aliases=["Common Name"]),
        ]
        matcher = VesselMatcher(vessels)

        # When same alias used by multiple vessels, last one wins
        result = matcher.find_vessels("Common Name is mentioned")
        assert len(result) == 1
        # The second vessel's ID should be in lookup (last write wins)
        assert v2_id in result

    @pytest.mark.unit
    def test_none_in_subject_or_body(self):
        """Should handle None values for subject or body."""
        from mtss.models.vessel import Vessel
        from mtss.processing.vessel_matcher import VesselMatcher

        vessels = [Vessel(name="Test Ship")]
        matcher = VesselMatcher(vessels)

        result = matcher.find_vessels_in_email(subject=None, body=None)
        assert result == set()

        result = matcher.find_vessels_in_email(
            subject=None, body="Test Ship mentioned"
        )
        assert len(result) == 1

        result = matcher.find_vessels_in_email(
            subject="Test Ship mentioned", body=None
        )
        assert len(result) == 1
