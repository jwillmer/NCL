"""Tests for vessel matching functionality.

Tests for VesselMatcher class that finds vessel names
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
                vessel_type="VLCC",
                vessel_class="Nordic Class",
            ),
            Vessel(
                id=uuid4(),
                name="MT Ocean Queen",
                vessel_type="SUEZMAX",
                vessel_class="Ocean Class",
            ),
            Vessel(
                id=uuid4(),
                name="MARAN CANOPUS",
                vessel_type="VLCC",
                vessel_class="Canopus Class",
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
        result = vessel_matcher.find_vessels_in_email(
            subject="VHF FREQUENCIES Update",
            body="Check the FREQUENCIES list",
        )

        # Should not match because FREQUENCIES doesn't contain a vessel name
        assert len(result) == 0

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
        """Should report correct number of names."""
        # 3 vessels with 3 names
        assert vessel_matcher.name_count == 3

    @pytest.mark.unit
    def test_get_types_for_ids(self, vessel_matcher, sample_vessels):
        """Should return unique vessel types for matched IDs."""
        # Get IDs for Nordic Star (VLCC) and MARAN CANOPUS (VLCC)
        ids = {sample_vessels[0].id, sample_vessels[2].id}
        types = vessel_matcher.get_types_for_ids(ids)
        assert types == ["VLCC"]

    @pytest.mark.unit
    def test_get_types_for_ids_multiple(self, vessel_matcher, sample_vessels):
        """Should return all unique vessel types for mixed IDs."""
        # Get IDs for all three vessels
        ids = {sample_vessels[0].id, sample_vessels[1].id, sample_vessels[2].id}
        types = vessel_matcher.get_types_for_ids(ids)
        assert sorted(types) == ["SUEZMAX", "VLCC"]

    @pytest.mark.unit
    def test_get_classes_for_ids(self, vessel_matcher, sample_vessels):
        """Should return unique vessel classes for matched IDs."""
        ids = {sample_vessels[0].id}
        classes = vessel_matcher.get_classes_for_ids(ids)
        assert classes == ["Nordic Class"]

    @pytest.mark.unit
    def test_get_classes_for_ids_multiple(self, vessel_matcher, sample_vessels):
        """Should return all unique vessel classes for multiple IDs."""
        ids = {sample_vessels[0].id, sample_vessels[1].id}
        classes = vessel_matcher.get_classes_for_ids(ids)
        assert sorted(classes) == ["Nordic Class", "Ocean Class"]

    @pytest.mark.unit
    def test_get_types_for_empty_ids(self, vessel_matcher):
        """Should return empty list for empty ID set."""
        types = vessel_matcher.get_types_for_ids(set())
        assert types == []

    @pytest.mark.unit
    def test_get_classes_for_empty_ids(self, vessel_matcher):
        """Should return empty list for empty ID set."""
        classes = vessel_matcher.get_classes_for_ids(set())
        assert classes == []


class TestVesselMatcherEdgeCases:
    """Edge case tests for VesselMatcher."""

    @pytest.mark.unit
    def test_handles_special_characters_in_name(self):
        """Should handle vessel names with special characters."""
        from mtss.models.vessel import Vessel
        from mtss.processing.vessel_matcher import VesselMatcher

        vessels = [
            Vessel(name="M/V Test-Ship", vessel_type="VLCC", vessel_class="Test Class"),
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
            Vessel(name="  Trimmed Name  ", vessel_type="VLCC", vessel_class="Test Class"),
        ]
        matcher = VesselMatcher(vessels)

        # Names should be trimmed when building lookup
        result = matcher.find_vessels("Reporting on Trimmed Name status.")
        assert len(result) == 1

    @pytest.mark.unit
    def test_none_in_subject_or_body(self):
        """Should handle None values for subject or body."""
        from mtss.models.vessel import Vessel
        from mtss.processing.vessel_matcher import VesselMatcher

        vessels = [Vessel(name="Test Ship", vessel_type="VLCC", vessel_class="Test Class")]
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
