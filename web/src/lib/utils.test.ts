import { describe, expect, it } from "vitest";

import { applyCitationsToMarkdown, transformRawCitations } from "./utils";
import type { CitationsFrame } from "@/types/rag";

describe("applyCitationsToMarkdown", () => {
  it("swaps a marker for a fully-attributed <cite> tag when payload is provided", () => {
    const payload: CitationsFrame = {
      version: 1,
      citations: [
        {
          chunk_id: "abc012345678",
          index: 1,
          source_id: "incident-2024-08.eml",
          source_title: "Engine alarm thread",
          page: 5,
          lines: [12, 24],
          archive_browse_uri: "/archive/folder/email.eml.md",
          archive_download_uri: "/archive/folder/email.eml",
          archive_verified: true,
        },
      ],
      invalid_chunk_ids: [],
    };

    const out = applyCitationsToMarkdown("Status update [C:abc012345678].", payload);

    // Mirrors CitationProcessor.replace_citation_markers attribute set.
    expect(out).toContain(
      '<cite id="abc012345678" doc="incident-2024-08.eml" title="Engine alarm thread" page="5" lines="12-24" download="folder/email.eml">1</cite>',
    );
    // Surrounding prose must remain intact.
    expect(out).toContain("Status update ");
    expect(out).toContain(".");
  });

  it("strips markers whose chunk_id appears in invalid_chunk_ids", () => {
    const payload: CitationsFrame = {
      version: 1,
      citations: [],
      invalid_chunk_ids: ["badbadbadbad"],
    };

    const out = applyCitationsToMarkdown("Foo [C:badbadbadbad] bar", payload);

    // Marker stripped + double-space collapsed back to a single space.
    expect(out).toBe("Foo bar");
    expect(out).not.toContain("[C:badbadbadbad]");
  });

  it("falls back to transformRawCitations byte-for-byte when payload is null", () => {
    const input = "Foo [C:abc012345678]";
    expect(applyCitationsToMarkdown(input, null)).toBe(transformRawCitations(input));
  });
});
