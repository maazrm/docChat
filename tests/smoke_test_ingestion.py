import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)

from ingestion.docling_parser import parse_pdf, get_markdown
from ingestion.chunker import chunk_document
from ingestion.image_captioner import caption_all_figures

doc_id = "google_env_2024"
pdf_path = "tests/sample_docs/google-2024-environmental-report.pdf"

print(f"Parsing {pdf_path}...")
doc = parse_pdf(pdf_path)

print(f"\nDocument: {len(doc.pages)} pages, {len(doc.tables)} tables, {len(doc.pictures)} figures")

# Markdown export preview
md = get_markdown(doc)
print(f"Markdown export: {len(md)} chars")
print(f"First 300 chars of markdown:\n{md[:300]}\n")

# Chunking
chunks = chunk_document(doc, doc_id)
print(f"Total text/table chunks: {len(chunks)}")

# Page distribution
page_counts = Counter(c["page"] for c in chunks)
print(f"Pages covered: {len(page_counts)} (pages {min(page_counts)}-{max(page_counts)})")

# Chunk type breakdown
type_counts = Counter(c["chunk_type"] for c in chunks)
print(f"Chunk types: {dict(type_counts)}")

# Sample chunks
print(f"\nFirst 5 chunks:")
for i, c in enumerate(chunks[:5]):
    text_preview = c["text"][:100].replace("\n", " ")
    print(f"  [{c['chunk_type']}] p{c['page']} | {c['section'] or '(no heading)'} | {text_preview}...")

# Section coverage
sections = set(c["section"] for c in chunks if c["section"])
print(f"\nDistinct sections found: {len(sections)}")
if sections:
    print(f"Sample sections: {list(sections)[:10]}")

# Figures
if doc.pictures:
    print(f"\nCapturing {len(doc.pictures)} figures...")
    fig_chunks = caption_all_figures(doc, doc_id)
    print(f"Captioned figures: {len(fig_chunks)}")
    for fc in fig_chunks[:3]:
        print(f"  [{fc['id']}] p{fc['page']}: {fc['text'][:120]}...")
else:
    print("\nNo figures found in document.")

print("\nSmoke test complete.")
