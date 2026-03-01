import json
import webbrowser
import urllib.parse
import collections
from slugify import slugify
import re

from evaluation.plot_style import CMAP_PRIMARY, apply_plot_style


class BookCollector:
    def __init__(self, dataset, book_dir, progress_file_path):
        self.dataset = dataset
        self.book_count = len(dataset["train"])
        self.year_re = re.compile(r"\d{4}")
        self.book_dir = book_dir
        self.progress_file_path = progress_file_path
        self.current_index = 0
        self.processed_books = []
        self.nonskipped = 0
        self._load_progress()
        (self.book_dir / "meta").mkdir(exist_ok=True)

    def _load_progress(self):
        self.book_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.progress_file_path, "r") as f:
                self.processed_books = json.load(f)
                self.current_index = len(self.processed_books)
                self.nonskipped = sum(
                    1 for book in self.processed_books if book["action"] == "saved"
                )
        except FileNotFoundError:
            self.processed_books = []

    def get_middle_lines(self, text, num_lines=30):
        lines = text.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return ""
        start = max(0, (total_lines // 2) - (num_lines // 2))
        end = min(total_lines, start + num_lines)
        middle_lines = lines[start:end]
        return "\n".join(middle_lines)

    def get_current_book(self):
        if self.current_index >= self.book_count:
            return None, None, "No more books to process", ""
        book = self.dataset["train"][self.current_index]
        meta = json.loads(book["METADATA"])
        title = meta.get("title", "Unknown")
        authors = meta.get("authors", "Unknown")
        info = f"""
        **Book {self.current_index + 1} of {self.book_count} ({self.nonskipped} saved)**

        **Title:** {title}

        **Authors:** {authors}

        **Original Issue Info:** {meta.get("issued", "N/A")}

        **Subjects:** {meta.get("subjects", "N/A")}

        **Bookshelves:** {meta.get("bookshelves", "N/A")}

        **Length:** {len(book["TEXT"]):,} characters
        """
        middle_context = self.get_middle_lines(book["TEXT"], 30)
        return book, meta, info, middle_context

    def save_book(self, book, meta, fact_checked_year, genre, title, authors):
        slug = slugify(title)
        with open(self.book_dir / f"{slug}.txt", "w", encoding="utf-8") as f:
            f.write(book["TEXT"])
        book_meta = {
            "id": meta["text_id"],
            "title": title,
            "authors": authors,
            "year": int(fact_checked_year) if fact_checked_year else None,
            "genre": genre,
            "slug": slug,
            "length": len(book["TEXT"]),
            "original_metadata": meta,
        }
        with open(self.book_dir / "meta" / f"{slug}.json", "w", encoding="utf-8") as f:
            json.dump(book_meta, f, indent=2)
        return slug

    def google_search(self, title, authors):
        query = f'"{title}" "{authors}" original publish date'
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        webbrowser.open(search_url)
        return f"Opened Google search for: {query}"

    def process_book(self, action, fact_checked_year, genre, title, authors):
        if self.current_index >= self.book_count:
            return "No more books to process", "", "", "", "", "", "Completed"
        book, meta, _, _ = self.get_current_book()
        status = "Error"
        if not meta:
            return "No metadata found for current book", "", "", "", "", "", "Error"
        if action == "save":
            if not fact_checked_year or not genre:
                return (
                    "Please fill in both publish date and genre",
                    fact_checked_year,
                    genre,
                    title,
                    authors,
                    "Error",
                )
            try:
                slug = self.save_book(
                    book, meta, fact_checked_year, genre, title, authors
                )
                self.processed_books.append(
                    {
                        "index": self.current_index,
                        "book_id": meta["text_id"],
                        "action": "saved",
                        "slug": slug,
                    }
                )
                status = f"Saved: {slug}"
                self.nonskipped += 1
            except Exception as e:
                return f"Error saving book: {str(e)}", "", "", "", "", "", "Error"
        elif action == "skip":
            self.processed_books.append(
                {
                    "index": self.current_index,
                    "book_id": meta["text_id"],
                    "action": "skipped",
                }
            )
            status = "Skipped book"
        with open(self.progress_file_path, "w") as f:
            json.dump(self.processed_books, f, indent=2)
        self.current_index += 1
        if self.current_index >= self.book_count:
            return "All books processed!", "", "", "", "", "", "Completed"
        next_book, next_meta, next_info, next_context = self.get_current_book()
        if not next_meta:
            return "No metadata found for next book", "", "", "", "", "", "Error"
        next_title = next_meta.get("title", "Unknown")
        next_authors = next_meta.get("authors", "Unknown")
        suggested_year = self.year_re.findall(next_meta.get("issued", ""))
        next_year = suggested_year[0] if suggested_year else ""
        return (
            next_info,
            next_context,
            next_year,
            next_title,
            next_authors,
            None,
            status,
        )

    def launch_interface(self):
        import gradio as gr

        initial_book, initial_meta, initial_info, initial_context = (
            self.get_current_book()
        )
        initial_title = initial_meta.get("title", "Unknown") if initial_meta else ""
        initial_authors = initial_meta.get("authors", "Unknown") if initial_meta else ""
        initial_year = ""
        if initial_meta:
            suggested_years = self.year_re.findall(initial_meta.get("issued", ""))
            initial_year = suggested_years[0] if suggested_years else ""

        with gr.Blocks(title="Book Metadata Fact-Checker") as interface:
            with gr.Row():
                with gr.Column(scale=2):
                    book_info = gr.Markdown(initial_info)
                    book_context = gr.Textbox(
                        value=initial_context,
                        label="Middle 30 Lines of Book",
                        lines=30,
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    title_input = gr.Textbox(
                        label="Title", value=initial_title, placeholder="Book title"
                    )
                    authors_input = gr.Textbox(
                        label="Authors",
                        value=initial_authors,
                        placeholder="Author names",
                    )
                    year_input = gr.Textbox(
                        label="Fact-checked Publish Date",
                        value=initial_year,
                        placeholder="YYYY",
                    )
                    genre_input = gr.Dropdown(
                        label="Genre",
                        choices=[
                            "Fiction",
                            "Fantasy",
                            "Science Fiction",
                            "Mystery",
                            "Western",
                            "History",
                            "Travel",
                        ],
                        value=None,
                        allow_custom_value=False,
                    )

                    google_btn = gr.Button("🔍 Google Search", variant="secondary")

                    with gr.Row():
                        save_btn = gr.Button("Save Book", variant="primary")
                        skip_btn = gr.Button("Skip Book", variant="secondary")

                    status = gr.Textbox(label="Status", interactive=False)

            # Event handlers
            google_btn.click(
                fn=self.google_search,
                inputs=[title_input, authors_input],
                outputs=[status],
            )

            save_btn.click(
                fn=lambda year, genre, title, authors: self.process_book(
                    "save", year, genre, title, authors
                ),
                inputs=[year_input, genre_input, title_input, authors_input],
                outputs=[
                    book_info,
                    book_context,
                    year_input,
                    title_input,
                    authors_input,
                    genre_input,
                    status,
                ],
            )

            skip_btn.click(
                fn=lambda year, genre, title, authors: self.process_book(
                    "skip", year, genre, title, authors
                ),
                inputs=[year_input, genre_input, title_input, authors_input],
                outputs=[
                    book_info,
                    book_context,
                    year_input,
                    title_input,
                    authors_input,
                    genre_input,
                    status,
                ],
            )

            interface.launch(share=False, debug=True)


def plot_genre_distribution() -> None:
    """Plot a pie chart showing the distribution of genres in the book collection."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import seaborn as sns
    from utils import BOOK_META_DIR, DATA_DIR

    apply_plot_style()

    genre_counts: collections.Counter = collections.Counter()
    for meta_file in BOOK_META_DIR.glob("*.json"):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            g = data.get("genre") or "Unknown"
            if isinstance(g, str):
                g = g.strip() or "Unknown"
            else:
                g = "Unknown"
            if g.lower() == "science fiction":
                g = "Sci-Fi"
            genre_counts[g] += 1
        except Exception:
            pass

    if not genre_counts:
        print("No genre data found for Matplotlib chart.")
        return

    genres, counts = zip(*genre_counts.most_common())
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(22, 13), dpi=120)

    def autopct_fmt(pct):
        return ("%0.1f%%" % pct) if pct >= 2 else ""

    colors = sns.color_palette(CMAP_PRIMARY, len(genres))
    wedges, _, autotexts = ax.pie(
        counts,
        colors=colors,
        labels=None,
        autopct=autopct_fmt,
        startangle=90,
        counterclock=False,
        pctdistance=0.8,
        textprops={"color": "#000", "fontsize": 38, "fontweight": "bold"},
        wedgeprops={"linewidth": 2.5, "edgecolor": "white"},
    )

    for t in autotexts:
        t.set_fontsize(38)
        t.set_fontweight("bold")

    legend_patches = [
        Patch(facecolor=c, edgecolor="white", label=f"{g}: {cnt}")
        for c, g, cnt in zip(colors, genres, counts)
    ]

    leg = ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        frameon=True,
        title_fontsize=44,
        borderpad=0.1,
        labelspacing=0.6,
        handlelength=2.0,
        handleheight=1.5,
        title="Genre: Book Count",
        fontsize=42,
        markerscale=1.1,
    )
    frame = leg.get_frame()
    frame.set_alpha(0.18)
    frame.set_edgecolor("#bcbcbc")
    frame.set_linewidth(1.2)
    frame.set_facecolor("#ffffff")

    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    plt.axis("off")
    pie_path = DATA_DIR / "figures" / "genre_distribution.png"
    fig.savefig(
        pie_path,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    print("Saved pie to", pie_path)

    try:
        from PIL import Image

        with Image.open(pie_path) as im:
            im = im.convert("RGBA")
            bbox = im.getbbox()
            if bbox:
                cropped = im.crop(bbox)
                cropped.save(pie_path)
                print("Cropped margins from saved image.")
            else:
                print("No non-transparent content found to crop.")
    except Exception as e:
        print(f"Image cropping failed: {e}")

    plt.show()
    print("Total books counted:", total)
    print("Done.")
