from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research-agent-utils")


@mcp.tool()
def word_count(text: str) -> str:
    """
    Count words and characters in a passage of text, and estimate
    reading time.

    Use this tool when the user asks how long a piece of text is,
    how many words or characters it has, or how long it would take
    to read.
    """
    words = text.split()
    word_total = len(words)
    char_total = len(text)
    reading_minutes = max(1, round(word_total / 200))

    return (
        f"Words: {word_total}\n"
        f"Characters: {char_total}\n"
        f"Estimated reading time: {reading_minutes} min (at 200 wpm)"
    )


@mcp.tool()
def format_citation(author: str, year: str, title: str, source: str, style: str = "apa") -> str:
    """
    Format a citation in APA or MLA style.

    Use this tool when the user asks to cite a source, format a
    reference, or build a bibliography entry.

    Args:
        author: Author name, "Last, First" format.
        year: Publication year.
        title: Title of the work.
        source: Where it was published (journal, website, publisher).
        style: "apa" or "mla". Defaults to "apa".
    """
    style = style.lower().strip()

    if style == "mla":
        return f'{author}. "{title}." {source}, {year}.'

    # default: apa
    return f"{author} ({year}). {title}. {source}."


if __name__ == "__main__":
    mcp.run(transport="stdio")