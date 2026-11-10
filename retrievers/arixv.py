import arxiv

client = arxiv.Client()

search = arxiv.Search(
    query="large language models",
    max_results=2
)

for i, paper in enumerate(client.results(search)):
    print(f"\nResult {i+1}")
    print("Title:", paper.title)
    print("Authors:", ", ".join(author.name for author in paper.authors))
    print("Summary:", paper.summary[:500])
